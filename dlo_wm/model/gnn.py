"""
GNN DLO World Model —— 主干网络（纯手写 message passing，无 torch_geometric）

架构沿用 Graph Network Simulator（GNS / DPI-Net 一脉）的 encode-process-decode：

  Encoder:  节点特征、边特征 各自 MLP 升维到 latent
  Processor: M 轮 message passing。每轮：
       edge update:  e_ij' = MLP_e([ e_ij, h_i, h_j ])
       node update:  h_i'  = MLP_n([ h_i, sum_j e_ij', u_i ])    (u_i = 动作驱动)
       残差连接，稳定深层传播
  Decoder:  从最终节点 latent 解码出多个物理 head

多 head（这正是你强调的"不要只预测图像"）：
  - acc        [N,3]  节点加速度 -> 积分得到 pos_{t+1}, vel_{t+1}
  - tension    [N]    下一帧张力场
  - contact    [N]    下一帧接触概率（logit）
  - topology   [C]    图级拓扑分类（对节点 latent 做 pooling）
  - failure    [1]    图级失败风险（logit）

failure 设计上不是独立学的"凭空一个数"，而是被监督成
"张力是否超限 或 拓扑是否进入卡死类"的函数（见 train/losses.py 里如何造 label），
对应你说的"failure risk 应是其他量的函数"。
"""

from __future__ import annotations
import torch
import torch.nn as nn

from ..data.schema import (
    NODE_FEAT_DIM, EDGE_FEAT_DIM, ACTION_DIM, POS_DIM,
    compute_edge_features,
)


def mlp(sizes, act=nn.SiLU, last_act=False):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or last_act:
            layers.append(act())
    return nn.Sequential(*layers)


class InteractionLayer(nn.Module):
    """一轮 message passing：先更新边，再聚合更新节点。带残差。"""

    def __init__(self, hidden):
        super().__init__()
        self.edge_mlp = mlp([hidden * 3, hidden, hidden])
        # 节点输入: [h_i, aggregated_msg, action_drive_proj]
        self.node_mlp = mlp([hidden * 2 + hidden, hidden, hidden])
        self.drive_proj = nn.Linear(ACTION_DIM, hidden)

    def forward(self, h, e, edge_index, drive):
        src, dst = edge_index[0], edge_index[1]
        # ---- edge update ----
        edge_in = torch.cat([e, h[src], h[dst]], dim=-1)
        e_new = e + self.edge_mlp(edge_in)            # 残差
        # ---- aggregate messages to dst nodes ----
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, e_new)
        # ---- node update ----
        d = self.drive_proj(drive)
        node_in = torch.cat([h, agg, d], dim=-1)
        h_new = h + self.node_mlp(node_in)            # 残差
        return h_new, e_new


class DLOWorldModel(nn.Module):
    def __init__(self, hidden=128, n_message_passing=6, n_topo_classes=3, dt=0.04):
        super().__init__()
        self.hidden = hidden
        self.dt = dt
        self.n_topo_classes = n_topo_classes

        # ---- encoders ----
        self.node_enc = mlp([NODE_FEAT_DIM, hidden, hidden])
        self.edge_enc = mlp([EDGE_FEAT_DIM, hidden, hidden])

        # ---- processor ----
        self.layers = nn.ModuleList(
            [InteractionLayer(hidden) for _ in range(n_message_passing)]
        )

        # ---- decoders (multi-head) ----
        self.acc_head     = mlp([hidden, hidden, POS_DIM])   # 节点加速度
        self.tension_head = mlp([hidden, hidden, 1])         # 张力 (>=0, softplus 见下)
        self.contact_head = mlp([hidden, hidden, 1])         # 接触 logit
        self.topo_head    = mlp([hidden, hidden, n_topo_classes])  # 作用于 pooled
        self.fail_head    = mlp([hidden, hidden, 1])         # 作用于 pooled

    def forward(self, state, drive, edge_index, is_contact):
        """
        单图前向。
        state: DLOState (当前帧)
        drive: [N,3] 动作驱动信号
        edge_index: [2,E]，is_contact: [E]
        返回 dict of predictions。
        """
        x = state.node_features()                       # [N, NODE_FEAT_DIM]
        e_feat = compute_edge_features(state.pos, edge_index, is_contact)

        h = self.node_enc(x)
        e = self.edge_enc(e_feat)
        for layer in self.layers:
            h, e = layer(h, e, edge_index, drive)

        # 节点级 head
        acc = self.acc_head(h)                          # [N,3]
        tension = torch.nn.functional.softplus(self.tension_head(h)).squeeze(-1)  # [N] >=0
        contact_logit = self.contact_head(h).squeeze(-1)  # [N]

        # 图级 head：mean-pool 节点 latent
        g = h.mean(dim=0, keepdim=True)                 # [1, hidden]
        topo_logits = self.topo_head(g).squeeze(0)      # [C]
        fail_logit = self.fail_head(g).squeeze()        # scalar

        # 由加速度积分出下一帧 pos / vel（半隐式）
        vel_next = state.vel + self.dt * acc
        pos_next = state.pos + self.dt * vel_next

        return {
            "acc": acc,
            "pos_next": pos_next,
            "vel_next": vel_next,
            "tension": tension,
            "contact_logit": contact_logit,
            "topo_logits": topo_logits,
            "fail_logit": fail_logit,
        }

    @torch.no_grad()
    def rollout(self, init_state, actions, edge_builder):
        """
        闭环多步 rollout：用自己的预测当作下一步输入，预测一整段未来。
        这正是"学习型物理模拟器 / 预测动作后果"的用法。

        init_state: DLOState
        actions: list[DLOAction]
        edge_builder: callable(pos)-> (edge_index, is_contact)
                      —— 每步根据预测位置重建接触边（接触是会变的）
        返回 list[DLOState]（预测轨迹，长度 len(actions)+1，含初始帧）
        """
        from ..data.schema import DLOState
        traj = [init_state]
        state = init_state
        for action in actions:
            edge_index, is_contact = edge_builder(state.pos)
            drive = action.to_node_drive(state.num_nodes)
            out = self.forward(state, drive, edge_index, is_contact)
            topo = out["topo_logits"].argmax().long()
            state = DLOState(
                pos=out["pos_next"],
                vel=out["vel_next"],
                tension=out["tension"],
                contact=(out["contact_logit"] > 0).float(),
                topology=topo,
            )
            traj.append(state)
        return traj
