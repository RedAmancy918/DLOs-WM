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
    ACTION_DIM,
    EDGE_FEAT_DIM,
    MATERIAL_FEATURE_DIM,
    NODE_FEAT_DIM,
    POS_DIM,
    compute_edge_features,
    node_contact_from_edges,
)


# MaterialCondition.global_features() 的既定顺序：
# [log_K, log_E, log_G, radius, linear_density,
#  mu_self_static, mu_self_kinetic]。模型直接消费已变换特征，不重复取 log。
# 数据层也可以通过构造参数传入别的维度，避免把 schema 绑定在模型实现里。
MATERIAL_INPUT_DIM = MATERIAL_FEATURE_DIM


def _with_derived_contact(state, edge_index, is_contact):
    """构造当前 forward 输入，contact 与同一步动态边保持一致。"""
    from ..data.schema import DLOState

    return DLOState(
        pos=state.pos,
        vel=state.vel,
        tension=state.tension,
        contact=node_contact_from_edges(
            state.num_nodes,
            edge_index,
            is_contact,
            dtype=state.pos.dtype,
        ),
        topology=state.topology,
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


class MaterialEncoder(nn.Module):
    """把 episode 级、已完成数值变换的材料特征编码为图条件向量。"""

    def __init__(self, input_dim=MATERIAL_INPUT_DIM, hidden=128):
        super().__init__()
        self.input_dim = input_dim
        self.hidden = hidden
        self.net = mlp([input_dim, hidden, hidden])

    def forward(self, material_features):
        """
        material_features: [M]（单图）或 [B, M]（批图）。
        保留前导维度，只把最后一维编码为 hidden。
        """
        if material_features.ndim not in (1, 2):
            raise ValueError(
                "material_features 必须是 [M] 或 [B, M]，"
                f"实际 shape={tuple(material_features.shape)}"
            )
        if material_features.shape[-1] != self.input_dim:
            raise ValueError(
                f"材料特征维度应为 {self.input_dim}，"
                f"实际为 {material_features.shape[-1]}"
            )
        return self.net(material_features)


class ConditionedInteractionLayer(nn.Module):
    """每轮都注入材料 latent 的 message passing 层。"""

    def __init__(self, hidden):
        super().__init__()
        # 边输入: [e_ij, h_i, h_j, material_i]
        self.edge_mlp = mlp([hidden * 4, hidden, hidden])
        # 节点输入: [h_i, aggregated_msg, action_drive, material_i]
        self.node_mlp = mlp([hidden * 4, hidden, hidden])
        self.drive_proj = nn.Linear(ACTION_DIM, hidden)

    def forward(self, h, e, edge_index, drive, material_node):
        """
        material_node: [N, hidden]，单图时由同一个材料 latent 广播，
                       批图时按 batch_idx 从 [B, hidden] 广播。
        """
        if material_node.shape != h.shape:
            raise ValueError(
                "material_node 必须与节点 latent 同 shape，"
                f"实际为 {tuple(material_node.shape)} 与 {tuple(h.shape)}"
            )

        src, dst = edge_index[0], edge_index[1]
        edge_in = torch.cat(
            [e, h[src], h[dst], material_node[src]], dim=-1
        )
        e_new = e + self.edge_mlp(edge_in)

        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, e_new)

        d = self.drive_proj(drive)
        node_in = torch.cat([h, agg, d, material_node], dim=-1)
        h_new = h + self.node_mlp(node_in)
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
            model_state = _with_derived_contact(
                state, edge_index, is_contact
            )
            drive = action.to_node_drive(
                state.num_nodes, current_pos=state.pos
            )
            out = self.forward(
                model_state, drive, edge_index, is_contact
            )
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


class MaterialConditionedDLOWorldModel(nn.Module):
    """
    材料条件化的单图世界模型。

    该类与 ``DLOWorldModel`` 完全独立，因而不会改变旧模型参数名、
    forward 签名或 checkpoint。材料条件既进入初始节点/边编码，也进入
    每一轮 message passing，使网络不能在 processor 中忽略物理条件。
    """

    def __init__(self, hidden=128, n_message_passing=6,
                 n_topo_classes=3, dt=0.04,
                 material_input_dim=MATERIAL_INPUT_DIM):
        super().__init__()
        self.hidden = hidden
        self.dt = dt
        self.n_topo_classes = n_topo_classes
        self.material_input_dim = material_input_dim

        self.material_enc = MaterialEncoder(material_input_dim, hidden)
        self.node_enc = mlp([NODE_FEAT_DIM + hidden, hidden, hidden])
        self.edge_enc = mlp([EDGE_FEAT_DIM + hidden, hidden, hidden])
        self.layers = nn.ModuleList(
            [ConditionedInteractionLayer(hidden)
             for _ in range(n_message_passing)]
        )

        self.acc_head = mlp([hidden, hidden, POS_DIM])
        self.tension_head = mlp([hidden, hidden, 1])
        self.contact_head = mlp([hidden, hidden, 1])
        self.topo_head = mlp([hidden, hidden, n_topo_classes])
        self.fail_head = mlp([hidden, hidden, 1])

    def _material_context(self, material_features, state, edge_index):
        """编码单图材料，并广播到节点和边。"""
        if material_features.ndim != 1:
            raise ValueError(
                "单图 material_features 必须是 [M]，"
                f"实际 shape={tuple(material_features.shape)}"
            )
        material_features = material_features.to(
            device=state.pos.device, dtype=state.pos.dtype
        )
        material_graph = self.material_enc(material_features)  # [hidden]
        material_node = material_graph.unsqueeze(0).expand(
            state.num_nodes, -1
        )
        material_edge = material_node[edge_index[0]]
        return material_node, material_edge

    def forward(self, state, drive, edge_index, is_contact,
                material_features):
        """
        单图前向。

        material_features: [M]，同一 episode 内保持不变的材料条件。
        """
        x = state.node_features()
        e_feat = compute_edge_features(state.pos, edge_index, is_contact)
        material_node, material_edge = self._material_context(
            material_features, state, edge_index
        )

        h = self.node_enc(torch.cat([x, material_node], dim=-1))
        e = self.edge_enc(torch.cat([e_feat, material_edge], dim=-1))
        for layer in self.layers:
            h, e = layer(
                h, e, edge_index, drive, material_node
            )

        acc = self.acc_head(h)
        tension = torch.nn.functional.softplus(
            self.tension_head(h)
        ).squeeze(-1)
        contact_logit = self.contact_head(h).squeeze(-1)

        g = h.mean(dim=0, keepdim=True)
        topo_logits = self.topo_head(g).squeeze(0)
        fail_logit = self.fail_head(g).squeeze()

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
    def rollout(self, init_state, actions, edge_builder,
                material_features):
        """
        材料条件化闭环 rollout。

        材料在整条 episode 中保持不变；边仍基于每一步预测位置动态重建，
        避免接触几何在多步预测中变旧。
        """
        from ..data.schema import DLOState

        traj = [init_state]
        state = init_state
        for action in actions:
            edge_index, is_contact = edge_builder(state.pos)
            model_state = _with_derived_contact(
                state, edge_index, is_contact
            )
            # target_pos 类型动作必须相对模型自己的预测位置渲染；若继续复用
            # 数据集里基于 GT state 算好的 delta，会在闭环评估中泄漏真值反馈。
            drive = action.to_node_drive(
                state.num_nodes, current_pos=state.pos
            )
            out = self.forward(
                model_state, drive, edge_index, is_contact,
                material_features,
            )
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
