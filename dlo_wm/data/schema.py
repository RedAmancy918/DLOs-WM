"""
DLO World Model —— 数据 schema / 张量约定

这是整个工程的"契约"。所有数据来源（合成生成器、你自己的仿真器、真实重建）
都必须把数据整理成这里定义的格式，模型才能消费。先把这个看懂，
其余代码都是围绕它转的。

================================================================
一条 DLO（绳/线/线缆）被离散成 N 个 centerline 节点，沿弧长排列。
图结构：
  - 节点 i ：centerline 上第 i 个点
  - 结构边 (i, i+1) ：相邻节点间的弹性段（永远存在，编码弯曲/拉伸）
  - 接触边 (i, j) ：当两段非相邻的绳互相靠近 / 自接触时动态出现

一帧的"状态" State 包含：
  pos      [N, 3]   节点 3D 坐标（centerline 几何）
  vel      [N, 3]   节点速度
  tension  [N]      沿弧长的标量张力场（定义在节点上；也可定义在边上，这里用节点）
  contact  [N]      每个节点是否处于接触（0/1，软标签可用概率）
  topology int      拓扑类别 id（如 0=unknot, 1=trefoil, ...）或 crossing 数

一个"动作" Action（双臂抓取-移动是 DLO 操作最常见的参数化）：
  grasp_idx  [G]    被抓取的节点索引（G 个抓手，通常 1~2）
  delta      [G,3]  每个抓手在该步的位移
我们把 action 渲染成每个节点的外部驱动信号 u [N, 3]，喂给 GNN。

一个"转移样本" Transition = (State_t, Action_t, State_{t+1})。
训练时模型学 f(State_t, Action_t) -> 预测 State_{t+1} 的各分量。
================================================================
"""

from dataclasses import dataclass
import torch


# ------- 各分量维度，集中放这里，全工程引用，避免魔法数字散落 -------
POS_DIM = 3
VEL_DIM = 3
# 节点特征 = pos(3) + vel(3) + tension(1) + contact(1) = 8
NODE_FEAT_DIM = POS_DIM + VEL_DIM + 1 + 1
# 边特征（结构边）：相对位移(3) + 距离(1) + 是否接触边(1) = 5
EDGE_FEAT_DIM = 3 + 1 + 1
# 动作驱动信号维度（每节点外力/位移）
ACTION_DIM = 3


@dataclass
class DLOState:
    """单帧 DLO 状态。所有张量第一维是 N（节点数）。"""
    pos: torch.Tensor       # [N, 3]
    vel: torch.Tensor       # [N, 3]
    tension: torch.Tensor   # [N]
    contact: torch.Tensor   # [N]  0/1 or prob
    topology: torch.Tensor  # scalar long tensor, 拓扑类别 id

    @property
    def num_nodes(self) -> int:
        return self.pos.shape[0]

    def node_features(self) -> torch.Tensor:
        """拼成 GNN 输入的节点特征 [N, NODE_FEAT_DIM]。"""
        return torch.cat(
            [
                self.pos,
                self.vel,
                self.tension.unsqueeze(-1),
                self.contact.unsqueeze(-1),
            ],
            dim=-1,
        )

    def to(self, device):
        return DLOState(
            self.pos.to(device),
            self.vel.to(device),
            self.tension.to(device),
            self.contact.to(device),
            self.topology.to(device),
        )


@dataclass
class DLOAction:
    """单步动作：G 个抓手各自的抓取点与位移。"""
    grasp_idx: torch.Tensor  # [G] long
    delta: torch.Tensor      # [G, 3]

    def to_node_drive(self, num_nodes: int) -> torch.Tensor:
        """
        把抓手动作"散射"成每个节点的驱动信号 u [N, 3]。
        被抓节点拿到对应 delta，其余为 0。
        （更真实的做法可以按弧长距离做高斯衰减，这里先用硬赋值，
         留在 TODO 里方便你替换。）
        """
        u = torch.zeros(num_nodes, ACTION_DIM,
                        device=self.delta.device, dtype=self.delta.dtype)
        u[self.grasp_idx] = self.delta
        return u

    def to(self, device):
        return DLOAction(self.grasp_idx.to(device), self.delta.to(device))


def build_edges(num_nodes: int,
                contact_pairs: torch.Tensor | None = None,
                device="cpu"):
    """
    构造边索引与"是否接触边"标记。

    返回:
        edge_index [2, E]  ：每列是一条有向边 (src, dst)。结构边做成双向。
        is_contact [E]     ：该边是否为接触边（1）还是结构边（0）

    structural edges: (i, i+1) 双向，共 2*(N-1) 条
    contact edges:    传入的 contact_pairs [[i,j],...]，也做双向
    """
    src, dst, is_c = [], [], []
    for i in range(num_nodes - 1):
        src += [i, i + 1]
        dst += [i + 1, i]
        is_c += [0, 0]
    if contact_pairs is not None and len(contact_pairs) > 0:
        for i, j in contact_pairs.tolist():
            src += [int(i), int(j)]
            dst += [int(j), int(i)]
            is_c += [1, 1]
    edge_index = torch.tensor([src, dst], dtype=torch.long, device=device)
    is_contact = torch.tensor(is_c, dtype=torch.float32, device=device)
    return edge_index, is_contact


def compute_edge_features(pos: torch.Tensor,
                          edge_index: torch.Tensor,
                          is_contact: torch.Tensor) -> torch.Tensor:
    """
    根据当前节点位置，算每条边的几何特征。
    每步前向都重算（因为 pos 在 rollout 中变化）。

    edge_feat = [ rel_pos(3), dist(1), is_contact(1) ]  -> [E, EDGE_FEAT_DIM]
    """
    src, dst = edge_index[0], edge_index[1]
    rel = pos[dst] - pos[src]                    # [E, 3]
    dist = rel.norm(dim=-1, keepdim=True)        # [E, 1]
    return torch.cat([rel, dist, is_contact.unsqueeze(-1)], dim=-1)
