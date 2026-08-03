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

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

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
# MaterialCondition.global_features() 的磁盘/checkpoint 契约。即使维度不变，
# 顺序或变换发生变化也必须触发显式版本迁移，不能静默复用旧权重。
MATERIAL_FEATURE_NAMES = (
    "log_K",
    "log_E",
    "log_G",
    "mean_radius",
    "linear_density",
    "mu_self_static",
    "mu_self_kinetic",
)
MATERIAL_FEATURE_DIM = len(MATERIAL_FEATURE_NAMES)


@dataclass
class DLOState:
    """单帧 DLO 真实动态状态。所有张量第一维是 N（节点数）。

    障碍物 SDF、法向和最近障碍物编号不属于状态；它们必须由当前位置与
    环境定义在每次前向（尤其是多步 rollout）中重新派生。
    """
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
    """单步动作：G 个抓手各自的抓取点与控制目标。

    ``grasp_idx`` 与 ``delta`` 保留原有位置和语义，所以旧代码中的
    ``DLOAction(grasp_idx, delta)`` 无需修改。其余字段用于保存更完整的
    抓手控制命令；缺省时不改变旧路径行为。
    """
    grasp_idx: torch.Tensor  # [G] long
    delta: torch.Tensor      # [G, 3]
    gripper_id: torch.Tensor | None = None      # [G] long，物理抓手编号
    target_pos: torch.Tensor | None = None      # [G, 3]，世界系目标位置
    target_vel: torch.Tensor | None = None      # [G, 3]，世界系目标速度
    grasp_active: torch.Tensor | None = None    # [G] bool，本步是否激活抓取
    duration: torch.Tensor | float | None = None  # scalar 或 [G]，控制持续时间

    def __post_init__(self):
        # 持续时间在配置中常写成 Python 标量，内部仍统一为张量便于迁移设备。
        if self.duration is not None and not isinstance(self.duration, torch.Tensor):
            self.duration = torch.as_tensor(
                self.duration, dtype=self.delta.dtype, device=self.delta.device)

    @property
    def num_grippers(self) -> int:
        return int(self.grasp_idx.numel())

    def validate(self, num_nodes: int | None = None) -> DLOAction:
        """检查动作形状，并在给出节点数时检查所有激活抓取索引。"""
        if self.grasp_idx.ndim != 1:
            raise ValueError("grasp_idx 必须是 [G]")
        if self.grasp_idx.dtype != torch.long:
            raise TypeError("grasp_idx 必须使用 torch.long")

        g = self.num_grippers
        if self.delta.shape != (g, ACTION_DIM):
            raise ValueError(f"delta 必须是 [{g}, {ACTION_DIM}]")
        if not torch.isfinite(self.delta).all():
            raise ValueError("delta 包含非有限值")

        if self.gripper_id is not None:
            if self.gripper_id.shape != (g,):
                raise ValueError(f"gripper_id 必须是 [{g}]")
            if self.gripper_id.dtype != torch.long:
                raise TypeError("gripper_id 必须使用 torch.long")
        for name, value in (("target_pos", self.target_pos),
                            ("target_vel", self.target_vel)):
            if value is not None:
                if value.shape != (g, POS_DIM):
                    raise ValueError(f"{name} 必须是 [{g}, {POS_DIM}]")
                if not torch.isfinite(value).all():
                    raise ValueError(f"{name} 包含非有限值")
        if self.grasp_active is not None and self.grasp_active.shape != (g,):
            raise ValueError(f"grasp_active 必须是 [{g}]")
        if self.duration is not None:
            if self.duration.ndim > 1 or self.duration.numel() not in (1, g):
                raise ValueError("duration 必须是标量或 [G]")
            if not torch.isfinite(self.duration).all() or (self.duration < 0).any():
                raise ValueError("duration 必须是有限的非负数")

        if num_nodes is not None:
            if num_nodes <= 0:
                raise ValueError("num_nodes 必须大于 0")
            active = self._active_mask(device=self.grasp_idx.device)
            invalid = active & ((self.grasp_idx < 0) | (self.grasp_idx >= num_nodes))
            if invalid.any():
                bad = self.grasp_idx[invalid].tolist()
                raise ValueError(f"激活抓取的节点索引越界: {bad}")
        return self

    def _active_mask(self, device: torch.device | str) -> torch.Tensor:
        """返回抓手激活掩码；旧动作缺省为全部激活。"""
        if self.grasp_active is None:
            return torch.ones(self.num_grippers, dtype=torch.bool, device=device)
        return self.grasp_active.to(device=device, dtype=torch.bool)

    def to_node_drive(self, num_nodes: int,
                      current_pos: torch.Tensor | None = None) -> torch.Tensor:
        """
        把抓手动作"散射"成每个节点的驱动信号 u [N, 3]。
        被抓节点拿到对应 drive，其余为 0。若同时给出 ``current_pos`` 和
        ``target_pos``，drive 在本次调用中按 ``target_pos-current_pos`` 重算，
        使 rollout 基于预测位置做闭环伺服；否则沿用旧 ``delta`` 语义。
        （更真实的做法可以按弧长距离做高斯衰减，这里先用硬赋值，
         留在 TODO 里方便你替换。）
        """
        self.validate(num_nodes=num_nodes)
        use_target = current_pos is not None and self.target_pos is not None
        if current_pos is not None:
            if current_pos.shape != (num_nodes, POS_DIM):
                raise ValueError(f"current_pos 必须是 [{num_nodes}, {POS_DIM}]")
            if not torch.isfinite(current_pos).all():
                raise ValueError("current_pos 包含非有限值")
        reference = current_pos if use_target else self.delta
        u = torch.zeros(num_nodes, ACTION_DIM,
                        device=reference.device, dtype=reference.dtype)
        idx = self.grasp_idx.to(reference.device)
        active = self._active_mask(device=reference.device)
        # 失活抓手常用 -1 作哨兵；必须先过滤，避免 PyTorch 把它解释为末节点。
        if active.any():
            if use_target:
                target = self.target_pos.to(
                    device=reference.device, dtype=reference.dtype)
                u[idx[active]] = target[active] - current_pos[idx[active]]
            else:
                delta = self.delta.to(
                    device=reference.device, dtype=reference.dtype)
                u[idx[active]] = delta[active]
        return u

    def to(self, device):
        return DLOAction(
            grasp_idx=self.grasp_idx.to(device),
            delta=self.delta.to(device),
            gripper_id=_optional_to(self.gripper_id, device),
            target_pos=_optional_to(self.target_pos, device),
            target_vel=_optional_to(self.target_vel, device),
            grasp_active=_optional_to(self.grasp_active, device),
            duration=_optional_to(self.duration, device),
        )


@dataclass
class MaterialCondition:
    """一条 DLO 在一段 episode 内保持不变的物理条件。

    DLO-Lab 的质量和半径定义在顶点上，因此分别保存为 ``[N]``；只有
    静止长度定义在相邻顶点构成的段上，形状为 ``[N-1]``。
    """
    rest_length: torch.Tensor    # [N-1]
    node_mass: torch.Tensor      # [N]
    node_radius: torch.Tensor    # [N]
    K: torch.Tensor              # scalar，拉伸刚度
    E: torch.Tensor              # scalar，弯曲刚度
    G: torch.Tensor              # scalar，扭转刚度
    mu_self_static: torch.Tensor   # scalar，自接触静摩擦系数
    mu_self_kinetic: torch.Tensor  # scalar，自接触动摩擦系数

    def __post_init__(self):
        # 允许调用方传入 Python 标量，但对象内部统一保存标量张量。
        reference = self.rest_length
        if not isinstance(reference, torch.Tensor):
            raise TypeError("rest_length 必须是 torch.Tensor")
        for name in ("K", "E", "G", "mu_self_static", "mu_self_kinetic"):
            value = getattr(self, name)
            if not isinstance(value, torch.Tensor):
                setattr(self, name, torch.as_tensor(
                    value, dtype=reference.dtype, device=reference.device))

    @property
    def num_nodes(self) -> int:
        return int(self.node_mass.shape[0])

    def validate(self, num_nodes: int | None = None) -> MaterialCondition:
        """验证物理量形状、有限性和基本取值范围。"""
        for name in ("rest_length", "node_mass", "node_radius"):
            value = getattr(self, name)
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} 必须是 torch.Tensor")
            if value.ndim != 1:
                raise ValueError(f"{name} 必须是一维张量")
            if not value.is_floating_point():
                raise TypeError(f"{name} 必须是浮点张量")
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} 包含非有限值")
            if (value <= 0).any():
                raise ValueError(f"{name} 必须严格大于 0")

        n = self.num_nodes
        if n < 2:
            raise ValueError("DLO 至少需要两个节点")
        if self.node_radius.shape != (n,):
            raise ValueError(f"node_radius 必须是 [{n}]")
        if self.rest_length.shape != (n - 1,):
            raise ValueError(f"rest_length 必须是 [{n - 1}]")
        if num_nodes is not None and n != num_nodes:
            raise ValueError(f"材料节点数 {n} 与 episode 节点数 {num_nodes} 不一致")

        tensor_device = self.rest_length.device
        for name in ("node_mass", "node_radius", "K", "E", "G",
                     "mu_self_static", "mu_self_kinetic"):
            value = getattr(self, name)
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} 必须是 torch.Tensor")
            if value.device != tensor_device:
                raise ValueError("MaterialCondition 的所有张量必须位于同一设备")

        for name in ("K", "E", "G", "mu_self_static", "mu_self_kinetic"):
            value = getattr(self, name)
            if value.ndim != 0:
                raise ValueError(f"{name} 必须是标量张量")
            if not value.is_floating_point():
                raise TypeError(f"{name} 必须是浮点张量")
            if not torch.isfinite(value):
                raise ValueError(f"{name} 必须是有限值")
            if value < 0:
                raise ValueError(f"{name} 必须大于等于 0")
        for name in ("K", "E", "G"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} 必须严格大于 0，才能安全取对数")
        if self.mu_self_kinetic > self.mu_self_static:
            raise ValueError("动摩擦系数不能大于静摩擦系数")
        return self

    def linear_density(self) -> torch.Tensor:
        """返回整条 DLO 的线密度 ``总节点质量 / 总静止长度``。"""
        self.validate()
        return self.node_mass.sum() / self.rest_length.sum()

    def global_features(self) -> torch.Tensor:
        """返回 7 维图级特征。

        顺序固定为 ``log(K), log(E), log(G), mean(radius),
        linear_density, mu_self_static, mu_self_kinetic``。前三项取对数以压缩
        刚度参数的数量级，其余归一化由训练集统计量负责。
        """
        self.validate()
        dtype = self.rest_length.dtype
        device = self.rest_length.device
        scalars = [self.K.log(), self.E.log(), self.G.log(),
                   self.node_radius.mean(), self.linear_density(),
                   self.mu_self_static, self.mu_self_kinetic]
        return torch.stack([x.to(device=device, dtype=dtype) for x in scalars])

    def to(self, device) -> MaterialCondition:
        return MaterialCondition(
            rest_length=self.rest_length.to(device),
            node_mass=self.node_mass.to(device),
            node_radius=self.node_radius.to(device),
            K=self.K.to(device),
            E=self.E.to(device),
            G=self.G.to(device),
            mu_self_static=self.mu_self_static.to(device),
            mu_self_kinetic=self.mu_self_kinetic.to(device),
        )


@dataclass
class DLOEpisode:
    """版本化数据集中的完整轨迹单元。

    ``states`` 只保存真实动态状态。SDF、障碍物法向和最近障碍物编号等
    几何上下文应由图构建器按当前 ``pos`` 动态计算，不能缓存进这里。
    """
    material: MaterialCondition
    states: list[DLOState]
    actions: list[DLOAction]
    contact_pairs: list[torch.Tensor]
    macro_dt: float
    task: str = ""
    seed: int = 0
    id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def episode_id(self) -> str:
        """提供语义更明确的只读别名。"""
        return self.id

    @property
    def horizon(self) -> int:
        return len(self.actions)

    @property
    def num_nodes(self) -> int:
        if not self.states:
            raise ValueError("episode 不包含状态")
        return self.states[0].num_nodes

    def validate(self) -> DLOEpisode:
        """验证 T+1/T 时序关系和 episode 内各张量的节点契约。"""
        if not self.states:
            raise ValueError("states 至少包含初始状态")
        if len(self.states) != len(self.actions) + 1:
            raise ValueError("states/actions 必须满足长度 T+1/T")
        if len(self.contact_pairs) != len(self.states):
            raise ValueError("contact_pairs 必须与 states 等长")
        if not isinstance(self.macro_dt, (int, float)) or self.macro_dt <= 0:
            raise ValueError("macro_dt 必须是正数")
        if not isinstance(self.task, str):
            raise TypeError("task 必须是字符串")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TypeError("seed 必须是整数")
        if not isinstance(self.id, str):
            raise TypeError("id 必须是字符串")
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata 必须是字典")

        n = self.num_nodes
        self.material.validate(num_nodes=n)
        for index, state in enumerate(self.states):
            _validate_state(state, num_nodes=n, name=f"states[{index}]")
        for index, action in enumerate(self.actions):
            if not isinstance(action, DLOAction):
                raise TypeError(f"actions[{index}] 必须是 DLOAction")
            action.validate(num_nodes=n)
        for index, pairs in enumerate(self.contact_pairs):
            if not isinstance(pairs, torch.Tensor):
                raise TypeError(f"contact_pairs[{index}] 必须是 torch.Tensor")
            if pairs.ndim != 2 or pairs.shape[1] != 2:
                raise ValueError(f"contact_pairs[{index}] 必须是 [K, 2]")
            if pairs.dtype != torch.long:
                raise TypeError(f"contact_pairs[{index}] 必须使用 torch.long")
            if pairs.numel() > 0 and ((pairs < 0).any() or (pairs >= n).any()):
                raise ValueError(f"contact_pairs[{index}] 包含越界节点")
        return self

    def to(self, device) -> DLOEpisode:
        """把 episode 中的全部张量移动到指定设备。"""
        return DLOEpisode(
            material=self.material.to(device),
            states=[state.to(device) for state in self.states],
            actions=[action.to(device) for action in self.actions],
            contact_pairs=[pairs.to(device) for pairs in self.contact_pairs],
            macro_dt=self.macro_dt,
            task=self.task,
            seed=self.seed,
            id=self.id,
            metadata=self.metadata.copy(),
        )


def _optional_to(value: torch.Tensor | None, device):
    """移动可选张量，避免各数据类重复条件分支。"""
    return None if value is None else value.to(device)


def _validate_state(state: DLOState, num_nodes: int, name: str):
    """验证旧 DLOState 的既有张量约定，不改变其构造接口。"""
    if not isinstance(state, DLOState):
        raise TypeError(f"{name} 必须是 DLOState")
    expected = {
        "pos": (num_nodes, POS_DIM),
        "vel": (num_nodes, VEL_DIM),
        "tension": (num_nodes,),
        "contact": (num_nodes,),
    }
    for field_name, shape in expected.items():
        value = getattr(state, field_name)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name}.{field_name} 必须是 torch.Tensor")
        if value.shape != shape:
            raise ValueError(f"{name}.{field_name} 必须是 {list(shape)}")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name}.{field_name} 包含非有限值")
    if not isinstance(state.topology, torch.Tensor) or state.topology.ndim != 0:
        raise ValueError(f"{name}.topology 必须是标量张量")


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


def node_contact_from_edges(
    num_nodes: int,
    edge_index: torch.Tensor,
    is_contact: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """由当前动态接触边派生 per-node self-contact mask。

    该量和 edge features 一样依赖当前 ``pos`` 的构图结果，rollout 中不能把
    上一步 contact head 的预测当作下一步几何事实。
    """
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index 必须是 [2, E]")
    if is_contact.shape != (edge_index.shape[1],):
        raise ValueError("is_contact 必须是 [E]")
    if edge_index.device != is_contact.device:
        raise ValueError("edge_index 与 is_contact 必须位于同一设备")
    result = torch.zeros(
        num_nodes, dtype=dtype, device=edge_index.device
    )
    selected = edge_index[:, is_contact > 0.5]
    if selected.numel() > 0:
        nodes = selected.reshape(-1).unique()
        if (nodes < 0).any() or (nodes >= num_nodes).any():
            raise ValueError("接触边包含越界节点")
        result[nodes] = 1.0
    return result


def infer_self_contact_pairs(
    pos: torch.Tensor,
    node_radius: torch.Tensor,
    rest_length: torch.Tensor,
    *,
    contact_margin_scale: float = 0.5,
    distance_threshold: float | None = None,
) -> torch.Tensor:
    """统一的数据标注/rollout 自接触几何判据，返回无向 pairs ``[K,2]``。"""
    if pos.ndim != 2 or pos.shape[1] != POS_DIM:
        raise ValueError("pos 必须是 [N, 3]")
    n = pos.shape[0]
    if node_radius.shape != (n,):
        raise ValueError(f"node_radius 必须是 [{n}]")
    if rest_length.shape != (n - 1,):
        raise ValueError(f"rest_length 必须是 [{n - 1}]")
    if contact_margin_scale < 0:
        raise ValueError("contact_margin_scale 必须大于等于 0")
    radius = node_radius.to(device=pos.device, dtype=pos.dtype)
    rest = rest_length.to(device=pos.device, dtype=pos.dtype)
    if distance_threshold is None:
        radius_sum = radius[:, None] + radius[None, :]
        threshold = radius_sum * (1.0 + contact_margin_scale)
    else:
        if distance_threshold <= 0:
            raise ValueError("distance_threshold 必须大于 0")
        threshold = torch.full(
            (n, n), distance_threshold,
            device=pos.device,
            dtype=pos.dtype,
        )
    distance = torch.cdist(pos, pos)
    idx = torch.arange(n, device=pos.device)
    band_hops = max(
        1,
        int(torch.ceil(threshold.max() / rest.min()).item()),
    )
    band = (idx[:, None] - idx[None, :]).abs() <= band_hops
    mask = (distance < threshold) & (~band)
    return torch.nonzero(torch.triu(mask), as_tuple=False)


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
