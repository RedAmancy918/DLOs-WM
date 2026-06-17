"""
数据接口层。

设计目标：把"数据从哪来"和"模型怎么训"彻底解耦。
你只需要实现 TrajectoryProvider 接口，把你自己的仿真数据
（DER / XPBD / Isaac / MuJoCo）整理成 schema.DLOState 序列，
训练代码就能直接消费，不用改一行。

本文件提供两个东西：
  1. TrajectoryProvider 抽象接口  —— 你接自己数据时实现它
  2. SyntheticRope 一个最小程序化生成器 —— 让整个工程"开箱即跑"、
     方便你在接真实仿真前先验证模型/训练/评估管线是否正确。

注意：SyntheticRope 不是真实物理！它只是一个自洽的玩具动力学
（弹簧 + 简化张力 + 阈值接触），用来产生形状正确、各 head label 齐全
的转移样本，好让你先把 pipeline 跑通、把维度对齐、把 bug 抓干净。
真上仿真时把它换掉即可。
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import torch

from .schema import DLOState, DLOAction, build_edges


class TrajectoryProvider(ABC):
    """
    你自己的仿真数据接入这里。实现 sample_trajectory 即可。

    约定：一条轨迹是 T+1 个 DLOState 和 T 个 DLOAction，
    满足 state[t] --action[t]--> state[t+1]。
    contact_pairs 每帧可能不同，所以随 state 一起按帧给出。
    """

    @abstractmethod
    def sample_trajectory(self):
        """
        返回:
            states:  list[DLOState]      长度 T+1
            actions: list[DLOAction]     长度 T
            contact_pairs: list[Tensor]  长度 T+1，每个 [K_t, 2]（可空）
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def num_nodes(self) -> int:
        ...


class SyntheticRope(TrajectoryProvider):
    """
    自洽玩具绳子，仅用于打通管线。物理是简化的、不可当真。

    动力学（每个内部子步）：
      - 结构弹簧：把相邻节点拉回到 rest_length
      - 张力 = 相邻段弹簧力的合力大小（一个粗略代理）
      - 接触：任意非相邻节点对距离 < contact_radius 记为接触，
              并施加一个软排斥
      - 动作：被抓节点按 delta 移动
      - topology：玩具版用"接触对数量是否超阈值"粗暴映射成类别，
              真实场景应换成 Gauss linking / crossing number
    """

    def __init__(self, num_nodes=24, rest_length=0.05,
                 contact_radius=0.04, k_struct=40.0, k_contact=60.0,
                 damping=0.85, substeps=4, dt=0.04, seed=0):
        self._n = num_nodes
        self.rest = rest_length
        self.cr = contact_radius
        self.ks = k_struct
        self.kc = k_contact
        self.damping = damping
        self.substeps = substeps
        self.dt = dt
        self.g = torch.Generator().manual_seed(seed)

    @property
    def num_nodes(self) -> int:
        return self._n

    # ---------- 内部物理 ----------
    def _init_pos(self):
        """初始化成一条带随机扰动的直线。"""
        x = torch.arange(self._n).float() * self.rest
        pos = torch.stack([x, torch.zeros_like(x), torch.zeros_like(x)], dim=-1)
        pos += 0.01 * torch.randn(self._n, 3, generator=self.g)
        return pos

    def _find_contacts(self, pos):
        """O(N^2) 找非相邻接触对（玩具规模够用）。返回 [K,2] 与 per-node 0/1。"""
        n = self._n
        diff = pos.unsqueeze(0) - pos.unsqueeze(1)        # [n,n,3]
        d = diff.norm(dim=-1)                              # [n,n]
        idx = torch.arange(n)
        # 屏蔽自身与相邻
        mask = (d < self.cr)
        mask[idx, idx] = False
        for k in (-1, 0, 1):
            ii = idx[:-abs(k) or None]
            # 简单屏蔽 |i-j|<=1
        band = (torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1)) <= 1)
        mask = mask & (~band)
        pairs = torch.nonzero(torch.triu(mask), as_tuple=False)  # [K,2]
        node_contact = torch.zeros(n)
        if len(pairs) > 0:
            node_contact[pairs[:, 0]] = 1.0
            node_contact[pairs[:, 1]] = 1.0
        return pairs, node_contact

    def _step_physics(self, pos, vel, drive):
        """半隐式积分若干子步，返回新 pos, vel, tension, contact_pairs, node_contact。"""
        n = self._n
        tension = torch.zeros(n)
        pairs, node_contact = self._find_contacts(pos)
        for _ in range(self.substeps):
            force = torch.zeros(n, 3)
            seg = pos[1:] - pos[:-1]
            seglen = seg.norm(dim=-1, keepdim=True) + 1e-8
            stretch = seglen - self.rest
            f_struct = self.ks * stretch * (seg / seglen)     # [n-1,3]
            force[:-1] += f_struct
            force[1:] -= f_struct
            # 张力代理：节点处两侧结构力大小之和
            tmag = torch.zeros(n)
            fs = (self.ks * stretch.squeeze(-1)).abs()
            tmag[:-1] += fs
            tmag[1:] += fs
            tension = tmag
            # 接触排斥
            if len(pairs) > 0:
                i, j = pairs[:, 0], pairs[:, 1]
                rij = pos[i] - pos[j]
                dij = rij.norm(dim=-1, keepdim=True) + 1e-8
                pen = (self.cr - dij).clamp(min=0.0)
                fc = self.kc * pen * (rij / dij)
                force.index_add_(0, i, fc)
                force.index_add_(0, j, -fc)
            # 驱动（抓手）当作外力脉冲
            force = force + drive / self.dt
            vel = (vel + self.dt * force) * self.damping
            pos = pos + self.dt * vel
        return pos, vel, tension, pairs, node_contact

    def _topology_label(self, pairs):
        """玩具拓扑：按接触对数量分桶。真实应替换为拓扑不变量。"""
        k = 0 if pairs is None else len(pairs)
        if k == 0:
            return 0          # unknot-ish
        elif k <= 2:
            return 1
        else:
            return 2

    def _make_state(self, pos, vel, tension, pairs, node_contact):
        topo = torch.tensor(self._topology_label(pairs), dtype=torch.long)
        return DLOState(pos=pos.clone(), vel=vel.clone(),
                        tension=tension.clone(), contact=node_contact.clone(),
                        topology=topo)

    # ---------- 对外接口 ----------
    def sample_trajectory(self, T=20):
        n = self._n
        pos = self._init_pos()
        vel = torch.zeros(n, 3)
        # 先稳定一帧拿到初始 contact/tension
        pos, vel, tension, pairs, node_contact = self._step_physics(
            pos, vel, torch.zeros(n, 3))

        states = [self._make_state(pos, vel, tension, pairs, node_contact)]
        contact_pairs = [pairs]
        actions = []

        for t in range(T):
            # 随机抓一个端点附近的节点，给一个随机小位移
            g = int(torch.randint(0, n, (1,), generator=self.g))
            delta = 0.03 * torch.randn(1, 3, generator=self.g)
            action = DLOAction(grasp_idx=torch.tensor([g]), delta=delta)
            drive = action.to_node_drive(n)
            pos, vel, tension, pairs, node_contact = self._step_physics(pos, vel, drive)
            actions.append(action)
            states.append(self._make_state(pos, vel, tension, pairs, node_contact))
            contact_pairs.append(pairs)

        return states, actions, contact_pairs


def make_transition_batch(provider: TrajectoryProvider, n_traj=8, T=20):
    """
    把若干条轨迹拍平成单步转移样本，供监督训练。
    返回 list of dict，每个是一个 (state_t, action_t, state_{t+1}) 样本。
    """
    samples = []
    for _ in range(n_traj):
        states, actions, cpairs = provider.sample_trajectory(T=T)
        for t in range(len(actions)):
            samples.append({
                "state_t": states[t],
                "action_t": actions[t],
                "cpairs_t": cpairs[t],
                "state_tp1": states[t + 1],
            })
    return samples
