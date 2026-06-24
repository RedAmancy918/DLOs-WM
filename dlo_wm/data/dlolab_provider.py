"""
DLO-Lab -> DLOs-WM 数据桥接。

把 DLO-Lab（基于 Genesis 的可微 rod 仿真）产生的真实物理轨迹，整理成
dlo_wm.data.schema 约定的 (states, actions, contact_pairs)，用来替换玩具
SyntheticRope，让世界模型学真实物理。

用法（环境装好 DLO-Lab 后）：
    from dlo_wm.data.dlolab_provider import DLOLabProvider
    provider = DLOLabProvider(num_nodes=64)
    states, actions, cpairs = provider.sample_trajectory(T=20)
    # 然后把 scripts/run.py 里的 SyntheticRope(...) 换成这个 provider 即可。

================================================================
schema 五个量怎么来（对照 DLO-Lab 的 RODEntityState）：
    pos[N,3]      <- state.pos[0]            直接
    vel[N,3]      <- state.vel[0]            直接
    contact[N]    <- state.collided[0] 或几何自接触     直接/推断
    tension[N]    <- 由 edge 应变 (length-L0)/L0 * k 推    近似（rod solver 未暴露张力）
    topology(int) <- centerline 平面投影的 crossing number  几何推断
contact_pairs[K,2] <- 非相邻顶点几何邻近对（自接触），与 SyntheticRope / WM rollout 重推口径一致
action(抓点+位移)  <- 我们主动施加：选 G 个抓取顶点，每个 macro-step 给一个位移 delta
================================================================

注意：
- tension / topology 是物理上"有依据但近似"的代理（rod solver 没有直接吐张力场，
  也没有现成拓扑不变量）。它们比 SyntheticRope 的玩具版强得多，但绝对数值需在机器上标定：
  tension 的尺度由 `stretch_stiffness` 决定；topology 的 crossing 数依赖投影平面。
- 抓取驱动用"给抓取顶点注入速度"实现，解耦机器人；记录的 action.delta 是 commanded 位移。
  若要更贴合 benchmark 的双臂语义，可改成用 Franka / kinematic 顶点驱动（见 README 的 envs）。
- 全部几何量在世界系、单位米；WM 内部会做归一化，所以 stretch_stiffness 默认 1.0 即可先跑通。
"""

from __future__ import annotations
import math
import torch

from .schema import DLOState, DLOAction
from .dataset import TrajectoryProvider


class DLOLabProvider(TrajectoryProvider):
    """
    从 DLO-Lab 仿真采一条 DLO 轨迹，整理成 WM schema。

    Parameters
    ----------
    num_nodes : int
        绳子 centerline 顶点数 N（= rod n_vertices）。
    interval : float
        相邻顶点间隔（米），即 rod 段静止长度的初值参考。
    segment_radius : float
        rod 半径（米）。
    E, G : float
        rod 的弯曲/扭转刚度参数（透传给 gs.materials.ROD.Base）。
    anchor_ids : list[int] | None
        固定（钉住）的顶点；默认钉一端 [0,1] 让绳子在重力下不整体飞走。
    n_grasp : int
        抓手数 G（双臂取 2）。
    steps_interval : int
        每个 macro-step 内部跑多少个 scene.step()（仿真子步）。
    max_disp : float
        每个 macro-step 抓取位移 delta 的尺度（米）。
    contact_radius : float
        判定自接触的距离阈值（米）。
    stretch_stiffness : float
        张力代理的刚度系数（把无量纲应变换算成张力尺度）。默认 1.0。
    n_topo_classes : int
        拓扑类别数；crossing 数超过则截断到最后一类。
    device : str
    seed : int
    """

    def __init__(
        self,
        num_nodes: int = 64,
        interval: float = 0.01,
        segment_radius: float = 0.005,
        segment_mass: float = 0.001,
        K: float = 5e4,
        E: float = 1e5,
        G: float = 1e4,
        use_inextensible: bool = False,
        anchor_ids: list[int] | None = None,
        n_grasp: int = 2,
        steps_interval: int = 200,
        max_disp: float = 0.02,
        motion: str = "loop",
        fold_back_frac: float = 0.2,
        lift_height: float = 0.011,
        table_z: float = 0.006,
        contact_radius: float | None = None,
        stretch_stiffness: float | None = None,
        contact_mode: str = "self",
        n_topo_classes: int = 3,
        device: str = "cpu",
        seed: int = 0,
    ):
        self._n = num_nodes
        self.interval = interval
        self.segment_radius = segment_radius
        self.segment_mass = segment_mass
        self.K = K                       # 拉伸刚度（>0 且 use_inextensible=False 才能伸长，张力才有信号）
        self.E = E                       # 弯曲刚度
        self.G = G                       # 扭转刚度
        self.use_inextensible = use_inextensible
        self.anchor_ids = [0, 1] if anchor_ids is None else list(anchor_ids)
        self.n_grasp = n_grasp
        self.steps_interval = steps_interval
        self.max_disp = max_disp
        # 运动模式：random=随机小抓取（采几何/张力，不折叠）；
        #          fold=抓自由端折回压到绳身（产生自接触+交叉）；
        #          loop=抓自由端绕圈成环（产生交叉）。
        self.motion = motion
        self.fold_back_frac = fold_back_frac   # 折回落点离锚定端的比例（越小折得越狠）
        self.lift_height = lift_height         # 折回时抬起高度（越过绳身）
        self.table_z = table_z                 # 桌面高度（落点 z）
        # 自接触判定距离：默认 3*segment_radius。两股叠放时中心距≈2*radius(直径)，
        # 取 3*radius 留余量，才检得到"压在另一股上"的接触。
        self.contact_radius = 3.0 * segment_radius if contact_radius is None else contact_radius
        # 张力代理刚度：默认用拉伸刚度 K，使量纲为力 ~ K*应变
        self.stretch_stiffness = K if stretch_stiffness is None else stretch_stiffness
        # contact_mode: "self"=只算绳子自接触（与 contact_pairs 边一致，推荐）；
        #               "all"=自接触 ∪ 落地/外物碰撞（collided）
        self.contact_mode = contact_mode
        self.n_topo_classes = n_topo_classes
        self.device = device
        self.g = torch.Generator().manual_seed(seed)

        self._scene = None
        self._rope = None
        self._rest_len = None  # [E] 初始段长，作 tension 的应变基准

    @property
    def num_nodes(self) -> int:
        return self._n

    # ------------------------------------------------------------------
    # 场景构建（延迟到首次使用，避免 import 期硬依赖 genesis）
    # ------------------------------------------------------------------
    def _build_scene(self):
        import genesis as gs

        if not gs._initialized:
            gs.init(seed=0, precision="64", logging_level="warning", backend=gs.gpu)

        scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=1e-3, substeps=5),
            rod_options=gs.options.RODOptions(damping=30.0, angular_damping=20.0),
            show_viewer=False,
        )
        # 地面，给自接触/落地一个参照
        scene.add_entity(
            material=gs.materials.Rigid(needs_coup=True, coup_friction=0.1),
            morph=gs.morphs.Plane(fixed=True),
        )
        rope = scene.add_entity(
            material=gs.materials.ROD.Base(
                segment_radius=self.segment_radius,
                segment_mass=self.segment_mass,
                K=self.K, E=self.E, G=self.G,
                use_inextensible=self.use_inextensible,
            ),
            morph=gs.morphs.ParameterizedRod(
                type="rod",
                n_vertices=self._n,
                interval=self.interval,
                axis="x",
                pos=(0.3, 0.0, 0.05),   # 低位起始：warmup 后铺在桌面上，便于平面折叠/缠绕
                euler=(0.0, 0.0, 0.0),
            ),
        )
        scene.build(n_envs=1)
        if self.anchor_ids:
            rope.set_fixed_states(fixed_ids=self.anchor_ids)

        self._scene = scene
        self._rope = rope

        # tension 基准 = 绳子「未受力」时的实际段长：在 build 之后、施加重力之前抓取。
        # 不能用 interval（ParameterizedRod 实际自然段长 ≠ interval），
        # 也不能用 warmup 之后的边长（已被重力/动力学拉伸）。
        st0 = rope.get_state()
        self._rest_len = st0.length[0].detach().to(self.device).as_subclass(torch.Tensor).float().clone()  # [E] f32
        # 初始（直线、未受力）位形，供批量生成时 _reset() 复位
        self._init_pos = st0.pos[0].detach().to(self.device).as_subclass(torch.Tensor).float().clone()  # [N,3]
        # 让绳子先在重力下稳定几步再开始采样
        for _ in range(self.steps_interval):
            scene.step()

    def _reset(self):
        """把绳子复位到初始直线 + 零速 + 重新铺到桌面，使每条轨迹相互独立。"""
        self._rope.set_position(self._init_pos)
        self._rope.set_velocity(torch.zeros(self._n, 3, device="cpu"))
        if self.anchor_ids:
            self._rope.set_fixed_states(fixed_ids=self.anchor_ids)
        for _ in range(self.steps_interval):
            self._scene.step()

    # ------------------------------------------------------------------
    # 状态读取 + 派生量
    # ------------------------------------------------------------------
    def _read_state(self) -> DLOState:
        st = self._rope.get_state()
        # genesis 把 state 包成自定义 Tensor 子类，这里剥成普通 torch.Tensor，
        # 避免污染下游 WM 代码（格式化 / 训练）。
        def plain(x):
            return x.detach().to(self.device).as_subclass(torch.Tensor)
        pos = plain(st.pos[0]).float()       # [N,3]
        vel = plain(st.vel[0]).float()       # [N,3]
        length = plain(st.length[0]).float()  # [E]
        collided = plain(st.collided[0]).bool()  # [N]

        tension = self._edge_strain_to_node_tension(length).float()
        pairs, geo_contact = self._self_contacts(pos)
        if self.contact_mode == "all":
            contact = (collided | geo_contact.bool()).float()
        else:  # "self"：只取绳子自接触，与 contact_pairs（接触边）口径一致
            contact = geo_contact.float()
        topo = torch.tensor(self._crossing_number(pos), dtype=torch.long, device=self.device)

        state = DLOState(pos=pos, vel=vel, tension=tension, contact=contact, topology=topo)
        return state, pairs

    def _edge_strain_to_node_tension(self, length: torch.Tensor) -> torch.Tensor:
        """edge 应变 -> 节点张力代理。tension 只取拉伸（压缩绳子会屈曲，不承力）。"""
        rest = self._rest_len if self._rest_len is not None else length
        strain = (length - rest) / (rest + 1e-8)          # [E]
        edge_t = (self.stretch_stiffness * strain).clamp(min=0.0)  # [E]
        # edge(i)=段(i,i+1) -> 节点：相邻两段的平均
        n = self._n
        node_t = torch.zeros(n, device=self.device)
        node_t[:-1] += edge_t
        node_t[1:] += edge_t
        cnt = torch.ones(n, device=self.device)
        cnt[1:-1] = 2.0
        return node_t / cnt

    def _self_contacts(self, pos: torch.Tensor):
        """非相邻顶点几何邻近对（自接触）。返回 pairs[K,2] 与 per-node 0/1。
        口径与 SyntheticRope / WM rollout 的 edge_builder_from_contacts 一致。"""
        n = self._n
        d = torch.cdist(pos, pos)                                  # [N,N]
        idx = torch.arange(n, device=self.device)
        # 按弧长排除近邻：沿绳间距 < contact_radius 的点对（|i-j|*interval 太近）天然靠近，
        # 不算自接触，否则会把同一段绳的邻点误判为接触。
        band_k = max(1, int(math.ceil(self.contact_radius / self.interval)))
        band = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs() <= band_k
        mask = (d < self.contact_radius) & (~band)
        pairs = torch.nonzero(torch.triu(mask), as_tuple=False)    # [K,2]
        node_contact = torch.zeros(n, device=self.device)
        if len(pairs) > 0:
            node_contact[pairs[:, 0]] = 1.0
            node_contact[pairs[:, 1]] = 1.0
        return pairs.to(self.device), node_contact

    def _crossing_number(self, pos: torch.Tensor) -> int:
        """centerline 投影到 xy 平面，数非相邻线段的 2D 交叉数，映射成拓扑类别。
        真实拓扑不变量（Gauss linking / 结不变量）可在此替换。"""
        p = pos[:, :2]
        n = self._n
        crossings = 0
        for i in range(n - 1):
            a1, a2 = p[i], p[i + 1]
            for j in range(i + 2, n - 1):
                if i == 0 and j == n - 2:
                    continue  # 首尾段相邻（闭环时）跳过
                b1, b2 = p[j], p[j + 1]
                if _seg_intersect(a1, a2, b1, b2):
                    crossings += 1
        return min(crossings, self.n_topo_classes - 1)

    # ------------------------------------------------------------------
    # 抓取驱动
    # ------------------------------------------------------------------
    def _apply_action(self, action: DLOAction):
        """伺服抓取顶点跟随位移：每个子步重设抓取端速度（_tgt['vel'] 是一次性的，
        必须逐步重设），其余顶点保持真实动力学。这样抓取端才会真正拽动绳子。"""
        dt = 1e-3
        v_cmd = action.delta / (self.steps_interval * dt)  # [G,3]
        idx = action.grasp_idx
        for _ in range(self.steps_interval):
            vel = self._rope.get_state().vel[0].detach().clone()  # [N,3] 当前真实速度
            j = idx.to(vel.device)
            vel[j] = v_cmd.to(vel.dtype).to(vel.device)
            self._rope.set_velocity(vel)
            self._scene.step()

    def _sample_action(self) -> DLOAction:
        # 在非锚定顶点里随机选 G 个抓手，各给一个随机小位移
        candidates = [i for i in range(self._n) if i not in self.anchor_ids]
        # genesis 把 torch 默认设备设成 cuda；我们的 generator 在 cpu，故显式 device="cpu"
        perm = torch.randperm(len(candidates), generator=self.g, device="cpu")[: self.n_grasp]
        grasp = torch.tensor([candidates[i] for i in perm.tolist()],
                             dtype=torch.long, device=self.device)
        delta = (self.max_disp * torch.randn(self.n_grasp, 3, generator=self.g, device="cpu")).to(self.device).float()
        return DLOAction(grasp_idx=grasp, delta=delta)

    def _plan_motion(self, state0, T: int):
        """规划自由端的折叠/缠绕路点，制造自接触与拓扑变化。
        返回 (grasp_idx:int, waypoints:[T,3] cpu)。抓自由端（末顶点 N-1，锚在头部）。"""
        pos = state0.pos.float().cpu()           # [N,3] cpu
        gidx = self._n - 1
        start = pos[gidx].clone()
        anchor = pos[self.anchor_ids[0]].clone()
        wps = torch.zeros(T, 3, device="cpu")

        if self.motion == "fold":
            # 自由端折回压到绳身（发卡对折）：xy 朝锚定端走到 fold_back_frac 处，途中抬起越过绳身再落下
            target = anchor + self.fold_back_frac * (start - anchor)
            target[2] = self.table_z
            for t in range(T):
                s = (t + 1) / T
                wp = start + s * (target - start)
                wp[2] = wp[2] + self.lift_height * math.sin(math.pi * s)
                wps[t] = wp

        elif self.motion == "loop":
            # 自由端在 xy 平面内绕一圈成环（产生交叉）。每条轨迹随机化半径/方向/抬起，增加多样性。
            body = start - anchor
            rand = lambda a, b: a + (b - a) * float(torch.rand(1, generator=self.g, device="cpu"))
            R = rand(0.25, 0.45) * body[:2].norm().clamp(min=1e-3)
            sign = 1.0 if torch.rand(1, generator=self.g, device="cpu") > 0.5 else -1.0
            lift = rand(0.009, 0.014)
            bx = body / (body.norm() + 1e-8)                 # 绳方向（近 x）
            perp = sign * torch.tensor([-bx[1], bx[0], 0.0], device="cpu")  # xy 平面内垂直方向
            center = start - R * bx
            for t in range(T):
                s = (t + 1) / T
                ang = 2 * math.pi * s
                wp = center + R * (math.cos(ang) * bx + math.sin(ang) * perp)
                wp[2] = self.table_z + lift * 0.5 * (1 - math.cos(2 * math.pi * s))
                wps[t] = wp
        else:
            raise ValueError(f"unknown motion: {self.motion}")

        return gidx, wps

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    def sample_trajectory(self, T: int = 20):
        if self._scene is None:
            self._build_scene()
        else:
            self._reset()   # 每条轨迹复位，保证相互独立

        state0, pairs0 = self._read_state()
        states = [state0]
        contact_pairs = [pairs0]
        actions = []

        plan = None if self.motion == "random" else self._plan_motion(state0, T)

        for t in range(T):
            if plan is None:
                action = self._sample_action()
            else:
                gidx, wps = plan
                cur = states[-1].pos[gidx]              # [3] cpu，当前抓取点
                delta = (wps[t] - cur).unsqueeze(0)     # [1,3] 伺服到路点
                action = DLOAction(
                    grasp_idx=torch.tensor([gidx], device=self.device),
                    delta=delta.to(self.device).float(),
                )
            self._apply_action(action)
            state, pairs = self._read_state()
            actions.append(action)
            states.append(state)
            contact_pairs.append(pairs)

        return states, actions, contact_pairs


def _seg_intersect(p1, p2, p3, p4) -> bool:
    """2D 线段相交判定（含端点退化的鲁棒处理）。"""
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0])
    d1 = ccw(p3, p4, p1)
    d2 = ccw(p3, p4, p2)
    d3 = ccw(p1, p2, p3)
    d4 = ccw(p1, p2, p4)
    if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)):
        return True
    return False
