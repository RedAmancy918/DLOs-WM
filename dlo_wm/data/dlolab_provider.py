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

from .schema import (
    DLOAction,
    DLOEpisode,
    DLOState,
    MaterialCondition,
    infer_self_contact_pairs,
    node_contact_from_edges,
)
from .dataset import TrajectoryProvider
from .material_sampling import (
    MaterialParameters,
    MaterialRandomizationConfig,
    counterfactual_material_sweep,
    sample_material_parameters,
)


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
        mu_self_static: float = 0.3,
        mu_self_kinetic: float = 0.25,
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
        tension_scale: float = 1000.0,
        contact_mode: str = "self",
        n_topo_classes: int = 3,
        device: str = "cpu",
        seed: int = 0,
        material_randomization: MaterialRandomizationConfig | None = None,
    ):
        self._n = num_nodes
        self.interval = interval
        self.segment_radius = segment_radius
        self.segment_mass = segment_mass
        self.K = K                       # 拉伸刚度（>0 且 use_inextensible=False 才能伸长，张力才有信号）
        self.E = E                       # 弯曲刚度
        self.G = G                       # 扭转刚度
        self.mu_self_static = mu_self_static
        self.mu_self_kinetic = mu_self_kinetic
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
        self._contact_radius_override = contact_radius
        self.contact_radius = 3.0 * segment_radius if contact_radius is None else contact_radius
        # 张力代理刚度：默认用拉伸刚度 K，使原始量纲为力 ~ K*应变
        self.stretch_stiffness = K if stretch_stiffness is None else stretch_stiffness
        # 张力归一化尺度：把物理张力(~O(1000))除以它归一到 O(1)，使各 head 损失量级可比。
        # 物理张力 = 存储张力 * tension_scale（可逆，便于还原）。
        self.tension_scale = tension_scale
        # contact_mode: "self"=只算绳子自接触（与 contact_pairs 边一致，推荐）；
        #               "all"=自接触 ∪ 落地/外物碰撞（collided）
        self.contact_mode = contact_mode
        self.n_topo_classes = n_topo_classes
        self.device = device
        self.seed = seed
        self.g = torch.Generator(device="cpu").manual_seed(seed)
        self.material_randomization = material_randomization
        if (
            self.use_inextensible
            and material_randomization is not None
            and material_randomization.K_scale != (1.0, 1.0)
        ):
            raise ValueError(
                "use_inextensible=True 时 K 被约束屏蔽，不能随机化 K"
            )
        self._episode_index = 0

        self._scene = None
        self._rope = None
        self._rest_len = None  # [E] 初始段长，作 tension 的应变基准
        self._init_pos = None
        self._active_material: MaterialParameters | None = None

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
                static_friction=self.mu_self_static,
                kinetic_friction=self.mu_self_kinetic,
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
        # DLO-Lab 的 ``segment_mass`` 实际定义在 N 个顶点，而自然长度定义在 N-1 条边。
        # 因此线密度必须用总顶点质量 / 总自然长度，不能做逐元素相除。
        base_density = (
            self._n * self.segment_mass / float(self._rest_len.sum())
        )
        self._active_material = MaterialParameters(
            K=self.K,
            E=self.E,
            G=self.G,
            linear_density=base_density,
            radius=self.segment_radius,
            mu_self_static=self.mu_self_static,
            mu_self_kinetic=self.mu_self_kinetic,
        )
        # 让绳子先在重力下稳定几步再开始采样
        for _ in range(self.steps_interval):
            scene.step()

    def _reset(
        self,
        settle_steps: int | None = None,
        material: MaterialParameters | None = None,
    ) -> MaterialCondition | None:
        """完整复位 solver 隐状态，再设置本 episode 材料并选择是否沉降。"""
        # 只改 pos/vel 会遗留 theta/omega/twist 等 rod 隐状态，反事实组会因此
        # 不再真正共享初始条件。官方 DLO-Lab system-id 路径也是 reset 后再 setter。
        self._scene.reset()
        self._rope.set_position(self._init_pos)
        self._rope.set_velocity(torch.zeros(self._n, 3, device="cpu"))
        if self.anchor_ids:
            self._rope.set_fixed_states(fixed_ids=self.anchor_ids)
        condition = (
            None if material is None else self._apply_material(material)
        )
        if settle_steps is None:
            settle_steps = self.steps_interval
        if settle_steps < 0:
            raise ValueError("settle_steps 必须大于等于 0")
        for _ in range(settle_steps):
            self._scene.step()
        return condition

    def _base_material_parameters(self) -> MaterialParameters:
        """返回按 solver 自然长度换算后的基准材料参数。"""
        if self._rest_len is None:
            raise RuntimeError("scene 尚未 build，无法确定真实自然长度")
        return MaterialParameters(
            K=self.K,
            E=self.E,
            G=self.G,
            linear_density=(
                self._n * self.segment_mass / float(self._rest_len.sum())
            ),
            radius=self.segment_radius,
            mu_self_static=self.mu_self_static,
            mu_self_kinetic=self.mu_self_kinetic,
        )

    def _apply_material(self, material: MaterialParameters) -> MaterialCondition:
        """通过 DLO-Lab 的公开 setter 更新底层 solver，并返回实际 episode 条件。

        这些 setter 直接调用 solver kernel；因此只改变均匀材料参数时无需重建
        Genesis scene。若未来随机化节点数、自然形状或环境几何，则仍须重建。
        """
        if self._rope is None or self._rest_len is None:
            raise RuntimeError("必须先 build scene 再设置材料")
        scalar = lambda value: torch.tensor([value], dtype=torch.float32)
        total_mass = material.linear_density * float(self._rest_len.sum())
        node_mass = torch.full(
            (1, self._n), total_mass / self._n, dtype=torch.float32
        )
        node_radius = torch.full(
            (1, self._n), material.radius, dtype=torch.float32
        )
        mu_s = torch.full(
            (1, self._n), material.mu_self_static, dtype=torch.float32
        )
        mu_k = torch.full(
            (1, self._n), material.mu_self_kinetic, dtype=torch.float32
        )

        self._rope.set_stretching_stiffness(scalar(material.K))
        self._rope.set_bending_stiffness(scalar(material.E))
        self._rope.set_twisting_stiffness(scalar(material.G))
        self._rope.set_segment_mass(node_mass)
        self._rope.set_segment_radius(node_radius)
        self._rope.set_mu_s(mu_s)
        self._rope.set_mu_k(mu_k)

        self._active_material = material
        # 张力是 K*应变的派生监督，材料改变时必须同步使用当前 K。
        self.stretch_stiffness = material.K
        if self._contact_radius_override is None:
            self.contact_radius = 3.0 * material.radius

        device = self._rest_len.device
        dtype = self._rest_len.dtype
        condition = MaterialCondition(
            rest_length=self._rest_len.clone(),
            node_mass=node_mass[0].to(device=device, dtype=dtype),
            node_radius=node_radius[0].to(device=device, dtype=dtype),
            K=torch.tensor(material.K, device=device, dtype=dtype),
            E=torch.tensor(material.E, device=device, dtype=dtype),
            G=torch.tensor(material.G, device=device, dtype=dtype),
            mu_self_static=torch.tensor(
                material.mu_self_static, device=device, dtype=dtype
            ),
            mu_self_kinetic=torch.tensor(
                material.mu_self_kinetic, device=device, dtype=dtype
            ),
        )
        return condition.validate(num_nodes=self._n)

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

        tension = (self._edge_strain_to_node_tension(length) / self.tension_scale).float()
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
        if self._rest_len is None:
            raise RuntimeError("scene 尚未 build，缺少真实 rest_length")
        radius_value = (
            self._active_material.radius
            if self._active_material is not None
            else self.segment_radius
        )
        node_radius = torch.full(
            (self._n,), radius_value, device=pos.device, dtype=pos.dtype
        )
        pairs = infer_self_contact_pairs(
            pos,
            node_radius,
            self._rest_len,
            contact_margin_scale=0.5,
            distance_threshold=self._contact_radius_override,
        )
        # 复用图侧同一派生逻辑；先构造仅含 contact pairs 的双向索引。
        if len(pairs) > 0:
            contact_edge_index = torch.cat(
                [pairs.t(), pairs[:, [1, 0]].t()], dim=1
            )
            contact_flags = torch.ones(
                contact_edge_index.shape[1], device=pos.device
            )
        else:
            contact_edge_index = torch.empty(
                (2, 0), dtype=torch.long, device=pos.device
            )
            contact_flags = torch.empty(0, device=pos.device)
        node_contact = node_contact_from_edges(
            self._n,
            contact_edge_index,
            contact_flags,
            dtype=pos.dtype,
        )
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
        action.validate(num_nodes=self._n)
        active = action._active_mask(action.grasp_idx.device)
        v_cmd = action.delta[active] / (self.steps_interval * dt)  # [G_active,3]
        idx = action.grasp_idx[active]
        for _ in range(self.steps_interval):
            vel = self._rope.get_state().vel[0].detach().clone()  # [N,3] 当前真实速度
            j = idx.to(vel.device)
            if len(j) > 0:
                vel[j] = v_cmd.to(vel.dtype).to(vel.device)
            self._rope.set_velocity(vel)
            self._scene.step()

    def _make_action(
        self,
        grasp_idx: torch.Tensor,
        delta: torch.Tensor,
        current_pos: torch.Tensor,
        target_pos: torch.Tensor | None = None,
    ) -> DLOAction:
        """创建完整控制记录，同时保持旧 ``delta`` 驱动接口可用。"""
        grasp_idx = grasp_idx.to(device=self.device, dtype=torch.long)
        delta = delta.to(device=self.device, dtype=torch.float32)
        if target_pos is None:
            target_pos = current_pos[grasp_idx] + delta
        macro_dt = self.steps_interval * 1e-3
        return DLOAction(
            grasp_idx=grasp_idx,
            delta=delta,
            gripper_id=torch.arange(
                len(grasp_idx), dtype=torch.long, device=self.device
            ),
            target_pos=target_pos.to(self.device).float(),
            target_vel=delta / macro_dt,
            grasp_active=torch.ones(
                len(grasp_idx), dtype=torch.bool, device=self.device
            ),
            duration=torch.tensor(
                macro_dt, dtype=torch.float32, device=self.device
            ),
        ).validate(num_nodes=self._n)

    def _sample_action(
        self,
        current_pos: torch.Tensor,
        generator: torch.Generator,
    ) -> DLOAction:
        # 在非锚定顶点里随机选 G 个抓手，各给一个随机小位移
        candidates = [i for i in range(self._n) if i not in self.anchor_ids]
        # genesis 把 torch 默认设备设成 cuda；我们的 generator 在 cpu，故显式 device="cpu"
        perm = torch.randperm(
            len(candidates), generator=generator, device="cpu"
        )[: self.n_grasp]
        grasp = torch.tensor([candidates[i] for i in perm.tolist()],
                             dtype=torch.long, device=self.device)
        delta = (
            self.max_disp
            * torch.randn(
                self.n_grasp, 3, generator=generator, device="cpu"
            )
        ).to(self.device).float()
        return self._make_action(grasp, delta, current_pos)

    def _plan_motion(
        self,
        reference_pos: torch.Tensor,
        T: int,
        generator: torch.Generator,
    ):
        """规划自由端的折叠/缠绕路点，制造自接触与拓扑变化。
        返回 (grasp_idx:int, waypoints:[T,3] cpu)。路点以未受力的 canonical
        位形定义，从而不同材料的 paired episode 接收到相同绝对控制目标。
        """
        pos = reference_pos.float().cpu()        # [N,3] cpu
        gidx = self._n - 1
        start = pos[gidx].clone()
        anchor_idx = self.anchor_ids[0] if self.anchor_ids else 0
        anchor = pos[anchor_idx].clone()
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
            rand = lambda a, b: a + (b - a) * float(
                torch.rand(1, generator=generator, device="cpu")
            )
            R = rand(0.25, 0.45) * body[:2].norm().clamp(min=1e-3)
            sign = (
                1.0
                if torch.rand(1, generator=generator, device="cpu") > 0.5
                else -1.0
            )
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
    def _resolve_material(
        self,
        material: MaterialParameters | None,
        generator: torch.Generator,
    ) -> MaterialParameters:
        if material is not None:
            return material
        base = self._base_material_parameters()
        if self.material_randomization is None:
            return base
        if (
            self.use_inextensible
            and self.material_randomization.K_scale != (1.0, 1.0)
        ):
            raise ValueError(
                "use_inextensible=True 时 K 被约束屏蔽，不能随机化 K"
            )
        return sample_material_parameters(
            base, self.material_randomization, generator
        )

    def sample_episode(
        self,
        T: int = 20,
        *,
        material: MaterialParameters | None = None,
        episode_id: str | None = None,
        seed: int | None = None,
        action_seed: int | None = None,
        settle_steps: int | None = None,
        metadata: dict | None = None,
    ) -> DLOEpisode:
        """生成一个材料条件化 episode。

        ``seed`` 控制材料采样，``action_seed`` 独立控制动作/路点。反事实组给
        多个 episode 传相同 ``action_seed`` 和 ``settle_steps=0``，即可保证
        初始位形与绝对抓手目标一致，只改变材料条件。
        """
        if T <= 0:
            raise ValueError("T 必须大于 0")
        if self._scene is None:
            self._build_scene()

        episode_seed = (
            self.seed + self._episode_index if seed is None else seed
        )
        control_seed = (
            episode_seed + 1_000_003 if action_seed is None else action_seed
        )
        material_generator = torch.Generator(device="cpu").manual_seed(
            episode_seed
        )
        action_generator = torch.Generator(device="cpu").manual_seed(
            control_seed
        )

        params = self._resolve_material(material, material_generator)
        condition = self._reset(
            settle_steps=settle_steps, material=params
        )
        assert condition is not None

        state0, pairs0 = self._read_state()
        states = [state0]
        contact_pairs = [pairs0]
        actions = []

        plan = (
            None
            if self.motion == "random"
            else self._plan_motion(self._init_pos, T, action_generator)
        )

        for t in range(T):
            if plan is None:
                action = self._sample_action(
                    states[-1].pos, action_generator
                )
            else:
                gidx, waypoints = plan
                cur = states[-1].pos[gidx]
                target = waypoints[t].to(self.device)
                delta = (target - cur).unsqueeze(0)
                action = self._make_action(
                    torch.tensor([gidx], device=self.device),
                    delta,
                    states[-1].pos,
                    target_pos=target.unsqueeze(0),
                )
            self._apply_action(action)
            state, pairs = self._read_state()
            actions.append(action)
            states.append(state)
            contact_pairs.append(pairs)

        if episode_id is None:
            episode_id = f"{self.motion}-{episode_seed:08d}"
        episode_metadata = {
            "provider": "DLOLabProvider",
            "control_seed": control_seed,
            "settle_steps": (
                self.steps_interval if settle_steps is None else settle_steps
            ),
            "material_parameters": params.to_dict(),
            "use_inextensible": self.use_inextensible,
            "steps_interval": self.steps_interval,
            "tension_scale": self.tension_scale,
            "contact_mode": self.contact_mode,
            "contact_margin_scale": 0.5,
            "contact_distance_threshold": self._contact_radius_override,
        }
        if metadata:
            episode_metadata.update(metadata)
        episode = DLOEpisode(
            material=condition,
            states=states,
            actions=actions,
            contact_pairs=contact_pairs,
            macro_dt=self.steps_interval * 1e-3,
            task=self.motion,
            seed=episode_seed,
            id=episode_id,
            metadata=episode_metadata,
        ).validate()
        self._episode_index += 1
        return episode

    def sample_counterfactual_group(
        self,
        T: int,
        parameter: str,
        scales: list[float] | tuple[float, ...],
        *,
        seed: int,
        group_id: str | None = None,
        base_material: MaterialParameters | None = None,
    ) -> list[DLOEpisode]:
        """生成只改变一个材料量的 paired counterfactual episode 组。"""
        if self.use_inextensible and parameter == "K":
            raise ValueError(
                "use_inextensible=True 时 K 反事实没有可辨识物理效应"
            )
        if self._scene is None:
            self._build_scene()
        base = base_material or self._base_material_parameters()
        materials = counterfactual_material_sweep(
            base, parameter, scales
        )
        group_id = group_id or f"cf-{parameter}-{seed:08d}"
        control_seed = seed + 2_000_003
        episodes = []
        for index, (scale, varied) in enumerate(zip(scales, materials)):
            episodes.append(self.sample_episode(
                T=T,
                material=varied,
                episode_id=f"{group_id}-{index:02d}",
                seed=seed,
                action_seed=control_seed,
                # 不做材料相关沉降，保证组内 state_0 的完整动力学状态一致。
                settle_steps=0,
                metadata={
                    "counterfactual_group_id": group_id,
                    "counterfactual_parameter": parameter,
                    "counterfactual_scale": float(scale),
                },
            ))
        return episodes

    def sample_trajectory(self, T: int = 20):
        """旧三元组接口；内部仍走 v2 episode，保证两条路径物理一致。"""
        episode = self.sample_episode(T=T)
        return episode.states, episode.actions, episode.contact_pairs


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
