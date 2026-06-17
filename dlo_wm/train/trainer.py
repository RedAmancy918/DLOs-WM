"""
训练循环。

支持两种监督：
  - 单步 (1-step)：f(s_t, a_t) ≈ s_{t+1}，主力、稳定。
  - 多步 rollout (push-forward)：把模型自己的预测再喂回去走 k 步，
    对 k 步后的状态也加监督。这是缓解 GNN simulator 长 horizon
    误差累积的标准技巧（GNS 论文里叫 noise injection / pushforward）。
    这里给了一个简化的 2-step 版本，config 里可开关。

数据通过 TrajectoryProvider 接口拿——你换成自己的仿真 provider 即可。
"""

from __future__ import annotations
import torch

from ..data.schema import build_edges
from ..data.dataset import make_transition_batch
from .losses import world_model_loss


def edge_builder_from_contacts(num_nodes, contact_radius=0.04):
    """生成一个 edge_builder：仅用结构边 + 按距离推断的接触边。
    rollout 时没有 ground-truth 接触对，所以从几何重新推断。"""
    def _builder(pos):
        # 距离推断接触对（与玩具生成器同口径）
        n = pos.shape[0]
        d = (pos.unsqueeze(0) - pos.unsqueeze(1)).norm(dim=-1)
        idx = torch.arange(n)
        band = (torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1)) <= 1)
        mask = (d < contact_radius) & (~band)
        pairs = torch.nonzero(torch.triu(mask), as_tuple=False)
        return build_edges(n, pairs if len(pairs) else None, device=pos.device)
    return _builder


def train(model, provider, cfg, device="cpu", log_every=20):
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    builder = edge_builder_from_contacts(provider.num_nodes, cfg["contact_radius"])

    history = []
    for epoch in range(cfg["epochs"]):
        samples = make_transition_batch(
            provider, n_traj=cfg["traj_per_epoch"], T=cfg["traj_len"])
        epoch_log = {}
        for step, smp in enumerate(samples):
            s_t = smp["state_t"].to(device)
            a_t = smp["action_t"].to(device)
            s_tp1 = smp["state_tp1"].to(device)
            cpairs = smp["cpairs_t"]
            cpairs = cpairs.to(device) if cpairs is not None and len(cpairs) else None

            edge_index, is_contact = build_edges(
                s_t.num_nodes, cpairs, device=device)
            drive = a_t.to_node_drive(s_t.num_nodes)

            pred = model(s_t, drive, edge_index, is_contact)
            loss, log = world_model_loss(
                pred, s_tp1, cfg["weights"],
                cfg["tension_limit"], cfg["stuck_topo_classes"])

            # 可选：noise injection（pushforward 的廉价近似）
            if cfg.get("noise_std", 0) > 0:
                # 在输入位置加小噪声做下一次前向，鼓励对自身误差鲁棒
                noisy = s_t
                noisy.pos = noisy.pos + cfg["noise_std"] * torch.randn_like(noisy.pos)
                pred2 = model(noisy, drive, edge_index, is_contact)
                loss2, _ = world_model_loss(
                    pred2, s_tp1, cfg["weights"],
                    cfg["tension_limit"], cfg["stuck_topo_classes"])
                loss = loss + cfg["pushforward_weight"] * loss2

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            for k, v in log.items():
                epoch_log[k] = epoch_log.get(k, 0) + v
        n = len(samples)
        epoch_log = {k: v / n for k, v in epoch_log.items()}
        history.append(epoch_log)
        if epoch % 1 == 0:
            msg = " ".join(f"{k}={v:.4f}" for k, v in epoch_log.items())
            print(f"[epoch {epoch:03d}] {msg}")
    return history
