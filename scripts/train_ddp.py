"""
多卡 DDP 训练（A100 x N）。

启动:
    torchrun --standalone --nproc_per_node=<卡数> scripts/train_ddp.py

要点:
  - DistributedDataParallel 包模型，每卡跑一份，梯度自动 all-reduce
  - AMP 用 bf16（A100 原生支持，数值比 fp16 稳，无需 GradScaler）
  - 每卡用不同随机种子生成合成轨迹，等效数据并行
  - 只在 rank 0 打印 / 存档
接真实数据时：把 per-rank 的 SyntheticRope 换成你的 provider +
真正的 DistributedSampler 切分数据集即可。
"""

import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from configs.default import DEFAULT_CONFIG
from configs.a100 import A100_CONFIG
from dlo_wm.data.dataset import SyntheticRope, make_transition_batch
from dlo_wm.data.batch import collate_transitions
from dlo_wm.model.gnn_batched import BatchedDLOWorldModel, batched_loss


def setup():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, local_rank, dist.get_world_size()


def main():
    cfg = {**DEFAULT_CONFIG, **A100_CONFIG}
    rank, local_rank, world = setup()
    is_main = (rank == 0)
    device = torch.device(f"cuda:{local_rank}")
    torch.manual_seed(1234 + rank)

    provider = SyntheticRope(num_nodes=cfg["num_nodes"],
                             contact_radius=cfg["contact_radius"],
                             seed=1234 + rank)
    model = BatchedDLOWorldModel(
        hidden=cfg["hidden"], n_message_passing=cfg["n_message_passing"],
        n_topo_classes=cfg["n_topo_classes"], dt=cfg["dt"]).to(device)
    model = DDP(model, device_ids=[local_rank])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])

    if is_main:
        n = sum(p.numel() for p in model.parameters())
        print(f"world_size={world}  params={n/1e6:.2f}M  "
              f"batch/卡={cfg['batch_size']}  hidden={cfg['hidden']}  "
              f"mp={cfg['n_message_passing']}  nodes={cfg['num_nodes']}")

    for epoch in range(cfg["epochs"]):
        model.train()
        # 造一批轨迹 -> 拍平成 transition -> 切成 batch
        samples = make_transition_batch(
            provider, n_traj=cfg["traj_per_epoch"], T=cfg["traj_len"])
        t0 = time.time()
        running = {}
        bs = cfg["batch_size"]
        for i in range(0, len(samples), bs):
            chunk = samples[i:i + bs]
            bg = collate_transitions(chunk, device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = model(bg)
                loss, log = batched_loss(
                    pred, bg, cfg["weights"],
                    cfg["tension_limit"], cfg["stuck_topo_classes"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            for k, v in log.items():
                running[k] = running.get(k, 0) + v
        if is_main:
            nb = max(1, (len(samples) + bs - 1) // bs)
            dt = time.time() - t0
            msg = " ".join(f"{k}={v/nb:.4f}" for k, v in running.items())
            print(f"[ep {epoch:03d}] {msg}  ({dt:.1f}s, {len(samples)/dt:.0f} smp/s)")

    if is_main:
        os.makedirs("runs", exist_ok=True)
        torch.save(model.module.state_dict(), "runs/model_ddp.pt")
        print("saved runs/model_ddp.pt")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
