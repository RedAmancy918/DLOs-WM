"""
A100 压测 / benchmark（单卡）。

目的：在一张 80GB A100 上，逐档加压，找到 batch_size / num_nodes /
模型规模 能吃到什么量级——GPU 利用率、显存占用、吞吐(samples/s)。
DDP 多卡的总吞吐近似 = 单卡吞吐 x 卡数（数据并行），所以先把单卡吃满最关键。

运行（在 A100 机器上）:
    python scripts/benchmark.py

它会跑几组配置，每组：预热 -> 计时若干 step -> 打印 显存峰值 + samples/s。
OOM 的配置会被捕获并标记，帮你定位上限。
"""

import os, sys, time, gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from configs.default import DEFAULT_CONFIG
from dlo_wm.data.dataset import SyntheticRope, make_transition_batch
from dlo_wm.data.batch import collate_transitions
from dlo_wm.model.gnn_batched import BatchedDLOWorldModel, batched_loss


def run_one(num_nodes, hidden, mp, batch_size, device, n_steps=10, amp=True):
    """跑一组配置，返回 (samples_per_s, peak_mem_GB) 或抛 OOM。"""
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(num_nodes=num_nodes)
    provider = SyntheticRope(num_nodes=num_nodes,
                             contact_radius=cfg["contact_radius"])
    model = BatchedDLOWorldModel(hidden=hidden, n_message_passing=mp,
                                 n_topo_classes=cfg["n_topo_classes"],
                                 dt=cfg["dt"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # 造够一个 batch 的样本
    samples = make_transition_batch(
        provider, n_traj=max(8, batch_size // 4 + 1), T=24)
    samples = samples[:batch_size]
    if len(samples) < batch_size:  # 补足
        samples = (samples * (batch_size // len(samples) + 1))[:batch_size]

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()

    def step():
        bg = collate_transitions(samples, device=device)
        ctx = torch.autocast("cuda", dtype=torch.bfloat16) if amp else _null()
        with ctx:
            pred = model(bg)
            loss, _ = batched_loss(pred, bg, cfg["weights"],
                                   cfg["tension_limit"], cfg["stuck_topo_classes"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    # 预热
    for _ in range(3):
        step()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_steps):
        step()
    torch.cuda.synchronize()
    dt = time.time() - t0

    peak = torch.cuda.max_memory_allocated(device) / 1e9
    sps = n_steps * batch_size / dt
    del model, opt
    gc.collect(); torch.cuda.empty_cache()
    return sps, peak


class _null:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def main():
    if not torch.cuda.is_available():
        print("没有 CUDA，本脚本需在 A100 机器上运行。")
        print("（沙盒无 GPU，这里只验证代码可导入、逻辑正确。）")
        return
    device = torch.device("cuda")
    name = torch.cuda.get_device_name(device)
    total = torch.cuda.get_device_properties(device).total_memory / 1e9
    print(f"GPU: {name}  显存: {total:.0f}GB\n")

    # 逐档加压：每行一组配置
    configs = [
        # (num_nodes, hidden, mp, batch_size)
        (64, 128,  6,  256),
        (64, 256, 10,  256),
        (64, 256, 10,  512),
        (64, 256, 10, 1024),
        (128, 256, 10, 512),
        (128, 384, 12, 512),
        (128, 512, 12, 1024),
    ]
    print(f"{'nodes':>6}{'hidden':>8}{'mp':>4}{'batch':>7}"
          f"{'samples/s':>12}{'peak_GB':>10}  status")
    print("-" * 60)
    for nn, hd, mp, bs in configs:
        try:
            sps, peak = run_one(nn, hd, mp, bs, device)
            status = "OK"
            print(f"{nn:>6}{hd:>8}{mp:>4}{bs:>7}{sps:>12.0f}{peak:>10.1f}  {status}")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                print(f"{nn:>6}{hd:>8}{mp:>4}{bs:>7}{'—':>12}{'—':>10}  OOM")
            else:
                raise
    print("\n提示：找到 OK 的最大档作为单卡设置，DDP 总吞吐≈单卡×卡数。")


if __name__ == "__main__":
    main()
