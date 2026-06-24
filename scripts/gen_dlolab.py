"""
DLOLabProvider 冒烟测试 / 数据生成入口。

环境装好 DLO-Lab + DLOs-WM 后，先跑这个确认数据通路对齐：
    # 需要能 import 到 genesis（DLO-Lab）和 dlo_wm（本仓库）
    python scripts/gen_dlolab.py --num_nodes 64 --T 20

它会用 DLO-Lab 采一条轨迹，打印各 head 的张量形状/数值范围，做维度对齐检查。
加 --save runs/dlolab_demo.pt 可把若干条轨迹拍平成 transition 样本存盘。
"""

import argparse
import torch

from dlo_wm.data.dlolab_provider import DLOLabProvider
from dlo_wm.data.dataset import make_transition_batch
from dlo_wm.data.schema import NODE_FEAT_DIM, build_edges, compute_edge_features


def check_trajectory(states, actions, cpairs):
    s0 = states[0]
    N = s0.num_nodes
    print(f"[traj] T={len(actions)}  N={N}  states={len(states)}  cpairs={len(cpairs)}")
    print(f"  pos      {tuple(s0.pos.shape)}  range [{s0.pos.min().item():.3f}, {s0.pos.max().item():.3f}]")
    print(f"  vel      {tuple(s0.vel.shape)}  range [{s0.vel.min().item():.3f}, {s0.vel.max().item():.3f}]")
    print(f"  tension  {tuple(s0.tension.shape)}  range [{s0.tension.min().item():.4f}, {s0.tension.max().item():.4f}]")
    print(f"  contact  {tuple(s0.contact.shape)}  sum={s0.contact.sum().item():.0f}")
    print(f"  topology classes over traj: {sorted({int(s.topology) for s in states})}")
    print(f"  node_features dim = {s0.node_features().shape[-1]} (expect {NODE_FEAT_DIM})")
    assert s0.node_features().shape[-1] == NODE_FEAT_DIM

    # 边构造 + 边特征维度对齐检查
    ei, isc = build_edges(N, cpairs[0])
    ef = compute_edge_features(s0.pos, ei, isc)
    print(f"  edges    {tuple(ei.shape)}  contact_edges={int(isc.sum())}  edge_feat {tuple(ef.shape)}")
    print("  [ok] 维度对齐通过")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_nodes", type=int, default=64)
    ap.add_argument("--T", type=int, default=20)
    ap.add_argument("--n_traj", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", type=str, default=None)
    args = ap.parse_args()

    provider = DLOLabProvider(num_nodes=args.num_nodes, seed=args.seed)

    states, actions, cpairs = provider.sample_trajectory(T=args.T)
    check_trajectory(states, actions, cpairs)

    if args.save:
        samples = make_transition_batch(provider, n_traj=args.n_traj, T=args.T)
        torch.save(samples, args.save)
        print(f"[save] {len(samples)} transitions -> {args.save}")


if __name__ == "__main__":
    main()
