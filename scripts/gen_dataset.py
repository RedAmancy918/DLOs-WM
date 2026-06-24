"""
批量生成 DLO-Lab loop 轨迹并存盘，供 run_dlolab.py 训练。

    python scripts/gen_dataset.py --n_traj 40 --num_nodes 48 --T 18 \
        --steps_interval 120 --out runs/dlolab_loop.pt

一条轨迹 ~十几秒（GPU 仿真），40 条约 10~15 分钟。每条之间 provider 自动复位 + loop 随机化。
"""
import sys, os, argparse, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from dlo_wm.data.dlolab_provider import DLOLabProvider


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_traj", type=int, default=40)
    ap.add_argument("--num_nodes", type=int, default=48)
    ap.add_argument("--T", type=int, default=18)
    ap.add_argument("--steps_interval", type=int, default=120)
    ap.add_argument("--motion", type=str, default="loop")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="runs/dlolab_loop.pt")
    args = ap.parse_args()

    provider = DLOLabProvider(num_nodes=args.num_nodes, motion=args.motion,
                              steps_interval=args.steps_interval, seed=args.seed)

    trajs = []
    t0 = time.time()
    for i in range(args.n_traj):
        states, actions, cpairs = provider.sample_trajectory(T=args.T)
        # 统计该轨迹的 contact/topology 覆盖，确认有非平凡信号
        topo_set = sorted({int(s.topology) for s in states})
        contact_steps = sum(int((s.contact > 0.5).any()) for s in states)
        ten_max = max(float(s.tension.max()) for s in states)
        trajs.append((states, actions, cpairs))
        print(f"[{i+1}/{args.n_traj}] topo={topo_set} contact_steps={contact_steps}/{len(states)} "
              f"ten_max={ten_max:.0f}  ({time.time()-t0:.0f}s)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"trajs": trajs, "num_nodes": args.num_nodes}, args.out)
    # 汇总
    n_topo1 = sum(any(int(s.topology) > 0 for s in st) for st, _, _ in trajs)
    n_contact = sum(any((s.contact > 0.5).any() for s in st) for st, _, _ in trajs)
    print(f"\n[saved] {len(trajs)} trajs -> {args.out}")
    print(f"  有交叉(topo>0)的轨迹: {n_topo1}/{len(trajs)}")
    print(f"  有自接触的轨迹:      {n_contact}/{len(trajs)}")


if __name__ == "__main__":
    main()
