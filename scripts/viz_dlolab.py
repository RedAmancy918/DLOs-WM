"""
眼检 DLOLabProvider 产出的轨迹是否物理合理。
直接画 centerline 的 pos（WM 真正消费的几何），不走 genesis 渲染（服务器无显示）。
输出一张 PNG：行=侧视(XZ)/俯视(XY)，列=若干时刻，节点按 tension 上色，标题给 topo/contact。
    python scripts/viz_dlolab.py --num_nodes 40 --T 12 --out runs/viz.png
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dlo_wm.data.dlolab_provider import DLOLabProvider


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_nodes", type=int, default=40)
    ap.add_argument("--T", type=int, default=12)
    ap.add_argument("--max_disp", type=float, default=0.03)
    ap.add_argument("--motion", type=str, default="fold")
    ap.add_argument("--lift", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="runs/viz.png")
    args = ap.parse_args()

    provider = DLOLabProvider(num_nodes=args.num_nodes, max_disp=args.max_disp,
                              motion=args.motion, lift_height=args.lift, seed=args.seed)
    states, actions, cpairs = provider.sample_trajectory(T=args.T)

    # 选最多 6 个时刻
    T = len(states)
    cols = min(6, T)
    idxs = np.linspace(0, T - 1, cols).round().astype(int)

    # 统一坐标范围
    allp = np.stack([s.pos.numpy() for s in states])  # [T,N,3]
    xlim = (allp[..., 0].min(), allp[..., 0].max())
    ylim = (allp[..., 1].min(), allp[..., 1].max())
    zlim = (allp[..., 2].min(), allp[..., 2].max())
    tmax = max(1e-6, float(np.stack([s.tension.numpy() for s in states]).max()))

    fig, axes = plt.subplots(2, cols, figsize=(3.2 * cols, 6.4))
    for c, t in enumerate(idxs):
        s = states[t]
        p = s.pos.numpy()
        ten = s.tension.numpy()
        con = s.contact.numpy()
        npairs = len(cpairs[t]) if cpairs[t] is not None else 0
        topo = int(s.topology)

        for r, (a, b, an, bn, lim_a, lim_b) in enumerate([
            (0, 2, "x", "z", xlim, zlim),   # 侧视 XZ
            (0, 1, "x", "y", xlim, ylim),   # 俯视 XY
        ]):
            ax = axes[r, c]
            ax.plot(p[:, a], p[:, b], "-", color="0.6", lw=1, zorder=1)
            sc = ax.scatter(p[:, a], p[:, b], c=ten, cmap="viridis",
                            vmin=0, vmax=tmax, s=18, zorder=2)
            # 标出接触节点
            cmask = con > 0.5
            if cmask.any():
                ax.scatter(p[cmask, a], p[cmask, b], facecolors="none",
                           edgecolors="red", s=60, lw=1.2, zorder=3)
            ax.set_xlim(lim_a); ax.set_ylim(lim_b)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel(an); ax.set_ylabel(bn)
            if r == 0:
                ax.set_title(f"t={t}  topo={topo}\ncontact={int(cmask.sum())} pairs={npairs}",
                             fontsize=9)
    fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.6, label="tension")
    fig.suptitle(f"DLOLabProvider  N={args.num_nodes}  T={args.T}  "
                 f"(side XZ top / top XY bottom; red ring = contact node)", fontsize=11)
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"[saved] {args.out}")
    # 顺带打印逐帧统计
    for t, s in enumerate(states):
        print(f"  t={t:2d}  topo={int(s.topology)}  contact={int((s.contact>0.5).sum())}  "
              f"tension[max]={float(s.tension.max()):.1f}  "
              f"pos_z[min]={float(s.pos[:,2].min()):.3f}")


if __name__ == "__main__":
    main()
