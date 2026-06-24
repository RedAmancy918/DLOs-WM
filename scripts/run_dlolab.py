"""
用预生成的 DLO-Lab loop 数据训练第一个 GNN World Model。

先：python scripts/gen_dataset.py --out runs/dlolab_loop.pt
后：python scripts/run_dlolab.py --data runs/dlolab_loop.pt

不 import genesis（数据已离线生成），纯 CPU 训练 + rollout 评估，
写 runs/model_dlolab.pt 和 runs/report_dlolab.json。
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from configs.default import DEFAULT_CONFIG
from dlo_wm.data.cached_provider import CachedTrajectoryProvider
from dlo_wm.model.gnn import DLOWorldModel
from dlo_wm.train.trainer import train
from dlo_wm.eval.rollout import evaluate_rollout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="runs/dlolab_loop.pt")
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args()
    torch.manual_seed(0)

    provider = CachedTrajectoryProvider(args.data, seed=0)

    # 针对 DLO-Lab 数据的配置覆盖
    cfg = dict(DEFAULT_CONFIG)
    cfg.update({
        "num_nodes": provider.num_nodes,
        "contact_radius": 0.015,      # = 3*segment_radius，与 provider 自接触口径一致
        "traj_len": 18,               # 与生成时 T 一致
        "traj_per_epoch": len(provider._trajs),  # 每 epoch 过一遍数据量
        "dt": 0.12,                   # 真实 macro-step = steps_interval(120)*1e-3
        "tension_limit": 1500.0,      # DLO-Lab 张力尺度（~0..2500）下的 failure 阈值
        "epochs": args.epochs,
        # 各 head 损失量级差异大（tension ~1000、pos ~0.01），重设权重使其可比，
        # 避免 tension 主导梯度。
        "weights": {"pos": 5.0, "tension": 1e-3, "contact": 0.5, "topo": 0.3, "fail": 0.2},
    })
    print("config:", json.dumps({k: cfg[k] for k in
          ["num_nodes","contact_radius","traj_len","traj_per_epoch","dt","tension_limit","epochs"]}, ensure_ascii=False))

    model = DLOWorldModel(hidden=cfg["hidden"], n_message_passing=cfg["n_message_passing"],
                          n_topo_classes=cfg["n_topo_classes"], dt=cfg["dt"])
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M\n")

    print("=== 训练 ===")
    history = train(model, provider, cfg)

    print("\n=== Rollout 评估 ===")
    report = evaluate_rollout(model, provider, cfg, n_traj=10)
    print("pos_rmse 随步数:", [f"{v:.4f}" for v in report["pos_rmse_per_step"][:10]])
    print("topo_acc 随步数:", [f"{v:.2f}" for v in report["topo_acc_per_step"][:10]])
    print("contact_acc 随步数:", [f"{v:.2f}" for v in report["contact_acc_per_step"][:10]])
    print(f"failure_acc: {report['failure_acc']:.2f}")

    os.makedirs("runs", exist_ok=True)
    torch.save(model.state_dict(), "runs/model_dlolab.pt")
    with open("runs/report_dlolab.json", "w") as f:
        json.dump({"history": history, "report": report}, f, indent=2)
    print("\n已保存 runs/model_dlolab.pt 和 runs/report_dlolab.json")


if __name__ == "__main__":
    main()
