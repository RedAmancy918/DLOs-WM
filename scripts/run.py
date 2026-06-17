"""
主入口：训练 GNN DLO World Model，然后做 rollout 评估。

用法:
    python scripts/run.py

接你自己的数据：把下面的 SyntheticRope 换成你实现的 TrajectoryProvider 子类即可，
其余代码不用动。
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from configs.default import DEFAULT_CONFIG
from dlo_wm.data.dataset import SyntheticRope
from dlo_wm.model.gnn import DLOWorldModel
from dlo_wm.train.trainer import train
from dlo_wm.eval.rollout import evaluate_rollout


def main():
    cfg = DEFAULT_CONFIG
    torch.manual_seed(0)

    # ---- 数据源 ----
    # >>> 把这一行换成你自己的 provider，例如 MyIsaacRope(...) <<<
    provider = SyntheticRope(num_nodes=cfg["num_nodes"],
                             contact_radius=cfg["contact_radius"])

    # ---- 模型 ----
    model = DLOWorldModel(hidden=cfg["hidden"],
                          n_message_passing=cfg["n_message_passing"],
                          n_topo_classes=cfg["n_topo_classes"],
                          dt=cfg["dt"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params/1e6:.2f}M\n")

    # ---- 训练 ----
    print("=== 训练 ===")
    history = train(model, provider, cfg)

    # ---- 评估 ----
    print("\n=== Rollout 评估 ===")
    report = evaluate_rollout(model, provider, cfg, n_traj=20)
    print("\npos_rmse 随步数:",
          [f"{v:.4f}" for v in report["pos_rmse_per_step"][:10]])
    print("topo_acc 随步数:",
          [f"{v:.2f}" for v in report["topo_acc_per_step"][:10]])
    print("contact_acc 随步数:",
          [f"{v:.2f}" for v in report["contact_acc_per_step"][:10]])
    print(f"failure_acc: {report['failure_acc']:.2f}")

    # 保存
    os.makedirs("runs", exist_ok=True)
    torch.save(model.state_dict(), "runs/model.pt")
    with open("runs/report.json", "w") as f:
        json.dump({"history": history, "report": report}, f, indent=2)
    print("\n已保存 runs/model.pt 和 runs/report.json")


if __name__ == "__main__":
    main()
