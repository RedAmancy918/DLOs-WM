"""
DLO World Model 性能评估（held-out 测试集 + 基线 + 修正指标）。

    # 先生成独立测试集（不同 seed，与训练集不重叠）
    python scripts/gen_dataset.py --seed 1000 --out runs/dlolab_eval.pt
    # 评估
    python scripts/eval_dlolab.py --model runs/model_dlolab.pt --data runs/dlolab_eval.pt

测什么：
  - 闭环 rollout 的 pos_rmse@k（绝对 + 相对绳长），对比 WM / persistence / const-vel 基线
  - contact 用 precision/recall/F1（稀疏，准确率会骗人）
  - topo 用整体 acc + 「有交叉帧」上的 recall（类别不均衡）
  - failure 用 acc + AUC（teacher-forced，跨所有 transition）
  - 关键指标给 mean ± std（跨 eval 轨迹）
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from configs.default import DEFAULT_CONFIG
from dlo_wm.data.schema import DLOState
from dlo_wm.data.cached_provider import CachedTrajectoryProvider
from dlo_wm.model.gnn import DLOWorldModel
from dlo_wm.train.trainer import edge_builder_from_contacts
from dlo_wm.train.losses import derive_failure_label


# ---------------- baselines（产出与 model.rollout 同构的 pred_traj） ----------------
def rollout_persistence(init, T):
    """什么都不动：s_{t+1}=s_t → pred[k]=init。"""
    return [init for _ in range(T + 1)]


def rollout_const_vel(init, T, dt):
    """匀速外推：pos_k = pos_0 + vel_0 * dt * k；其余量沿用 init。"""
    traj = [init]
    for k in range(1, T + 1):
        traj.append(DLOState(
            pos=init.pos + init.vel * (dt * k),
            vel=init.vel, tension=init.tension,
            contact=init.contact, topology=init.topology))
    return traj


# ---------------- 指标 ----------------
def auc(labels, scores):
    pos = [s for l, s in zip(labels, scores) if l > 0.5]
    neg = [s for l, s in zip(labels, scores) if l <= 0.5]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def rope_length(state):
    seg = (state.pos[1:] - state.pos[:-1]).norm(dim=-1)
    return float(seg.sum())


def eval_method(pred_fn, provider, H, n_traj, rng_reset=0):
    """对一种预测方法跑评估，返回 per-traj 的 pos_rmse[H] 与聚合的 contact/topo 计数。"""
    pos_rmse = []          # list of [H] tensors, 每条轨迹一份
    rel_rmse = []
    tp = fp = fn = 0
    topo_correct = topo_total = 0
    cross_hit = cross_total = 0

    provider._rng.seed(rng_reset)      # 固定取样顺序，三种方法看同一批轨迹
    for _ in range(n_traj):
        states, actions, _ = provider.sample_trajectory(T=H)
        init = states[0]
        pred = pred_fn(init, actions)
        L = rope_length(init)
        per = torch.zeros(H)
        for k in range(1, min(H, len(states) - 1) + 1):
            gt, pr = states[k], pred[k]
            per[k - 1] = torch.sqrt(((pr.pos - gt.pos) ** 2).mean())
            # contact P/R/F1（逐节点聚合）
            pc, gc = (pr.contact > 0.5), (gt.contact > 0.5)
            tp += int((pc & gc).sum());  fp += int((pc & ~gc).sum());  fn += int((~pc & gc).sum())
            # topo
            topo_correct += int(pr.topology == gt.topology); topo_total += 1
            if int(gt.topology) > 0:
                cross_total += 1
                cross_hit += int(int(pr.topology) > 0)
        pos_rmse.append(per)
        rel_rmse.append(per / max(L, 1e-6))

    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return {
        "pos_rmse": torch.stack(pos_rmse),      # [n_traj, H]
        "rel_rmse": torch.stack(rel_rmse),
        "contact_PRF": (prec, rec, f1),
        "topo_acc": topo_correct / max(topo_total, 1),
        "cross_recall": cross_hit / max(cross_total, 1),
        "cross_total": cross_total,
    }


def fmt_ms(x):  # mean±std over trajs, 给 1/中/末三个 horizon
    m, s = x.mean(0), x.std(0)
    H = x.shape[1]
    ks = [0, H // 2, H - 1]
    return "  ".join(f"@{k+1}:{m[k]:.4f}±{s[k]:.4f}" for k in ks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/model_dlolab.pt")
    ap.add_argument("--data", default="runs/dlolab_eval.pt")
    ap.add_argument("--horizon", type=int, default=18)
    ap.add_argument("--n_traj", type=int, default=20)
    args = ap.parse_args()

    cfg = dict(DEFAULT_CONFIG)
    cfg.update({"contact_radius": 0.015, "dt": 0.12, "tension_limit": 1.5})
    provider = CachedTrajectoryProvider(args.data, seed=0)
    H = min(args.horizon, len(provider.sample_trajectory()[1]))  # 不超过数据长度

    model = DLOWorldModel(hidden=cfg["hidden"], n_message_passing=cfg["n_message_passing"],
                          n_topo_classes=cfg["n_topo_classes"], dt=cfg["dt"])
    model.load_state_dict(torch.load(args.model, weights_only=True))
    model.eval()
    builder = edge_builder_from_contacts(provider.num_nodes, cfg["contact_radius"])

    @torch.no_grad()
    def wm_fn(init, actions):
        return model.rollout(init, actions, builder)

    methods = {
        "WM":          wm_fn,
        "persistence": lambda init, a: rollout_persistence(init, H),
        "const-vel":   lambda init, a: rollout_const_vel(init, H, cfg["dt"]),
    }
    res = {name: eval_method(fn, provider, H, args.n_traj) for name, fn in methods.items()}

    print(f"\n=== Held-out 评估  (eval={args.data}, n_traj={args.n_traj}, horizon={H}) ===\n")
    print("[pos_rmse  mean±std]  (越低越好；WM 须明显打赢基线)")
    for n in methods: print(f"  {n:12s} {fmt_ms(res[n]['pos_rmse'])}")
    print("\n[pos_rmse / 绳长  相对误差]")
    for n in methods: print(f"  {n:12s} {fmt_ms(res[n]['rel_rmse'])}")
    print("\n[contact  precision / recall / F1]  (稀疏正类)")
    for n in methods:
        p, r, f = res[n]["contact_PRF"]; print(f"  {n:12s} P={p:.3f} R={r:.3f} F1={f:.3f}")
    print(f"\n[topology]  (有交叉帧共 {res['WM']['cross_total']} 个)")
    for n in methods:
        print(f"  {n:12s} acc={res[n]['topo_acc']:.3f}  有交叉帧recall={res[n]['cross_recall']:.3f}")

    # failure：teacher-forced，跨所有 transition，acc + AUC（仅 WM 有意义）
    labels, scores = [], []
    provider._rng.seed(0)
    with torch.no_grad():
        for _ in range(args.n_traj):
            states, actions, _ = provider.sample_trajectory(T=H)
            for k in range(len(actions)):
                ei, isc = builder(states[k].pos)
                drive = actions[k].to_node_drive(states[k].num_nodes)
                out = model(states[k], drive, ei, isc)
                labels.append(float(derive_failure_label(states[k + 1], cfg["tension_limit"], cfg["stuck_topo_classes"])))
                scores.append(float(out["fail_logit"]))
    acc = sum((s > 0) == (l > 0.5) for l, s in zip(labels, scores)) / max(len(labels), 1)
    pos_rate = sum(l > 0.5 for l in labels) / max(len(labels), 1)
    print(f"\n[failure]  acc={acc:.3f}  AUC={auc(labels, scores):.3f}  (正类占比={pos_rate:.2f}, n={len(labels)})")


if __name__ == "__main__":
    main()
