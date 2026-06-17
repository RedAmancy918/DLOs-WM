"""
Rollout 评估。

评估一个 world model 好不好，单步 loss 不够——真正重要的是
闭环多步预测的稳定性（误差累积有多快），以及各 head 在 rollout 下还准不准。

指标：
  - pos_rmse@k     ：第 k 步预测位置 vs 真实的 RMSE，看误差随 horizon 增长
  - tension_mae@k  ：张力场误差
  - contact_acc@k  ：接触分类准确率
  - topo_acc@k     ：拓扑分类准确率
  - fail_auc-ish   ：failure 预测与派生真值的一致率（简化为 acc）
"""

from __future__ import annotations
import torch

from ..data.schema import build_edges
from ..train.losses import derive_failure_label
from ..train.trainer import edge_builder_from_contacts


@torch.no_grad()
def evaluate_rollout(model, provider, cfg, n_traj=20, device="cpu"):
    model.to(device).eval()
    builder = edge_builder_from_contacts(provider.num_nodes, cfg["contact_radius"])
    T = cfg["traj_len"]

    pos_err = torch.zeros(T)
    ten_err = torch.zeros(T)
    con_acc = torch.zeros(T)
    topo_acc = torch.zeros(T)
    fail_correct = 0
    fail_total = 0
    counts = torch.zeros(T)

    for _ in range(n_traj):
        states, actions, _ = provider.sample_trajectory(T=T)
        init = states[0].to(device)
        pred_traj = model.rollout(init, [a.to(device) for a in actions], builder)

        for k in range(1, len(states)):
            gt = states[k].to(device)
            pr = pred_traj[k]
            pos_err[k - 1] += torch.sqrt(((pr.pos - gt.pos) ** 2).mean())
            ten_err[k - 1] += (pr.tension - gt.tension).abs().mean()
            con_acc[k - 1] += (pr.contact == gt.contact).float().mean()
            topo_acc[k - 1] += float(pr.topology == gt.topology)
            counts[k - 1] += 1

        # failure：拿 rollout 终点处模型 fail head vs 真实派生 label
        # （这里简单用真实终态派生真值，模型预测用最后一步前向的 fail_logit）
        edge_index, is_contact = build_edges(
            init.num_nodes,
            None, device=device)
        drive = actions[-1].to(device).to_node_drive(init.num_nodes)
        out = model(states[-2].to(device), drive, edge_index, is_contact)
        pred_fail = (out["fail_logit"] > 0).float()
        true_fail = derive_failure_label(
            states[-1].to(device), cfg["tension_limit"], cfg["stuck_topo_classes"])
        fail_correct += float(pred_fail == true_fail)
        fail_total += 1

    counts = counts.clamp(min=1)
    report = {
        "pos_rmse_per_step": (pos_err / counts).tolist(),
        "tension_mae_per_step": (ten_err / counts).tolist(),
        "contact_acc_per_step": (con_acc / counts).tolist(),
        "topo_acc_per_step": (topo_acc / counts).tolist(),
        "failure_acc": fail_correct / max(fail_total, 1),
    }
    return report
