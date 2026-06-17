"""
多 head 损失。

关键设计点（对应讨论里强调的）：
  - failure label 不是凭空给的，而是从 ground-truth 的 tension/topology 派生：
        failure = (max tension 超过阈值) OR (topology 进入"卡死"类)
    这样 failure head 学的是其他物理量的函数，而非独立信号。
  - 各 head 损失加权求和，权重在 config 里调。
  - tension 用 relative/Huber 更稳（张力长尾）；contact 用 BCE；
    topology 用 CE；pos 用 MSE（也可换成对 acc 的监督）。
"""

import torch
import torch.nn.functional as F


def derive_failure_label(state_tp1, tension_limit, stuck_topo_classes):
    """
    从真实下一帧派生 failure 标签（0/1 float）。
    state_tp1: DLOState (ground truth)
    """
    over_tension = (state_tp1.tension.max() > tension_limit)
    stuck = torch.tensor(int(state_tp1.topology) in stuck_topo_classes)
    return (over_tension | stuck).float()


def world_model_loss(pred, target_state, weights, tension_limit, stuck_topo_classes):
    """
    pred: model.forward 的输出 dict
    target_state: DLOState (真实 t+1)
    weights: dict of head weights
    返回 (total_loss, logdict)
    """
    # 1) 几何：预测下一帧位置
    l_pos = F.mse_loss(pred["pos_next"], target_state.pos)
    # 2) 张力：Huber 抗长尾
    l_tension = F.smooth_l1_loss(pred["tension"], target_state.tension)
    # 3) 接触：BCE
    l_contact = F.binary_cross_entropy_with_logits(
        pred["contact_logit"], target_state.contact)
    # 4) 拓扑：CE（单图，扩成 batch 维）
    l_topo = F.cross_entropy(
        pred["topo_logits"].unsqueeze(0),
        target_state.topology.view(1))
    # 5) failure：label 由真实 tension/topo 派生
    fail_label = derive_failure_label(
        target_state, tension_limit, stuck_topo_classes)
    l_fail = F.binary_cross_entropy_with_logits(
        pred["fail_logit"].view(1), fail_label.view(1))

    total = (weights["pos"] * l_pos
             + weights["tension"] * l_tension
             + weights["contact"] * l_contact
             + weights["topo"] * l_topo
             + weights["fail"] * l_fail)

    log = {
        "loss": total.item(), "pos": l_pos.item(), "tension": l_tension.item(),
        "contact": l_contact.item(), "topo": l_topo.item(), "fail": l_fail.item(),
    }
    return total, log
