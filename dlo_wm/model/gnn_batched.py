"""
批图版 GNN World Model（GPU 吞吐版）。

与 model/gnn.py 同构，但前向消费 BatchedGraph，一次处理整个 batch。
图级 head（topology / failure）用 segment_mean 按图聚合。
节点级 head（pos / tension / contact）天然就是 per-node，无需改。

复用 model/gnn.py 里的 InteractionLayer / mlp，避免重复。
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gnn import mlp, InteractionLayer
from ..data.schema import (NODE_FEAT_DIM, EDGE_FEAT_DIM, POS_DIM,
                           compute_edge_features)
from ..data.batch import segment_mean, BatchedGraph


class BatchedDLOWorldModel(nn.Module):
    def __init__(self, hidden=256, n_message_passing=10, n_topo_classes=3, dt=0.04):
        super().__init__()
        self.hidden = hidden
        self.dt = dt
        self.node_enc = mlp([NODE_FEAT_DIM, hidden, hidden])
        self.edge_enc = mlp([EDGE_FEAT_DIM, hidden, hidden])
        self.layers = nn.ModuleList(
            [InteractionLayer(hidden) for _ in range(n_message_passing)])
        self.acc_head     = mlp([hidden, hidden, POS_DIM])
        self.tension_head = mlp([hidden, hidden, 1])
        self.contact_head = mlp([hidden, hidden, 1])
        self.topo_head    = mlp([hidden, hidden, n_topo_classes])
        self.fail_head    = mlp([hidden, hidden, 1])

    def forward(self, bg: BatchedGraph):
        e_feat = compute_edge_features(bg.pos, bg.edge_index, bg.is_contact)
        h = self.node_enc(bg.node_feat)
        e = self.edge_enc(e_feat)
        for layer in self.layers:
            h, e = layer(h, e, bg.edge_index, bg.drive)

        acc = self.acc_head(h)
        tension = F.softplus(self.tension_head(h)).squeeze(-1)
        contact_logit = self.contact_head(h).squeeze(-1)

        g = segment_mean(h, bg.batch_idx, bg.num_graphs)   # [B, hidden]
        topo_logits = self.topo_head(g)                    # [B, C]
        fail_logit = self.fail_head(g).squeeze(-1)         # [B]

        vel_next = bg.vel + self.dt * acc
        pos_next = bg.pos + self.dt * vel_next
        return {
            "acc": acc, "pos_next": pos_next, "vel_next": vel_next,
            "tension": tension, "contact_logit": contact_logit,
            "topo_logits": topo_logits, "fail_logit": fail_logit,
        }


def batched_loss(pred, bg, weights, tension_limit, stuck_topo_classes):
    """批图版多 head 损失。failure label 仍由 tgt tension/topo 派生（逐图）。"""
    l_pos = F.mse_loss(pred["pos_next"], bg.tgt_pos)
    l_tension = F.smooth_l1_loss(pred["tension"], bg.tgt_tension)
    l_contact = F.binary_cross_entropy_with_logits(pred["contact_logit"], bg.tgt_contact)
    l_topo = F.cross_entropy(pred["topo_logits"], bg.tgt_topo)

    # 逐图派生 failure：每图的 max tension
    B = bg.num_graphs
    max_ten = torch.full((B,), float("-inf"), device=bg.pos.device)
    max_ten = max_ten.scatter_reduce(
        0, bg.batch_idx, bg.tgt_tension, reduce="amax", include_self=True)
    max_ten = torch.where(torch.isinf(max_ten),
                          torch.zeros_like(max_ten), max_ten)
    over = max_ten > tension_limit
    stuck = torch.zeros(B, dtype=torch.bool, device=bg.pos.device)
    for c in stuck_topo_classes:
        stuck |= (bg.tgt_topo == c)
    fail_label = (over | stuck).float()
    l_fail = F.binary_cross_entropy_with_logits(pred["fail_logit"], fail_label)

    total = (weights["pos"] * l_pos + weights["tension"] * l_tension
             + weights["contact"] * l_contact + weights["topo"] * l_topo
             + weights["fail"] * l_fail)
    log = {"loss": total.item(), "pos": l_pos.item(), "tension": l_tension.item(),
           "contact": l_contact.item(), "topo": l_topo.item(), "fail": l_fail.item()}
    return total, log
