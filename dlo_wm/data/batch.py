"""
Disjoint 批图（GPU 吞吐的关键）。

单图逐样本前向在 A100 上利用率极低。标准做法是把 B 条绳的图
拼成一张"不相交大图"：节点堆叠、边索引按累计节点数偏移、
记录每个节点属于哪条绳（batch_idx），图级 pooling 用 segment 聚合。

这样一次前向处理整个 batch，tensor core 才吃得满。
"""

from __future__ import annotations
from dataclasses import dataclass
import torch

from .schema import (DLOState, DLOAction, build_edges,
                     compute_edge_features, ACTION_DIM)


@dataclass
class BatchedGraph:
    node_feat: torch.Tensor    # [sumN, NODE_FEAT_DIM]
    pos: torch.Tensor          # [sumN, 3]
    vel: torch.Tensor          # [sumN, 3]
    drive: torch.Tensor        # [sumN, 3]
    edge_index: torch.Tensor   # [2, sumE]
    is_contact: torch.Tensor   # [sumE]
    batch_idx: torch.Tensor    # [sumN] 每节点所属图 id (0..B-1)
    num_graphs: int
    ptr: torch.Tensor          # [B+1] 每图节点起始偏移（CSR 风格）
    # 目标（训练用，可空）
    tgt_pos: torch.Tensor | None = None
    tgt_tension: torch.Tensor | None = None
    tgt_contact: torch.Tensor | None = None
    tgt_topo: torch.Tensor | None = None      # [B]

    def to(self, device):
        for f in ("node_feat", "pos", "vel", "drive", "edge_index",
                  "is_contact", "batch_idx", "ptr", "tgt_pos",
                  "tgt_tension", "tgt_contact", "tgt_topo"):
            v = getattr(self, f)
            if torch.is_tensor(v):
                setattr(self, f, v.to(device, non_blocking=True))
        return self


def collate_transitions(samples, device="cpu"):
    """
    把若干 transition 样本（dict: state_t/action_t/cpairs_t/state_tp1）
    拼成一个 BatchedGraph。
    """
    node_feats, poss, vels, drives = [], [], [], []
    eis, iscs, bidx = [], [], []
    tgt_pos, tgt_ten, tgt_con, tgt_topo = [], [], [], []
    ptr = [0]
    offset = 0

    for gid, smp in enumerate(samples):
        s = smp["state_t"]
        a = smp["action_t"]
        s1 = smp["state_tp1"]
        n = s.num_nodes
        cp = smp.get("cpairs_t", None)
        cp = cp if (cp is not None and len(cp) > 0) else None

        ei, isc = build_edges(n, cp)
        node_feats.append(s.node_features())
        poss.append(s.pos); vels.append(s.vel)
        drives.append(a.to_node_drive(n))
        eis.append(ei + offset)          # 关键：边索引偏移
        iscs.append(isc)
        bidx.append(torch.full((n,), gid, dtype=torch.long))

        tgt_pos.append(s1.pos); tgt_ten.append(s1.tension)
        tgt_con.append(s1.contact); tgt_topo.append(s1.topology.view(1))

        offset += n
        ptr.append(offset)

    bg = BatchedGraph(
        node_feat=torch.cat(node_feats), pos=torch.cat(poss),
        vel=torch.cat(vels), drive=torch.cat(drives),
        edge_index=torch.cat(eis, dim=1), is_contact=torch.cat(iscs),
        batch_idx=torch.cat(bidx), num_graphs=len(samples),
        ptr=torch.tensor(ptr, dtype=torch.long),
        tgt_pos=torch.cat(tgt_pos), tgt_tension=torch.cat(tgt_ten),
        tgt_contact=torch.cat(tgt_con), tgt_topo=torch.cat(tgt_topo),
    )
    return bg.to(device)


def segment_mean(x, batch_idx, num_graphs):
    """图级 mean-pool：按 batch_idx 把节点 latent 聚合成 [B, D]。"""
    D = x.shape[-1]
    out = torch.zeros(num_graphs, D, device=x.device, dtype=x.dtype)
    out.index_add_(0, batch_idx, x)
    counts = torch.zeros(num_graphs, device=x.device, dtype=x.dtype)
    counts.index_add_(0, batch_idx, torch.ones_like(batch_idx, dtype=x.dtype))
    return out / counts.clamp(min=1).unsqueeze(-1)
