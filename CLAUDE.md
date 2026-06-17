# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A graph-neural-network **world model for deformable linear objects** (rope/cable/wire). It does not predict images — it predicts structured physical quantities (a "learned physics simulator") to forecast action consequences for long-horizon tasks (cable routing, insertion, knot tying/untying).

Pure hand-written message passing in PyTorch — **no `torch_geometric` or other GNN libraries**, only `torch>=2.0`.

## Commands

```bash
pip install -r requirements.txt          # only torch>=2.0

python scripts/run.py                    # CPU single-graph: train on SyntheticRope + rollout eval
                                         #   -> writes runs/model.pt and runs/report.json

python scripts/benchmark.py              # single-GPU sweep (A100); finds batch/node/model size limits
                                         #   no-op without CUDA (prints a note)

torchrun --standalone --nproc_per_node=<N> scripts/train_ddp.py   # multi-GPU DDP + bf16 AMP
                                         #   -> writes runs/model_ddp.pt
```

There is **no test suite, linter, or CI**. "Verification" in the sandbox means the code imports and runs on CPU; the GPU paths are validated for correctness/dimensions on CPU and meant to run on real A100s.

## Architecture

Encode–process–decode in the Graph Network Simulator (GNS / DPI-Net) lineage:

- **Graph**: nodes = discretized centerline points; **structural edges** `(i, i+1)` (always present, bidirectional, elastic segments); **contact edges** `(i, j)` (dynamic self-contact, inferred by geometric proximity).
- **Processor**: M rounds of message passing (`InteractionLayer`): edge update → `index_add_` aggregation to dst nodes → node update, both with residual connections. The action (dual-arm grasp displacement) is scattered into a per-node drive signal `u` and fed into **every** round.
- **Multi-head decoder**: node-level heads (`acc`→integrated to `pos_next`/`vel_next`, `tension`, `contact`) + graph-level heads via mean-pool (`topology`, `failure`).

**Key design invariant — `failure` is not an independent signal.** Its training label is *derived* from ground-truth `tension`/`topology` (`derive_failure_label`: max tension over `tension_limit` OR topology in `stuck_topo_classes`). Keep failure as a function of other physical quantities, not a free-floating prediction.

### The schema is the contract — read it first

`dlo_wm/data/schema.py` defines all tensor conventions (`DLOState`, `DLOAction`, `build_edges`, `compute_edge_features`) and the dimension constants (`NODE_FEAT_DIM`, `EDGE_FEAT_DIM`, `ACTION_DIM`). Every data source must conform to it; the rest of the code revolves around it. Edge features are **recomputed every forward step** because `pos` changes during rollout.

### Two parallel model/data paths — keep them in sync

The repo has a **single-graph CPU path** and a **batched-graph GPU path** that are structurally identical but separate. A change to model/loss logic usually needs to land in **both**:

| concern | single-graph (CPU) | batched-graph (GPU) |
|---|---|---|
| model | `model/gnn.py` (`DLOWorldModel`) | `model/gnn_batched.py` (`BatchedDLOWorldModel`) |
| loss | `train/losses.py` (`world_model_loss`) | `model/gnn_batched.py` (`batched_loss`) |
| training | `train/trainer.py` | `scripts/train_ddp.py` |
| data assembly | `make_transition_batch` (per-sample) | `collate_transitions` (disjoint big graph) |

The batched path stacks B ropes into one **disjoint graph** (`data/batch.py`): nodes concatenated, edge indices offset by cumulative node count, `batch_idx` tracks node→graph, graph-level pooling uses `segment_mean`. `BatchedDLOWorldModel` reuses `mlp` and `InteractionLayer` from `gnn.py` — only the heads and pooling differ. **Note:** pushforward/noise-injection regularization currently lives only in the single-graph `trainer.py`, not in `train_ddp.py`.

### Data interface

Real data plugs in by implementing `TrajectoryProvider` (`data/dataset.py`) — implement `num_nodes` and `sample_trajectory()` returning `(states, actions, contact_pairs)`. Swap the `SyntheticRope(...)` line in the entry script; nothing else changes. **`SyntheticRope` is a self-consistent toy dynamics (springs + threshold contact), not real physics** — it exists only to exercise the pipeline and align dimensions.

Things the toy generator stubs that real data must replace: `_topology_label` (use Gauss linking / crossing number / knot invariants), tension/contact ground truth (from simulator or force sensors), and physics parameters as conditioning features with domain randomization.

## Config

Hyperparameters, multi-head loss `weights`, and the failure rule (`tension_limit`, `stuck_topo_classes`) live in `configs/default.py` (`DEFAULT_CONFIG`). `configs/a100.py` (`A100_CONFIG`) overrides with large-scale starting values and is merged on top of the default in `train_ddp.py`.

## Evaluation

`eval/rollout.py` measures **closed-loop multi-step** quality (the metric that matters for a simulator), not just single-step loss: `pos_rmse@k`, `tension_mae@k`, `contact_acc@k`, `topo_acc@k` over the horizon, plus `failure_acc`. During rollout there is no ground-truth contact, so contact edges are re-inferred geometrically via `edge_builder_from_contacts` (same `contact_radius` criterion as the toy generator).

## Known hard problems (per design notes)

- Long-horizon rollout error accumulation — mitigated by noise injection (pushforward approximation), tunable in config; only wired into the single-graph trainer so far.
- Topology-change instants (crossings/knots) require edge re-wiring; currently inferred purely from geometric distance. Complex topologies likely need a hybrid symbolic/geometric contact module.

## Note on language

Source comments and docstrings are written in Chinese. Match the surrounding language and density when editing existing files.
