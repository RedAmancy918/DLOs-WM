"""训练 workshop 材料条件化 DLO world model。

归一化统计只从 train episode 拟合，并与模型配置一起写入 checkpoint。

示例：
    python scripts/run_material.py \
        --data runs/workshop_data/train.pt \
        --out runs/workshop_material.pt
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from configs.default import DEFAULT_CONFIG
from dlo_wm.data.cached_provider import CachedTrajectoryProvider
from dlo_wm.data.dataset import slice_episode
from dlo_wm.data.normalization import MaterialFeatureNormalizer
from dlo_wm.data.provenance import (
    extract_dataset_contract,
    sha256_file,
    source_tree_sha256,
    validate_dataset_contract,
    validate_manifest_entry,
)
from dlo_wm.eval.material_rollout import evaluate_material_rollout
from dlo_wm.model.gnn import (
    DLOWorldModel,
    MaterialConditionedDLOWorldModel,
)
from dlo_wm.train.material_trainer import train_material_conditioned


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _git_dirty():
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip()) if result.returncode == 0 else "unknown"


def _training_source() -> dict:
    return {
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "source_tree_sha256": source_tree_sha256(
            REPO_ROOT,
            (
                "dlo_wm",
                "configs",
                "scripts/run_material.py",
                "scripts/eval_material.py",
            ),
        ),
    }


def _prepare_output(path: Path, protected, *, force: bool) -> None:
    if path.suffix != ".pt":
        raise ValueError(f"checkpoint 输出必须使用 .pt 后缀: {path}")
    protected = {Path(item).expanduser().resolve() for item in protected}
    if path in protected:
        raise ValueError(f"checkpoint 输出不能覆盖输入或 manifest: {path}")
    if path.exists() and not force:
        raise FileExistsError(
            f"checkpoint 已存在，拒绝覆盖；如确需替换请加 --force: {path}"
        )


def _atomic_torch_save(payload, path: Path, *, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _prepare_output(path, (), force=force)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.stem}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求了 CUDA，但当前 PyTorch 看不到可用 GPU")
    return device


def _uniform_macro_dt(provider: CachedTrajectoryProvider) -> float:
    values = [float(episode.macro_dt) for episode in provider.episodes]
    if max(values) - min(values) > 1e-9:
        raise ValueError(
            "当前模型只支持单一 dt，但 train split 含多个 macro_dt: "
            f"{sorted(set(values))}"
        )
    return values[0]


class _IgnoreMaterialTrainingAdapter(torch.nn.Module):
    """让旧 GNN baseline 复用相同训练日程，但其参数量更小。"""

    def __init__(self, base_model: DLOWorldModel):
        super().__init__()
        self.base_model = base_model

    def forward(
        self,
        state,
        drive,
        edge_index,
        is_contact,
        material_features,
    ):
        del material_features
        return self.base_model(state, drive, edge_index, is_contact)

    @torch.no_grad()
    def rollout(
        self,
        init_state,
        actions,
        edge_builder,
        material_features,
    ):
        del material_features
        return self.base_model.rollout(init_state, actions, edge_builder)


class _ZeroMaterialTrainingAdapter(torch.nn.Module):
    """保留条件模型容量，但把每个 episode 的材料向量强制置零。"""

    def __init__(self, base_model: MaterialConditionedDLOWorldModel):
        super().__init__()
        self.base_model = base_model

    def forward(
        self,
        state,
        drive,
        edge_index,
        is_contact,
        material_features,
    ):
        return self.base_model(
            state,
            drive,
            edge_index,
            is_contact,
            torch.zeros_like(material_features),
        )

    @torch.no_grad()
    def rollout(
        self,
        init_state,
        actions,
        edge_builder,
        material_features,
    ):
        return self.base_model.rollout(
            init_state,
            actions,
            edge_builder,
            torch.zeros_like(material_features),
        )


class _SequentialEpisodeProvider:
    """每次 validation callback 都从固定 episode 顺序开始。"""

    def __init__(self, episodes):
        self._episodes = tuple(episodes)
        self._index = 0

    def sample_episode(self, T=None):
        episode = self._episodes[self._index % len(self._episodes)]
        self._index += 1
        return slice_episode(episode, T=T)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="训练材料条件化 DLO world model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", default="runs/workshop_data/train.pt")
    parser.add_argument(
        "--val-data",
        default=None,
        help="缺省使用 train.pt 同目录的 val.pt",
    )
    parser.add_argument("--out", default="runs/workshop_material.pt")
    parser.add_argument("--force", action="store_true",
                        help="显式允许替换已有 checkpoint")
    parser.add_argument("--device", default="auto",
                        help="auto、cpu、cuda 或 cuda:<index>")
    parser.add_argument("--seed", type=int, default=2026)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--traj-len", type=int, default=None,
                        help="缺省使用 train split 中最短 episode")
    parser.add_argument("--traj-per-epoch", type=int, default=None,
                        help="缺省按 train episode 数采样")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--val-horizon", type=int, default=None,
                        help="缺省使用 min(20, val 最短 horizon)")
    parser.add_argument("--val-episodes", type=int, default=None,
                        help="缺省使用全部 validation episode")
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)

    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--message-passing", type=int, default=6)
    parser.add_argument("--n-topo-classes", type=int, default=3)
    parser.add_argument(
        "--model-type",
        choices=("conditioned", "conditioned_zero", "unconditioned"),
        default="conditioned",
        help=(
            "主模型、同容量零材料消融，或参数量更小的旧 GNN baseline"
        ),
    )
    parser.add_argument("--dt", type=float, default=None,
                        help="缺省从数据 macro_dt 推断；显式值必须与数据一致")

    parser.add_argument("--rollout-updates-per-epoch", type=int, default=4)
    parser.add_argument("--rollout-horizon", type=int, default=10)
    parser.add_argument("--rollout-weight", type=float, default=0.5)
    parser.add_argument("--contact-margin-scale", type=float, default=0.5)
    parser.add_argument("--tension-limit", type=float, default=1.5)
    parser.add_argument("--stuck-topo-classes", nargs="*", type=int,
                        default=(2,))

    parser.add_argument("--weight-pos", type=float, default=5.0)
    parser.add_argument("--weight-tension", type=float, default=0.5)
    parser.add_argument("--weight-contact", type=float, default=0.5)
    parser.add_argument("--weight-topo", type=float, default=0.3)
    parser.add_argument("--weight-fail", type=float, default=0.2)
    args = parser.parse_args()

    positive = {
        "epochs": args.epochs,
        "lr": args.lr,
        "grad-clip": args.grad_clip,
        "hidden": args.hidden,
        "message-passing": args.message_passing,
        "n-topo-classes": args.n_topo_classes,
        "rollout-horizon": args.rollout_horizon,
    }
    for name, value in positive.items():
        if value <= 0:
            parser.error(f"{name} 必须大于 0")
    if args.traj_len is not None and args.traj_len <= 0:
        parser.error("traj-len 必须大于 0")
    if args.traj_per_epoch is not None and args.traj_per_epoch <= 0:
        parser.error("traj-per-epoch 必须大于 0")
    if args.val_horizon is not None and args.val_horizon <= 0:
        parser.error("val-horizon 必须大于 0")
    if args.val_episodes is not None and args.val_episodes <= 0:
        parser.error("val-episodes 必须大于 0")
    if args.early_stop_patience <= 0 or args.early_stop_min_delta < 0:
        parser.error(
            "early-stop-patience 必须大于 0，min-delta 必须大于等于 0"
        )
    if args.rollout_updates_per_epoch < 0 or args.rollout_weight < 0:
        parser.error("rollout 更新次数和权重必须大于等于 0")
    if args.weight_decay < 0 or args.contact_margin_scale < 0:
        parser.error("weight-decay/contact-margin-scale 必须大于等于 0")
    return args


def main():
    args = _parse_args()
    device = _resolve_device(args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    training_source = _training_source()

    data_path = Path(args.data).expanduser().resolve()
    startup_integrity = validate_manifest_entry(data_path, "train")
    provider = CachedTrajectoryProvider(str(data_path), seed=args.seed)
    if sha256_file(data_path) != startup_integrity["sha256"]:
        raise RuntimeError("train 数据在校验与加载之间发生变化")
    if provider.format_version != 2:
        raise ValueError("材料条件化训练只接受 v2 episode 数据")
    declared_split = provider.metadata.get("split")
    if declared_split != "train":
        raise ValueError(
            f"训练数据必须显式声明 split='train'，实际为 {declared_split!r}"
        )

    episode_horizons = [episode.horizon for episode in provider.episodes]
    min_horizon = min(episode_horizons)
    traj_len = min_horizon if args.traj_len is None else args.traj_len
    if traj_len > min_horizon:
        raise ValueError(
            f"traj-len={traj_len} 超过最短 train episode horizon={min_horizon}"
        )
    if args.rollout_horizon > min_horizon:
        raise ValueError(
            f"rollout-horizon={args.rollout_horizon} 超过最短 "
            f"train episode horizon={min_horizon}"
        )

    data_dt = _uniform_macro_dt(provider)
    data_contract = extract_dataset_contract(provider, label="train")
    validate_dataset_contract(provider, data_contract, label="train")
    data_contact_margin = data_contract["contact_margin_scale"]
    if data_contact_margin is None or not math.isclose(
        args.contact_margin_scale,
        float(data_contact_margin),
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError(
            "训练 contact_margin_scale="
            f"{args.contact_margin_scale} 与数据生成契约 "
            f"{data_contact_margin!r} 不一致"
        )

    val_path = (
        data_path.with_name("val.pt")
        if args.val_data is None
        else Path(args.val_data).expanduser().resolve()
    )
    startup_val_integrity = validate_manifest_entry(val_path, "val")
    val_provider = CachedTrajectoryProvider(str(val_path), seed=0)
    if sha256_file(val_path) != startup_val_integrity["sha256"]:
        raise RuntimeError("validation 数据在校验与加载之间发生变化")
    if val_provider.format_version != 2:
        raise ValueError("validation 必须是 v2 episode 数据")
    if val_provider.metadata.get("split") != "val":
        raise ValueError(
            "validation 数据必须显式声明 split='val'，实际为 "
            f"{val_provider.metadata.get('split')!r}"
        )
    validate_dataset_contract(
        val_provider, data_contract, label="validation"
    )
    val_dt = _uniform_macro_dt(val_provider)
    if abs(val_dt - data_dt) > 1e-9:
        raise ValueError(
            f"validation macro_dt={val_dt} 与 train={data_dt} 不一致"
        )
    min_val_horizon = min(
        episode.horizon for episode in val_provider.episodes
    )
    val_horizon = (
        min(20, min_val_horizon)
        if args.val_horizon is None
        else args.val_horizon
    )
    if val_horizon > min_val_horizon:
        raise ValueError(
            f"val-horizon={val_horizon} 超过最短 validation episode "
            f"horizon={min_val_horizon}"
        )
    val_count = (
        val_provider.num_episodes
        if args.val_episodes is None
        else args.val_episodes
    )
    if val_count > val_provider.num_episodes:
        raise ValueError(
            f"请求 {val_count} 条 validation episode，但只有 "
            f"{val_provider.num_episodes} 条"
        )
    val_episodes = tuple(val_provider.episodes[:val_count])
    out = Path(args.out).expanduser().resolve()
    _prepare_output(
        out,
        (
            data_path,
            val_path,
            startup_integrity["manifest_path"],
            startup_val_integrity["manifest_path"],
        ),
        force=args.force,
    )

    if args.dt is not None and abs(args.dt - data_dt) > 1e-9:
        raise ValueError(
            f"显式 dt={args.dt} 与数据 macro_dt={data_dt} 不一致"
        )
    dt = data_dt if args.dt is None else args.dt

    max_topology = max(
        int(state.topology)
        for episode in provider.episodes
        for state in episode.states
    )
    if max_topology >= args.n_topo_classes:
        raise ValueError(
            f"数据 topology 最大值为 {max_topology}，但模型只有 "
            f"{args.n_topo_classes} 类"
        )

    # 归一化只能用 train split，绝不能混入 val/test/OOD。
    normalizer = MaterialFeatureNormalizer.fit(
        [episode.material for episode in provider.episodes]
    )
    model_config = {
        "hidden": args.hidden,
        "n_message_passing": args.message_passing,
        "n_topo_classes": args.n_topo_classes,
        "dt": dt,
    }
    if args.model_type in ("conditioned", "conditioned_zero"):
        model_config["material_input_dim"] = int(normalizer.mean.numel())
        core_model = MaterialConditionedDLOWorldModel(**model_config)
        training_model = (
            core_model
            if args.model_type == "conditioned"
            else _ZeroMaterialTrainingAdapter(core_model)
        )
        model_class = "MaterialConditionedDLOWorldModel"
    else:
        core_model = DLOWorldModel(**model_config)
        training_model = _IgnoreMaterialTrainingAdapter(core_model)
        model_class = "DLOWorldModel"

    cfg = dict(DEFAULT_CONFIG)
    cfg.update({
        "num_nodes": provider.num_nodes,
        "traj_len": traj_len,
        "traj_per_epoch": (
            provider.num_episodes
            if args.traj_per_epoch is None
            else args.traj_per_epoch
        ),
        "hidden": args.hidden,
        "n_message_passing": args.message_passing,
        "n_topo_classes": args.n_topo_classes,
        "dt": dt,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "epochs": args.epochs,
        "val_horizon": val_horizon,
        "val_episodes": val_count,
        "early_stop_patience": args.early_stop_patience,
        "early_stop_min_delta": args.early_stop_min_delta,
        "rollout_updates_per_epoch": args.rollout_updates_per_epoch,
        "rollout_horizon": args.rollout_horizon,
        "rollout_weight": args.rollout_weight,
        "contact_margin_scale": args.contact_margin_scale,
        "tension_limit": args.tension_limit,
        "stuck_topo_classes": list(args.stuck_topo_classes),
        "weights": {
            "pos": args.weight_pos,
            "tension": args.weight_tension,
            "contact": args.weight_contact,
            "topo": args.weight_topo,
            "fail": args.weight_fail,
        },
    })

    print(
        f"device={device} model_type={args.model_type} "
        f"episodes={provider.num_episodes}"
    )
    print("model_config:", json.dumps(model_config, ensure_ascii=False))
    print(
        "train_config:",
        json.dumps(
            {key: cfg[key] for key in (
                "epochs", "traj_len", "traj_per_epoch", "lr",
                "rollout_updates_per_epoch", "rollout_horizon",
                "rollout_weight", "val_horizon", "val_episodes",
                "early_stop_patience", "dt",
            )},
            ensure_ascii=False,
        ),
    )
    print(
        f"模型参数量: "
        f"{sum(p.numel() for p in core_model.parameters()) / 1e6:.2f}M"
    )

    def validation_fn(candidate_model):
        report = evaluate_material_rollout(
            candidate_model,
            _SequentialEpisodeProvider(val_episodes),
            n_episodes=val_count,
            horizon=val_horizon,
            device=device,
            normalizer=normalizer,
            shuffle_material=False,
            seed=0,
            contact_margin_scale=args.contact_margin_scale,
        )
        return report["metrics_at_horizon"]["position_nrmse"][
            str(val_horizon)
        ]["mean"]

    history = train_material_conditioned(
        training_model,
        provider,
        cfg,
        device=device,
        seed=args.seed,
        normalizer=normalizer,
        validation_fn=validation_fn,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
    )

    final_integrity = validate_manifest_entry(data_path, "train")
    if final_integrity != startup_integrity:
        raise RuntimeError("train 数据或 manifest 在训练期间发生变化")
    final_val_integrity = validate_manifest_entry(val_path, "val")
    if final_val_integrity != startup_val_integrity:
        raise RuntimeError("validation 数据或 manifest 在训练期间发生变化")
    if _training_source() != training_source:
        raise RuntimeError("训练源码在训练期间发生变化")
    best_epoch = min(
        range(len(history)),
        key=lambda index: history[index]["val_position_nrmse"],
    )
    best_validation = history[best_epoch]["val_position_nrmse"]

    checkpoint = {
        "checkpoint_version": 3,
        "model_type": args.model_type,
        "model_class": model_class,
        "state_dict": {
            name: value.detach().cpu()
            for name, value in core_model.state_dict().items()
        },
        "model_config": model_config,
        "num_parameters": sum(
            parameter.numel() for parameter in core_model.parameters()
        ),
        "config": cfg,
        "normalization": normalizer.to_dict(),
        "history": history,
        "seed": args.seed,
        "train_data": str(data_path),
        "train_data_sha256": startup_integrity["sha256"],
        "train_manifest": startup_integrity["manifest_path"],
        "train_manifest_sha256": startup_integrity["manifest_sha256"],
        "validation": {
            **startup_val_integrity,
            "horizon": val_horizon,
            "n_episodes": val_count,
            "metric": "position_nrmse",
            "best_epoch": best_epoch,
            "best_value": best_validation,
            "early_stopped": len(history) < args.epochs,
            "patience": args.early_stop_patience,
            "min_delta": args.early_stop_min_delta,
        },
        "dataset_contract": data_contract,
        "train_data_metadata": provider.metadata,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": training_source["git_commit"],
        "training_source": training_source,
    }
    _atomic_torch_save(checkpoint, out, force=args.force)
    print(f"[saved] checkpoint -> {out}")


if __name__ == "__main__":
    main()
