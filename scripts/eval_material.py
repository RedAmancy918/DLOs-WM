"""评估 workshop 材料条件化模型的 ID / OOD / 反事实性能。

每个 ID/OOD split 都评估正确材料条件与 shuffled-condition 消融；反事实文件
按 counterfactual_group_id 分组后调用 paired effect 指标。

示例：
    python scripts/eval_material.py \
        --checkpoint runs/workshop_material.pt \
        --id-data runs/workshop_data/test_id.pt \
        --ood-data runs/workshop_data/test_ood_material.pt \
        --counterfactual-data runs/workshop_data/counterfactual.pt
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dlo_wm.data.cached_provider import CachedTrajectoryProvider
from dlo_wm.data.dataset import slice_episode
from dlo_wm.data.normalization import MaterialFeatureNormalizer
from dlo_wm.data.provenance import (
    sha256_file,
    source_tree_sha256,
    validate_dataset_contract,
    validate_manifest_entry,
)
from dlo_wm.eval.material_rollout import (
    evaluate_counterfactual_group,
    evaluate_material_rollout,
)
from dlo_wm.model.gnn import (
    DLOWorldModel,
    MaterialConditionedDLOWorldModel,
)


def _prepare_output(path: Path, protected, *, force: bool) -> None:
    if path.suffix != ".json":
        raise ValueError(f"评测输出必须使用 .json 后缀: {path}")
    protected = {Path(item).expanduser().resolve() for item in protected}
    if path in protected:
        raise ValueError(f"评测输出不能覆盖 checkpoint、数据或 manifest: {path}")
    if path.exists() and not force:
        raise FileExistsError(
            f"评测输出已存在，拒绝覆盖；如确需替换请加 --force: {path}"
        )


def _atomic_json_dump(payload: dict, path: Path, *, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _prepare_output(path, (), force=force)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.stem}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(
                _json_safe(payload),
                handle,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


class _SequentialEpisodeProvider:
    """让两次消融评估按同一顺序遍历 episode，避免随机重采样干扰。"""

    def __init__(self, episodes):
        self._episodes = tuple(episodes)
        self._index = 0

    def sample_episode(self, T=None):
        episode = self._episodes[self._index % len(self._episodes)]
        self._index += 1
        return slice_episode(episode, T=T)


def _current_training_source() -> dict:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "git_commit": (
            commit.stdout.strip() if commit.returncode == 0 else "unknown"
        ),
        "git_dirty": (
            bool(dirty.stdout.strip()) if dirty.returncode == 0 else "unknown"
        ),
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


class _UnconditionedRolloutAdapter(torch.nn.Module):
    """忽略材料编码，但保留逐 episode 半径构图和闭环 target_pos 控制。"""

    def __init__(self, base_model: DLOWorldModel):
        super().__init__()
        self.base_model = base_model

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


class _ZeroMaterialRolloutAdapter(torch.nn.Module):
    """同容量条件模型消融：rollout 中始终注入全零材料向量。"""

    def __init__(self, base_model: MaterialConditionedDLOWorldModel):
        super().__init__()
        self.base_model = base_model

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


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求了 CUDA，但当前 PyTorch 看不到可用 GPU")
    return device


def _load_eval_split(path: str, expected_split: str):
    integrity = validate_manifest_entry(path, expected_split)
    provider = CachedTrajectoryProvider(path, seed=0)
    if sha256_file(path) != integrity["sha256"]:
        raise RuntimeError(
            f"{expected_split} 数据在校验与加载之间发生变化"
        )
    if provider.format_version != 2:
        raise ValueError(f"{path} 不是带材料条件的 v2 episode 数据")
    declared = provider.metadata.get("split")
    if declared != expected_split:
        raise ValueError(
            f"{path} 声明 split={declared!r}，预期 {expected_split!r}"
        )
    return provider, integrity


def _validate_dataset_dt(provider, expected_dt: float, label: str) -> None:
    values = [float(episode.macro_dt) for episode in provider.episodes]
    if max(values) - min(values) > 1e-9:
        raise ValueError(
            f"{label} split 含多个 macro_dt: {sorted(set(values))}"
        )
    if abs(values[0] - expected_dt) > 1e-9:
        raise ValueError(
            f"{label} macro_dt={values[0]} 与 checkpoint dt={expected_dt} 不一致"
        )


def _select_episodes(provider, requested: int | None):
    count = provider.num_episodes if requested is None else requested
    if count <= 0:
        raise ValueError("n-episodes 必须大于 0")
    if count > provider.num_episodes:
        raise ValueError(
            f"请求 {count} 条 episode，但 split 只有 {provider.num_episodes} 条"
        )
    return list(provider.episodes[:count])


def _evaluate_split(
    model,
    episodes,
    *,
    horizon,
    device,
    normalizer,
    seed,
    contact_margin_scale,
    model_type,
):
    min_horizon = min(episode.horizon for episode in episodes)
    actual_horizon = min_horizon if horizon is None else horizon
    if actual_horizon <= 0 or actual_horizon > min_horizon:
        raise ValueError(
            f"评估 horizon={actual_horizon}，但最短 episode 为 {min_horizon}"
        )
    common = {
        "n_episodes": len(episodes),
        "horizon": actual_horizon,
        "device": device,
        "normalizer": normalizer,
        "contact_margin_scale": contact_margin_scale,
    }
    correct = evaluate_material_rollout(
        model,
        _SequentialEpisodeProvider(episodes),
        shuffle_material=False,
        seed=seed,
        **common,
    )
    if model_type == "conditioned":
        if len(episodes) < 2:
            raise ValueError("shuffled-condition 至少需要两条 episode")
        shuffled = evaluate_material_rollout(
            model,
            _SequentialEpisodeProvider(episodes),
            shuffle_material=True,
            seed=seed,
            **common,
        )
        shuffled["shuffle_seed"] = seed
        shuffled["derangement_verified"] = True
    else:
        shuffled = {
            "status": "not_applicable",
            "reason": f"{model_type} 不读取真实材料条件",
        }
    return {
        "correct_condition": correct,
        "shuffled_condition": shuffled,
    }


def _same_tensor(left, right, *, atol=1e-8) -> bool:
    if left is None or right is None:
        return left is None and right is None
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if left.is_floating_point():
        return bool(torch.allclose(left, right, rtol=0.0, atol=atol))
    return bool(torch.equal(left, right))


def _validate_counterfactual_group(group_id: str, episodes):
    """验证 paired 协议，并把唯一 scale=1.0 episode 放到 reference 首位。"""
    parameters = {
        episode.metadata.get("counterfactual_parameter")
        for episode in episodes
    }
    if len(parameters) != 1:
        raise ValueError(f"反事实组 {group_id!r} 的 parameter 不一致")
    parameter = next(iter(parameters))
    allowed_fields = {
        "K": {"K"},
        "E": {"E"},
        "linear_density": {"node_mass"},
        "radius": {"node_radius"},
    }
    if parameter not in allowed_fields:
        raise ValueError(f"反事实组 {group_id!r} 使用不可辨识参数 {parameter!r}")

    scales = [
        float(episode.metadata.get("counterfactual_scale", float("nan")))
        for episode in episodes
    ]
    if len(set(scales)) != len(scales) or not all(math.isfinite(x) for x in scales):
        raise ValueError(f"反事实组 {group_id!r} 的 scale 缺失或重复")
    baseline_indices = [
        index for index, scale in enumerate(scales)
        if abs(scale - 1.0) <= 1e-9
    ]
    if len(baseline_indices) != 1:
        raise ValueError(f"反事实组 {group_id!r} 必须且只能有一个 scale=1.0")
    baseline_index = baseline_indices[0]
    ordered = [episodes[baseline_index]] + [
        episode for index, episode in enumerate(episodes)
        if index != baseline_index
    ]
    reference = ordered[0]

    material_fields = (
        "rest_length", "node_mass", "node_radius", "K", "E", "G",
        "mu_self_static", "mu_self_kinetic",
    )
    changed_requested_axis = False
    for episode in ordered[1:]:
        if episode.horizon != reference.horizon:
            raise ValueError(f"反事实组 {group_id!r} horizon 不一致")
        if episode.task != reference.task:
            raise ValueError(f"反事实组 {group_id!r} motion/task 不一致")
        if episode.metadata.get("control_seed") != reference.metadata.get(
            "control_seed"
        ):
            raise ValueError(f"反事实组 {group_id!r} control_seed 不一致")
        if not _same_tensor(episode.states[0].pos, reference.states[0].pos):
            raise ValueError(f"反事实组 {group_id!r} state0.pos 不一致")
        if not _same_tensor(episode.states[0].vel, reference.states[0].vel):
            raise ValueError(f"反事实组 {group_id!r} state0.vel 不一致")

        for field in material_fields:
            same = _same_tensor(
                getattr(episode.material, field),
                getattr(reference.material, field),
            )
            if field in allowed_fields[parameter]:
                changed_requested_axis |= not same
            elif not same:
                raise ValueError(
                    f"反事实组 {group_id!r} 除 {parameter} 外还改变了 {field}"
                )

        for step, (action, reference_action) in enumerate(zip(
            episode.actions, reference.actions
        )):
            for field in (
                "grasp_idx", "gripper_id", "grasp_active",
                "target_pos", "duration",
            ):
                if not _same_tensor(
                    getattr(action, field), getattr(reference_action, field)
                ):
                    raise ValueError(
                        f"反事实组 {group_id!r} action[{step}].{field} 不一致"
                    )
    if not changed_requested_axis:
        raise ValueError(f"反事实组 {group_id!r} 并未实际改变 {parameter}")
    return ordered


def _evaluate_counterfactual(
    model,
    provider,
    *,
    device,
    normalizer,
    contact_margin_scale,
    max_groups,
    model_type,
    seed,
):
    groups = {}
    skipped = []
    for episode in provider.episodes:
        group_id = episode.metadata.get("counterfactual_group_id")
        if group_id is None:
            skipped.append(episode.episode_id)
            continue
        groups.setdefault(str(group_id), []).append(episode)

    if skipped:
        raise ValueError(
            "counterfactual split 含缺少 counterfactual_group_id 的 episode: "
            f"{skipped}"
        )
    if not groups:
        raise ValueError("counterfactual split 不包含任何有效组")

    group_items = sorted(groups.items())
    if max_groups is not None:
        group_items = group_items[:max_groups]
    reports = []
    for group_index, (group_id, episodes) in enumerate(group_items):
        if len(episodes) < 2:
            raise ValueError(f"反事实组 {group_id!r} 少于两条 episode")
        episodes = _validate_counterfactual_group(group_id, episodes)
        correct = evaluate_counterfactual_group(
            model,
            episodes,
            device=device,
            normalizer=normalizer,
            contact_margin_scale=contact_margin_scale,
            shuffle_material=False,
            seed=seed + group_index,
        )
        if model_type == "conditioned":
            shuffled = evaluate_counterfactual_group(
                model,
                episodes,
                device=device,
                normalizer=normalizer,
                contact_margin_scale=contact_margin_scale,
                shuffle_material=True,
                seed=seed + group_index,
            )
        else:
            shuffled = {
                "status": "not_applicable",
                "reason": f"{model_type} 不读取真实材料条件",
            }
        reports.append({
            "counterfactual_group_id": group_id,
            "correct_condition": correct,
            "shuffled_condition": shuffled,
        })
    return {
        "n_groups": len(reports),
        "groups": reports,
    }


def _json_safe(value):
    """把非有限指标转成 null，确保输出是严格 JSON。"""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _parse_args():
    parser = argparse.ArgumentParser(
        description="评估材料条件化 world model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", default="runs/workshop_material.pt")
    parser.add_argument("--id-data", default="runs/workshop_data/test_id.pt")
    parser.add_argument(
        "--ood-data", default="runs/workshop_data/test_ood_material.pt"
    )
    parser.add_argument(
        "--counterfactual-data",
        default="runs/workshop_data/counterfactual.pt",
    )
    parser.add_argument("--out", default="runs/workshop_eval.json")
    parser.add_argument("--force", action="store_true",
                        help="显式允许替换已有评测 JSON")
    parser.add_argument("--device", default="auto",
                        help="auto、cpu、cuda 或 cuda:<index>")
    parser.add_argument("--n-episodes", type=int, default=None,
                        help="每个 ID/OOD split 的 episode 数；缺省为全部")
    parser.add_argument("--horizon", type=int, default=None,
                        help="缺省使用所评 split 的最短 episode horizon")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--contact-margin-scale", type=float, default=0.5)
    parser.add_argument("--max-counterfactual-groups", type=int, default=None)
    args = parser.parse_args()
    if args.n_episodes is not None and args.n_episodes <= 0:
        parser.error("n-episodes 必须大于 0")
    if args.horizon is not None and args.horizon <= 0:
        parser.error("horizon 必须大于 0")
    if args.contact_margin_scale < 0:
        parser.error("contact-margin-scale 必须大于等于 0")
    if (args.max_counterfactual_groups is not None
            and args.max_counterfactual_groups <= 0):
        parser.error("max-counterfactual-groups 必须大于 0")
    return args


def main():
    args = _parse_args()
    device = _resolve_device(args.device)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    eval_inputs = [
        checkpoint_path,
        Path(args.id_data).expanduser().resolve(),
        Path(args.ood_data).expanduser().resolve(),
    ]
    if args.counterfactual_data:
        eval_inputs.append(
            Path(args.counterfactual_data).expanduser().resolve()
        )
    _prepare_output(
        out,
        eval_inputs + [path.parent / "manifest.json" for path in eval_inputs[1:]],
        force=args.force,
    )
    checkpoint_sha256 = sha256_file(checkpoint_path)
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    if checkpoint.get("checkpoint_version") != 3:
        raise ValueError(
            "不支持的 checkpoint_version="
            f"{checkpoint.get('checkpoint_version')!r}；当前仅支持 3"
        )
    model_type = checkpoint.get("model_type")
    if model_type is None:
        model_type = (
            "conditioned"
            if checkpoint.get("model_class")
            == "MaterialConditionedDLOWorldModel"
            else "unconditioned"
        )
    expected_class = {
        "conditioned": "MaterialConditionedDLOWorldModel",
        "conditioned_zero": "MaterialConditionedDLOWorldModel",
        "unconditioned": "DLOWorldModel",
    }.get(model_type)
    if expected_class is None or checkpoint.get("model_class") != expected_class:
        raise ValueError(
            f"checkpoint model_type/model_class 不兼容: "
            f"{model_type!r}/{checkpoint.get('model_class')!r}"
        )
    for key in (
        "state_dict", "model_config", "config", "normalization",
        "train_data", "train_data_sha256", "train_manifest",
        "train_manifest_sha256", "validation", "dataset_contract",
        "training_source",
    ):
        if key not in checkpoint:
            raise ValueError(f"checkpoint 缺少字段 {key!r}")

    recorded_source = checkpoint["training_source"]
    current_source = _current_training_source()
    for key in ("git_commit", "source_tree_sha256"):
        if recorded_source.get(key) != current_source.get(key):
            raise ValueError(
                f"评估源码 {key} 与 checkpoint 不一致: "
                f"recorded={recorded_source.get(key)!r}, "
                f"current={current_source.get(key)!r}"
            )

    train_provenance = {
        "data_path": checkpoint["train_data"],
        "data_sha256": checkpoint["train_data_sha256"],
        "manifest_path": checkpoint["train_manifest"],
        "manifest_sha256": checkpoint["train_manifest_sha256"],
        "locally_verified": False,
    }
    train_path = Path(checkpoint["train_data"]).expanduser()
    train_manifest_path = Path(checkpoint["train_manifest"]).expanduser()
    if train_path.is_file() != train_manifest_path.is_file():
        raise ValueError(
            "checkpoint 记录的 train 数据与 manifest 只存在其一"
        )
    if train_path.is_file() and train_manifest_path.is_file():
        current_train = validate_manifest_entry(train_path, "train")
        if current_train["sha256"] != checkpoint["train_data_sha256"]:
            raise ValueError("checkpoint 记录的 train 数据 SHA256 已失配")
        if current_train["manifest_sha256"] != checkpoint[
            "train_manifest_sha256"
        ]:
            raise ValueError("checkpoint 记录的 train manifest SHA256 已失配")
        train_provenance["locally_verified"] = True

    validation_provenance = dict(checkpoint["validation"])
    for key in (
        "path", "sha256", "manifest_path", "manifest_sha256",
        "horizon", "n_episodes", "metric", "best_epoch", "best_value",
    ):
        if key not in validation_provenance:
            raise ValueError(f"checkpoint validation 缺少字段 {key!r}")
    validation_provenance["locally_verified"] = False
    validation_path = Path(validation_provenance["path"]).expanduser()
    validation_manifest_path = Path(
        validation_provenance["manifest_path"]
    ).expanduser()
    _prepare_output(
        out,
        (
            train_path,
            train_manifest_path,
            validation_path,
            validation_manifest_path,
        ),
        force=args.force,
    )
    if validation_path.is_file() != validation_manifest_path.is_file():
        raise ValueError(
            "checkpoint 记录的 validation 数据与 manifest 只存在其一"
        )
    if validation_path.is_file() and validation_manifest_path.is_file():
        current_validation = validate_manifest_entry(validation_path, "val")
        if current_validation["sha256"] != validation_provenance["sha256"]:
            raise ValueError("checkpoint 记录的 validation 数据 SHA256 已失配")
        if current_validation["manifest_sha256"] != validation_provenance[
            "manifest_sha256"
        ]:
            raise ValueError(
                "checkpoint 记录的 validation manifest SHA256 已失配"
            )
        validation_provenance["locally_verified"] = True

    normalizer = MaterialFeatureNormalizer.from_dict(
        checkpoint["normalization"]
    )
    model_config = dict(checkpoint["model_config"])
    if "dt" not in model_config:
        raise ValueError("checkpoint model_config 缺少由 train macro_dt 推断的 dt")
    if model_type in ("conditioned", "conditioned_zero"):
        if model_config.get("material_input_dim") != normalizer.mean.numel():
            raise ValueError("checkpoint 模型材料维度与 normalization 不一致")
        base_model = MaterialConditionedDLOWorldModel(**model_config)
        base_model.load_state_dict(checkpoint["state_dict"], strict=True)
        model = (
            base_model
            if model_type == "conditioned"
            else _ZeroMaterialRolloutAdapter(base_model)
        )
    else:
        base_model = DLOWorldModel(**model_config)
        base_model.load_state_dict(checkpoint["state_dict"], strict=True)
        model = _UnconditionedRolloutAdapter(base_model)
    model.to(device).eval()

    id_provider, id_integrity = _load_eval_split(
        args.id_data, "test_id"
    )
    ood_provider, ood_integrity = _load_eval_split(
        args.ood_data, "test_ood_material"
    )
    expected_contract = checkpoint["dataset_contract"]
    validate_dataset_contract(
        id_provider, expected_contract, label="ID"
    )
    validate_dataset_contract(
        ood_provider, expected_contract, label="OOD", ood=True
    )
    trained_contact_margin = float(
        checkpoint["config"].get("contact_margin_scale", 0.5)
    )
    if not math.isclose(
        args.contact_margin_scale,
        trained_contact_margin,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError(
            f"eval contact_margin_scale={args.contact_margin_scale} 与训练 "
            f"{trained_contact_margin} 不一致"
        )
    _validate_dataset_dt(id_provider, float(model_config["dt"]), "ID")
    _validate_dataset_dt(ood_provider, float(model_config["dt"]), "OOD")
    id_episodes = _select_episodes(id_provider, args.n_episodes)
    ood_episodes = _select_episodes(ood_provider, args.n_episodes)

    print(
        f"device={device} model_type={model_type} "
        f"ID={len(id_episodes)} OOD={len(ood_episodes)} "
        f"checkpoint={checkpoint_path}"
    )
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_git_commit": checkpoint.get("git_commit", "unknown"),
        "source_verification": {
            "recorded": recorded_source,
            "current": current_source,
            "commit_and_tree_match": True,
        },
        "train_provenance": train_provenance,
        "validation_selection": validation_provenance,
        "model_type": model_type,
        "model_config": model_config,
        "num_parameters": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "datasets": {
            "id": {
                **id_integrity,
                "metadata": id_provider.metadata,
            },
            "ood_material": {
                **ood_integrity,
                "metadata": ood_provider.metadata,
            },
        },
        "id": _evaluate_split(
            model,
            id_episodes,
            horizon=args.horizon,
            device=device,
            normalizer=normalizer,
            seed=args.seed,
            contact_margin_scale=args.contact_margin_scale,
            model_type=model_type,
        ),
        "ood_material": _evaluate_split(
            model,
            ood_episodes,
            horizon=args.horizon,
            device=device,
            normalizer=normalizer,
            seed=args.seed,
            contact_margin_scale=args.contact_margin_scale,
            model_type=model_type,
        ),
    }

    if args.counterfactual_data:
        cf_provider, cf_integrity = _load_eval_split(
            args.counterfactual_data, "counterfactual"
        )
        validate_dataset_contract(
            cf_provider,
            expected_contract,
            label="counterfactual",
            counterfactual=True,
        )
        _validate_dataset_dt(
            cf_provider, float(model_config["dt"]), "counterfactual"
        )
        report["datasets"]["counterfactual"] = {
            **cf_integrity,
            "metadata": cf_provider.metadata,
        }
        report["counterfactual"] = _evaluate_counterfactual(
            model,
            cf_provider,
            device=device,
            normalizer=normalizer,
            contact_margin_scale=args.contact_margin_scale,
            max_groups=args.max_counterfactual_groups,
            model_type=model_type,
            seed=args.seed,
        )

    for split, integrity in (
        ("test_id", id_integrity),
        ("test_ood_material", ood_integrity),
    ):
        if validate_manifest_entry(integrity["path"], split) != integrity:
            raise RuntimeError(f"{split} 数据或 manifest 在评估期间发生变化")
    if args.counterfactual_data:
        if validate_manifest_entry(
            cf_integrity["path"], "counterfactual"
        ) != cf_integrity:
            raise RuntimeError(
                "counterfactual 数据或 manifest 在评估期间发生变化"
            )

    if sha256_file(checkpoint_path) != checkpoint_sha256:
        raise RuntimeError("checkpoint 在评估期间发生变化")

    _atomic_json_dump(report, out, force=args.force)
    print(f"[saved] report -> {out}")


if __name__ == "__main__":
    main()
