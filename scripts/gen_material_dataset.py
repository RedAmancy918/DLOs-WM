"""生成 workshop 材料条件化 v2 数据集。

一次运行可生成互不泄漏的 train / val / test_id / test_ood_material
episode split，以及共享控制目标、只改变一个材料参数的 counterfactual 组。
默认参数是先验证端到端通路的 workshop MVP；完整协议建议把三种 motion 都纳入，
并为每个 OOD 轴/尾部、每个反事实参数分别写入独立目录：

示例：
    python scripts/gen_material_dataset.py --out-dir runs/workshop_data
    python scripts/gen_material_dataset.py --only train val test_id \
        --motions random fold loop --out-dir runs/workshop_v1/id
    python scripts/gen_material_dataset.py --only test_ood_material \
        --motions random fold loop --ood-parameter K --ood-tail high \
        --out-dir runs/workshop_v1/ood_K_high
    python scripts/gen_material_dataset.py --only counterfactual \
        --motions loop fold --cf-parameter E \
        --out-dir runs/workshop_v1/cf_E
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dlo_wm.data.dlolab_provider import DLOLabProvider
from dlo_wm.data.material_sampling import MaterialRandomizationConfig
from dlo_wm.data.provenance import source_tree_sha256
from dlo_wm.data.serialization import FORMAT_VERSION, episode_to_dict


SPLITS = ("train", "val", "test_id", "test_ood_material", "counterfactual")
SEED_OFFSETS = {
    "train": 0,
    "val": 1_000_000,
    "test_id": 2_000_000,
    "test_ood_material": 3_000_000,
    "counterfactual": 4_000_000,
}
OOD_DEFAULTS = {
    "K": {"low": (0.4, 0.6), "high": (1.6, 2.0)},
    "E": {"low": (0.4, 0.6), "high": (1.6, 2.0)},
    "linear_density": {"low": (0.60, 0.75), "high": (1.30, 1.50)},
    "radius": {"low": (0.75, 0.85), "high": (1.15, 1.25)},
}


def _git_commit() -> str:
    """读取生成代码版本；在非 git 目录运行时给出明确占位符。"""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _git_dirty() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and bool(result.stdout.strip())


def _dlolab_git_info() -> dict:
    """从当前可导入的 Genesis 源码定位并指纹化 DLO-Lab。"""
    spec = importlib.util.find_spec("genesis")
    if spec is None or spec.origin is None:
        return {
            "git_commit": "unknown",
            "git_dirty": "unknown",
            "source_tree_sha256": "unknown",
        }
    repo = Path(spec.origin).resolve().parents[1]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo,
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
        "source_tree_sha256": source_tree_sha256(repo, ("genesis",)),
    }


def _id_randomization_kwargs(args) -> dict:
    return {
        "K_scale": tuple(args.id_k_scale),
        "E_scale": tuple(args.id_e_scale),
        "G_scale": (1.0, 1.0),
        "density_scale": tuple(args.id_density_scale),
        "radius_scale": tuple(args.id_radius_scale),
        "mu_static_scale": (1.0, 1.0),
        "mu_kinetic_scale": (1.0, 1.0),
    }


def _resolved_ood_scale(args) -> tuple[float, float]:
    if args.ood_scale is not None:
        return tuple(args.ood_scale)
    return OOD_DEFAULTS[args.ood_parameter][args.ood_tail]


def _randomization(args, *, ood: bool) -> MaterialRandomizationConfig:
    values = _id_randomization_kwargs(args)
    if ood:
        field = {
            "K": "K_scale",
            "E": "E_scale",
            "linear_density": "density_scale",
            "radius": "radius_scale",
        }[args.ood_parameter]
        # 只有指定轴越过训练范围；其余可辨识参数仍从 ID 分布采样。
        values[field] = _resolved_ood_scale(args)
    return MaterialRandomizationConfig(
        **values,
    )


def _generation_record(args, timestamp: str) -> dict:
    config = dict(vars(args))
    config["seed_offsets"] = dict(SEED_OFFSETS)
    config["resolved_ood_scale"] = list(_resolved_ood_scale(args))
    dlolab = _dlolab_git_info()
    return {
        "timestamp_utc": timestamp,
        "git_commit": _git_commit(),
        "source_tree_sha256": source_tree_sha256(
            REPO_ROOT,
            ("dlo_wm", "configs", "scripts/gen_material_dataset.py"),
        ),
        "dlolab_git_commit": dlolab["git_commit"],
        "dlolab_git_dirty": dlolab["git_dirty"],
        "dlolab_source_tree_sha256": dlolab["source_tree_sha256"],
        "git_dirty": _git_dirty(),
        "config": config,
    }


def _save_split(
    out_dir: Path,
    split: str,
    episodes,
    generation: dict,
    num_nodes: int,
    *,
    force: bool,
) -> Path:
    """只保存 primitive 字典；不把 dataclass 对象直接写进 pickle。"""
    if not episodes:
        raise ValueError(f"split {split!r} 没有 episode，拒绝写空数据集")
    payload = {
        "format_version": FORMAT_VERSION,
        "split": split,
        "num_nodes": num_nodes,
        "episodes": [episode_to_dict(episode) for episode in episodes],
        "generation": generation,
    }
    path = out_dir / f"{split}.pt"
    if path.exists() and not force:
        raise FileExistsError(
            f"生成期间目标 split 已出现，拒绝覆盖：{path}"
        )
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{split}.", suffix=".tmp", dir=out_dir, delete=False
        ) as handle:
            temporary = Path(handle.name)
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_manifest(path: Path, *, force: bool) -> dict:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if not isinstance(manifest, dict):
            raise TypeError("manifest 顶层必须是字典")
        if manifest.get("format_version") != FORMAT_VERSION:
            raise ValueError(
                f"manifest format_version={manifest.get('format_version')!r}"
            )
        for field in ("files", "episode_counts", "sha256"):
            value = manifest.get(field, {})
            if not isinstance(value, dict):
                raise TypeError(f"manifest.{field} 必须是字典")
        if not isinstance(manifest.get("generations", []), list):
            raise TypeError("manifest.generations 必须是列表")
        return manifest
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        if not force:
            raise RuntimeError(
                f"现有 manifest 无法安全合并：{path}；"
                "确认后可用 --force 重建"
            ) from error
        return {}


def _preflight_outputs(
    out_dir: Path,
    splits,
    existing_manifest: dict,
    *,
    force: bool,
) -> None:
    conflicts = []
    manifest_files = existing_manifest.get("files", {})
    for split in splits:
        target = out_dir / f"{split}.pt"
        if target.exists() or split in manifest_files:
            conflicts.append(str(target))
    if conflicts and not force:
        raise FileExistsError(
            "拒绝覆盖已冻结 split；如确需替换请显式加 --force:\n  "
            + "\n  ".join(conflicts)
        )


def _atomic_json_dump(payload: dict, path: Path) -> None:
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
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _generate_regular_split(provider, args, split: str, count: int):
    is_ood = split == "test_ood_material"
    provider.material_randomization = _randomization(args, ood=is_ood)
    offset = SEED_OFFSETS[split]
    episodes = []
    for index in range(count):
        provider.motion = args.motions[index % len(args.motions)]
        episode_seed = args.seed + offset + index
        control_seed = args.seed + offset + 500_000 + index
        metadata = {
            "split": split,
            "split_index": index,
            "motion_balance_index": index % len(args.motions),
        }
        if is_ood:
            metadata.update({
                "ood_parameter": args.ood_parameter,
                "ood_tail": args.ood_tail,
                "ood_scale_range": list(_resolved_ood_scale(args)),
            })
        episode = provider.sample_episode(
            T=args.horizon,
            seed=episode_seed,
            action_seed=control_seed,
            settle_steps=args.settle_steps,
            episode_id=f"{split}-{index:06d}",
            metadata=metadata,
        )
        episodes.append(episode)
        topo = sorted({int(state.topology) for state in episode.states})
        contact_steps = sum(
            int((state.contact > 0.5).any()) for state in episode.states
        )
        print(
            f"[{split} {index + 1}/{count}] topo={topo} "
            f"contact={contact_steps}/{len(episode.states)}"
        )
    return episodes


def _same_tensor(left, right, *, atol=1e-8) -> bool:
    if left is None or right is None:
        return left is None and right is None
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if left.is_floating_point():
        return bool(torch.allclose(left, right, rtol=0.0, atol=atol))
    return bool(torch.equal(left, right))


def _validate_generated_counterfactual_group(group, parameter: str) -> None:
    """保存前验证反事实组只改变指定材料轴且完整复用控制目标。"""
    if len(group) < 2:
        raise RuntimeError("反事实组至少需要两条 episode")
    group_ids = {
        episode.metadata.get("counterfactual_group_id") for episode in group
    }
    parameters = {
        episode.metadata.get("counterfactual_parameter") for episode in group
    }
    scales = [
        float(episode.metadata.get("counterfactual_scale", float("nan")))
        for episode in group
    ]
    if len(group_ids) != 1 or None in group_ids:
        raise RuntimeError("反事实组的 counterfactual_group_id 不一致")
    if parameters != {parameter}:
        raise RuntimeError("反事实组的 counterfactual_parameter 不一致")
    if (not bool(torch.isfinite(torch.tensor(scales)).all())
            or len(set(scales)) != len(scales)):
        raise RuntimeError("反事实组的 counterfactual_scale 缺失或重复")
    reference = group[0]
    if float(reference.metadata["counterfactual_scale"]) != 1.0:
        raise RuntimeError("反事实组首条 episode 必须是 scale=1.0 基线")
    parameter_value = lambda episode: {
        "K": episode.material.K,
        "E": episode.material.E,
        "linear_density": episode.material.linear_density(),
        "radius": episode.material.node_radius.mean(),
    }[parameter]
    baseline_value = parameter_value(reference)
    allowed_fields = {
        "K": {"K"},
        "E": {"E"},
        "linear_density": {"node_mass"},
        "radius": {"node_radius"},
    }[parameter]
    material_fields = (
        "rest_length", "node_mass", "node_radius", "K", "E", "G",
        "mu_self_static", "mu_self_kinetic",
    )
    changed_requested_axis = False
    for episode in group[1:]:
        if episode.horizon != reference.horizon or episode.task != reference.task:
            raise RuntimeError("反事实组 horizon/motion 不一致")
        if episode.metadata.get("control_seed") != reference.metadata.get(
            "control_seed"
        ):
            raise RuntimeError("反事实组 control_seed 不一致")
        if not _same_tensor(episode.states[0].pos, reference.states[0].pos):
            raise RuntimeError("反事实组 state0.pos 不一致")
        if not _same_tensor(episode.states[0].vel, reference.states[0].vel):
            raise RuntimeError("反事实组 state0.vel 不一致")
        for field in material_fields:
            same = _same_tensor(
                getattr(episode.material, field),
                getattr(reference.material, field),
            )
            if field in allowed_fields:
                changed_requested_axis |= not same
            elif not same:
                raise RuntimeError(
                    f"反事实组除 {parameter} 外还改变了 {field}"
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
                    raise RuntimeError(
                        f"反事实组 action[{step}].{field} 不一致"
                    )
        declared_scale = float(
            episode.metadata["counterfactual_scale"]
        )
        actual_scale = parameter_value(episode) / baseline_value
        if not torch.isclose(
            actual_scale,
            torch.tensor(
                declared_scale,
                dtype=actual_scale.dtype,
                device=actual_scale.device,
            ),
            rtol=1e-5,
            atol=1e-7,
        ):
            raise RuntimeError(
                f"反事实 metadata scale={declared_scale} 与实际 "
                f"{parameter} 倍率={float(actual_scale)} 不一致"
            )
    if not changed_requested_axis:
        raise RuntimeError(f"反事实组并未实际改变 {parameter}")


def _generate_counterfactual(provider, args):
    provider.material_randomization = None
    episodes = []
    scales = _counterfactual_scales(args.cf_scales)
    for group_index in range(args.n_counterfactual):
        provider.motion = args.motions[group_index % len(args.motions)]
        seed = args.seed + SEED_OFFSETS["counterfactual"] + group_index
        group_id = f"cf-{args.cf_parameter}-{group_index:06d}"
        group = provider.sample_counterfactual_group(
            T=args.horizon,
            parameter=args.cf_parameter,
            scales=scales,
            seed=seed,
            group_id=group_id,
        )
        for episode in group:
            episode.metadata.update({
                "split": "counterfactual",
                "counterfactual_group_index": group_index,
                "motion_balance_index": group_index % len(args.motions),
            })
        _validate_generated_counterfactual_group(group, args.cf_parameter)
        episodes.extend(group)
        print(
            f"[counterfactual {group_index + 1}/{args.n_counterfactual}] "
            f"group={group_id} motion={provider.motion} scales={scales}"
        )
    return episodes


def _counterfactual_scales(scales) -> list[float]:
    """把唯一的 1.0 基线排到首位，匹配 paired evaluator 的 reference 协议。"""
    baseline = [scale for scale in scales if abs(scale - 1.0) <= 1e-9]
    if len(baseline) != 1:
        raise ValueError("cf-scales 必须且只能包含一个 1.0 基线")
    return [1.0] + [float(scale) for scale in scales if abs(scale - 1.0) > 1e-9]


def _assert_episode_disjoint(split_episodes: dict[str, list]) -> None:
    """阻止 episode 或其随机控制序列跨常规 split 复用。"""
    seen_ids: dict[str, str] = {}
    seen_randomness: dict[tuple[int, int], str] = {}
    for split, episodes in split_episodes.items():
        for episode in episodes:
            if episode.episode_id in seen_ids:
                raise RuntimeError(
                    f"episode id {episode.episode_id!r} 同时出现在 "
                    f"{seen_ids[episode.episode_id]} 和 {split}"
                )
            seen_ids[episode.episode_id] = split
            if split == "counterfactual":
                # 反事实组内有意共享 seed/control_seed，不参与此项唯一性检查。
                continue
            key = (episode.seed, int(episode.metadata["control_seed"]))
            if key in seen_randomness:
                raise RuntimeError(
                    f"随机序列 {key} 同时出现在 {seen_randomness[key]} 和 {split}"
                )
            seen_randomness[key] = split


def _parse_args():
    parser = argparse.ArgumentParser(
        description="生成材料条件化 DLO-Lab v2 workshop MVP 数据集",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out-dir", default="runs/workshop_data")
    parser.add_argument(
        "--force",
        action="store_true",
        help="显式允许替换所选 split；未选择的 manifest 条目仍会保留",
    )
    parser.add_argument("--only", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--n-train", type=int, default=64)
    parser.add_argument("--n-val", type=int, default=12)
    parser.add_argument("--n-test-id", type=int, default=20)
    parser.add_argument("--n-test-ood-material", type=int, default=20)
    parser.add_argument("--n-counterfactual", type=int, default=8,
                        help="反事实组数量；每组包含 len(cf-scales) 条 episode")
    parser.add_argument("--horizon", type=int, default=18)
    parser.add_argument("--num-nodes", type=int, default=48)
    parser.add_argument("--steps-interval", type=int, default=120)
    parser.add_argument("--settle-steps", type=int, default=None)
    parser.add_argument(
        "--motions",
        nargs="+",
        choices=("random", "fold", "loop"),
        default=("loop",),
        help="按 episode/group 轮转，完整协议用 random fold loop",
    )
    parser.add_argument("--seed", type=int, default=2026)

    parser.add_argument("--K", type=float, default=5e4)
    parser.add_argument("--E", type=float, default=1e5)
    parser.add_argument("--G", type=float, default=1e4)
    parser.add_argument("--segment-mass", type=float, default=0.001)
    parser.add_argument("--radius", type=float, default=0.005)
    parser.add_argument("--mu-static", type=float, default=0.3)
    parser.add_argument("--mu-kinetic", type=float, default=0.25)
    parser.add_argument("--max-disp", type=float, default=0.02)
    parser.add_argument("--lift-height", type=float, default=0.011)
    parser.add_argument("--fold-back-frac", type=float, default=0.2)
    parser.add_argument("--tension-scale", type=float, default=1000.0)

    parser.add_argument("--id-k-scale", nargs=2, type=float,
                        default=(0.7, 1.4), metavar=("LOW", "HIGH"))
    parser.add_argument("--id-e-scale", nargs=2, type=float,
                        default=(0.7, 1.4), metavar=("LOW", "HIGH"))
    parser.add_argument("--id-density-scale", nargs=2, type=float,
                        default=(0.8, 1.2), metavar=("LOW", "HIGH"))
    parser.add_argument("--id-radius-scale", nargs=2, type=float,
                        default=(0.9, 1.1), metavar=("LOW", "HIGH"))
    parser.add_argument(
        "--ood-parameter",
        choices=("K", "E", "linear_density", "radius"),
        default="K",
        help="test_ood_material 唯一越过 ID 范围的材料轴",
    )
    parser.add_argument("--ood-tail", choices=("low", "high"), default="high")
    parser.add_argument(
        "--ood-scale",
        nargs=2,
        type=float,
        default=None,
        metavar=("LOW", "HIGH"),
        help="覆盖所选轴/尾部的默认 scale 范围",
    )

    parser.add_argument(
        "--cf-parameter",
        choices=("K", "E", "linear_density", "radius"),
        default="K",
    )
    parser.add_argument("--cf-scales", nargs="+", type=float,
                        default=(0.5, 0.7, 1.0, 1.4, 1.8))
    args = parser.parse_args()

    if args.horizon <= 0 or args.num_nodes < 2 or args.steps_interval <= 0:
        parser.error("horizon/steps-interval 必须为正，num-nodes 必须至少为 2")
    if args.settle_steps is not None and args.settle_steps < 0:
        parser.error("settle-steps 必须大于等于 0")
    if len(set(args.motions)) != len(args.motions):
        parser.error("motions 不应包含重复项")
    if len(set(args.only)) != len(args.only):
        parser.error("only 不应包含重复 split")
    if "counterfactual" in args.only and "random" in args.motions:
        parser.error(
            "counterfactual 不支持 random motion：其 target_pos 会随材料响应分叉；"
            "请使用 --motions loop fold"
        )
    if (len(args.cf_scales) < 2
            or len(set(args.cf_scales)) != len(args.cf_scales)
            or any(scale <= 0 for scale in args.cf_scales)):
        parser.error("cf-scales 至少两个、不得重复且必须全部为正数")
    try:
        _counterfactual_scales(args.cf_scales)
        id_ranges = {
            "K": tuple(args.id_k_scale),
            "E": tuple(args.id_e_scale),
            "linear_density": tuple(args.id_density_scale),
            "radius": tuple(args.id_radius_scale),
        }
        ood_low, ood_high = _resolved_ood_scale(args)
        id_low, id_high = id_ranges[args.ood_parameter]
        if ood_low <= 0 or ood_high < ood_low:
            raise ValueError("OOD scale 范围非法")
        if args.ood_tail == "low" and not ood_high < id_low:
            raise ValueError("low-tail OOD 必须完全低于对应 ID 范围")
        if args.ood_tail == "high" and not ood_low > id_high:
            raise ValueError("high-tail OOD 必须完全高于对应 ID 范围")
    except ValueError as error:
        parser.error(str(error))
    counts = {
        "train": args.n_train,
        "val": args.n_val,
        "test_id": args.n_test_id,
        "test_ood_material": args.n_test_ood_material,
        "counterfactual": args.n_counterfactual,
    }
    for split in args.only:
        if counts[split] <= 0:
            parser.error(f"所选 split {split} 的 episode 数必须大于 0")
    return args, counts


def main():
    args, counts = _parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    existing_manifest = _load_manifest(manifest_path, force=args.force)
    _preflight_outputs(
        out_dir,
        args.only,
        existing_manifest,
        force=args.force,
    )
    timestamp = datetime.now(timezone.utc).isoformat()
    generation = _generation_record(args, timestamp)

    provider = DLOLabProvider(
        num_nodes=args.num_nodes,
        segment_radius=args.radius,
        segment_mass=args.segment_mass,
        K=args.K,
        E=args.E,
        G=args.G,
        mu_self_static=args.mu_static,
        mu_self_kinetic=args.mu_kinetic,
        steps_interval=args.steps_interval,
        max_disp=args.max_disp,
        motion=args.motions[0],
        fold_back_frac=args.fold_back_frac,
        lift_height=args.lift_height,
        tension_scale=args.tension_scale,
        seed=args.seed,
        device="cpu",
    )

    started = time.time()
    generated: dict[str, list] = {}
    for split in args.only:
        if split == "counterfactual":
            episodes = _generate_counterfactual(provider, args)
        else:
            episodes = _generate_regular_split(
                provider, args, split, counts[split]
            )
        generated[split] = episodes

    _assert_episode_disjoint(generated)
    # 仿真可能运行数小时；落盘前重新读取，避免启动后的并行任务被静默覆盖。
    existing_manifest = _load_manifest(manifest_path, force=args.force)
    _preflight_outputs(
        out_dir,
        args.only,
        existing_manifest,
        force=args.force,
    )
    files = {}
    checksums = {}
    for split, episodes in generated.items():
        path = _save_split(
            out_dir,
            split,
            episodes,
            generation,
            args.num_nodes,
            force=args.force,
        )
        files[split] = str(path)
        checksums[split] = _sha256(path)
        print(f"[saved] {split}: {len(episodes)} episodes -> {path}")

    manifest = dict(existing_manifest)
    merged_files = dict(existing_manifest.get("files", {}))
    merged_counts = dict(existing_manifest.get("episode_counts", {}))
    merged_checksums = dict(existing_manifest.get("sha256", {}))
    merged_files.update(files)
    merged_counts.update({
        split: len(episodes) for split, episodes in generated.items()
    })
    merged_checksums.update(checksums)
    generations = list(existing_manifest.get("generations", []))
    generations.append({
        "timestamp_utc": timestamp,
        "git_commit": generation["git_commit"],
        "source_tree_sha256": generation["source_tree_sha256"],
        "dlolab_git_commit": generation["dlolab_git_commit"],
        "dlolab_git_dirty": generation["dlolab_git_dirty"],
        "dlolab_source_tree_sha256": generation[
            "dlolab_source_tree_sha256"
        ],
        "git_dirty": generation["git_dirty"],
        "config": generation["config"],
        "splits": list(generated),
        "episode_counts": {
            split: len(episodes) for split, episodes in generated.items()
        },
        "sha256": checksums,
        "elapsed_seconds": time.time() - started,
    })
    manifest.update({
        "format_version": FORMAT_VERSION,
        "created_at_utc": existing_manifest.get(
            "created_at_utc", timestamp
        ),
        "updated_at_utc": timestamp,
        "files": merged_files,
        "episode_counts": merged_counts,
        "sha256": merged_checksums,
        "generations": generations,
    })
    _atomic_json_dump(manifest, manifest_path)
    print(f"[done] manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
