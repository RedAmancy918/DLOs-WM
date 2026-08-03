"""Workshop 数据的完整性与生成协议契约。

这里的 SHA256 用于发现意外覆盖/混用，不替代签名或不可信输入的安全校验。
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


EPISODE_CONTRACT_FIELDS = (
    "provider",
    "use_inextensible",
    "steps_interval",
    "settle_steps",
    "tension_scale",
    "contact_mode",
    "contact_margin_scale",
    "contact_distance_threshold",
)

BASE_PARAMETER_KEYS = (
    "K",
    "E",
    "G",
    "segment_mass",
    "radius",
    "mu_static",
    "mu_kinetic",
)

ACTION_PROTOCOL_KEYS = (
    "horizon",
    "num_nodes",
    "steps_interval",
    "max_disp",
    "lift_height",
    "fold_back_frac",
    "tension_scale",
)

ID_RANGE_KEYS = (
    "id_k_scale",
    "id_e_scale",
    "id_density_scale",
    "id_radius_scale",
)

MATERIAL_AXES = (
    "K",
    "E",
    "linear_density",
    "radius",
)

AXIS_TO_ID_RANGE = {
    "K": "id_k_scale",
    "E": "id_e_scale",
    "linear_density": "id_density_scale",
    "radius": "id_radius_scale",
}

SOURCE_SUFFIXES = {
    ".py", ".pyi", ".c", ".cc", ".cpp", ".h", ".hpp",
    ".cu", ".cuh", ".toml", ".yaml", ".yml", ".json",
}


def sha256_file(path: str | Path) -> str:
    path = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_tree_sha256(
    repo_root: str | Path,
    relative_entries,
) -> str:
    """按相对路径和内容哈希指定源码树，覆盖未提交/未跟踪源码。"""
    repo_root = Path(repo_root).expanduser().resolve()
    paths = []
    for entry in relative_entries:
        path = (repo_root / entry).resolve()
        try:
            path.relative_to(repo_root)
        except ValueError as error:
            raise ValueError(f"源码路径越出仓库: {path}") from error
        if path.is_file():
            paths.append(path)
        elif path.is_dir():
            paths.extend(
                candidate
                for candidate in path.rglob("*")
                if candidate.is_file()
                and candidate.suffix.lower() in SOURCE_SUFFIXES
                and "__pycache__" not in candidate.parts
            )
        else:
            raise ValueError(f"源码路径不存在: {path}")

    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        relative = path.relative_to(repo_root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def validate_manifest_entry(
    path: str | Path,
    expected_split: str,
) -> dict:
    """校验 split 文件与同目录 manifest 的文件名和 SHA256。"""
    path = Path(path).expanduser().resolve()
    manifest_path = path.parent / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"数据缺少同目录 manifest.json: {path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"无法读取数据 manifest: {manifest_path}") from error
    if not isinstance(manifest, dict) or manifest.get("format_version") != 2:
        raise ValueError(f"manifest format_version 非法: {manifest_path}")

    files = manifest.get("files")
    checksums = manifest.get("sha256")
    if not isinstance(files, dict) or not isinstance(checksums, dict):
        raise ValueError(f"manifest 缺少 files/sha256: {manifest_path}")
    declared_file = files.get(expected_split)
    expected_sha = checksums.get(expected_split)
    if not isinstance(declared_file, str) or not isinstance(expected_sha, str):
        raise ValueError(
            f"manifest 缺少 split={expected_split!r} 条目: {manifest_path}"
        )
    if Path(declared_file).name != path.name:
        raise ValueError(
            f"manifest 声明文件 {declared_file!r}，实际请求 {str(path)!r}"
        )
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha:
        raise ValueError(
            f"{expected_split} SHA256 与 manifest 不一致: "
            f"expected={expected_sha}, actual={actual_sha}"
        )
    return {
        "path": str(path),
        "sha256": actual_sha,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
    }


def _uniform_episode_field(provider, field: str, label: str):
    value = provider.episodes[0].metadata.get(field)
    if any(
        episode.metadata.get(field) != value
        for episode in provider.episodes
    ):
        raise ValueError(f"{label} split 的 episode metadata.{field} 不一致")
    return value


def _normalized_config_value(value):
    if isinstance(value, tuple):
        return [_normalized_config_value(item) for item in value]
    if isinstance(value, list):
        return [_normalized_config_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _normalized_config_value(item)
            for key, item in sorted(value.items())
        }
    return value


def extract_dataset_contract(provider, *, label: str) -> dict:
    """提取 train/eval 必须匹配的仿真、动作和派生特征契约。"""
    first = provider.episodes[0]
    contract = {
        "num_nodes": int(provider.num_nodes),
        "macro_dt": float(first.macro_dt),
    }
    for field in EPISODE_CONTRACT_FIELDS:
        contract[field] = _uniform_episode_field(provider, field, label)
    if contract["contact_distance_threshold"] is not None:
        raise ValueError(
            "Workshop rollout 目前只支持半径派生接触阈值，"
            "不接受显式 contact_distance_threshold"
        )

    generation = provider.metadata.get("generation")
    config = generation.get("config") if isinstance(generation, dict) else None
    if not isinstance(config, dict):
        raise ValueError(f"{label} 数据缺少 generation.config")
    required = set(
        BASE_PARAMETER_KEYS + ACTION_PROTOCOL_KEYS + ID_RANGE_KEYS
        + ("motions", "seed")
    )
    missing = sorted(key for key in required if key not in config)
    if missing:
        raise ValueError(f"{label} generation.config 缺少字段: {missing}")

    motions = _normalized_config_value(config["motions"])
    if not isinstance(motions, list) or not motions:
        raise ValueError(f"{label} generation.config.motions 非法")
    observed_tasks = {episode.task for episode in provider.episodes}
    if not observed_tasks.issubset(set(motions)):
        raise ValueError(
            f"{label} episode task={sorted(observed_tasks)} 超出 "
            f"generation motions={motions}"
        )

    contract["base_parameters"] = {
        key: _normalized_config_value(config[key])
        for key in BASE_PARAMETER_KEYS
    }
    contract["action_protocol"] = {
        key: _normalized_config_value(config[key])
        for key in ACTION_PROTOCOL_KEYS
    }
    contract["id_ranges"] = {
        key: _normalized_config_value(config[key])
        for key in ID_RANGE_KEYS
    }
    contract["motions"] = motions
    contract["seed"] = int(config["seed"])
    contract["seed_offsets"] = _normalized_config_value(
        config.get("seed_offsets", {})
    )
    contract["generation_source"] = {
        "git_commit": generation.get("git_commit", "unknown"),
        "git_dirty": generation.get("git_dirty", "unknown"),
        "source_tree_sha256": generation.get(
            "source_tree_sha256", "unknown"
        ),
        "dlolab_git_commit": generation.get(
            "dlolab_git_commit", "unknown"
        ),
        "dlolab_git_dirty": generation.get(
            "dlolab_git_dirty", "unknown"
        ),
        "dlolab_source_tree_sha256": generation.get(
            "dlolab_source_tree_sha256", "unknown"
        ),
    }

    if contract["action_protocol"]["num_nodes"] != contract["num_nodes"]:
        raise ValueError(f"{label} config.num_nodes 与 episode 不一致")
    expected_dt = 1e-3 * float(contract["action_protocol"]["steps_interval"])
    if not math.isclose(
        expected_dt, contract["macro_dt"], rel_tol=1e-6, abs_tol=1e-9
    ):
        raise ValueError(f"{label} steps_interval 与 macro_dt 不一致")
    return contract


def _material_axis_scales(episode, contract: dict) -> dict[str, float]:
    base = contract["base_parameters"]
    rest_length = float(episode.material.rest_length.sum())
    base_density = (
        episode.num_nodes * float(base["segment_mass"]) / rest_length
    )
    return {
        "K": float(episode.material.K) / float(base["K"]),
        "E": float(episode.material.E) / float(base["E"]),
        "linear_density": (
            float(episode.material.linear_density()) / base_density
        ),
        "radius": (
            float(episode.material.node_radius.mean())
            / float(base["radius"])
        ),
    }


def _inside(scale: float, bounds, tolerance=1e-5) -> bool:
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
        return False
    low, high = map(float, bounds)
    return low - tolerance <= scale <= high + tolerance


def _validate_id_support(provider, contract: dict, label: str) -> None:
    for episode in provider.episodes:
        scales = _material_axis_scales(episode, contract)
        for axis, scale in scales.items():
            bounds = contract["id_ranges"][AXIS_TO_ID_RANGE[axis]]
            if not _inside(scale, bounds):
                raise ValueError(
                    f"{label} episode={episode.episode_id!r} 的 {axis} "
                    f"scale={scale:.6g} 超出 ID range={bounds}"
                )


def _generation_config(provider, label: str) -> dict:
    generation = provider.metadata.get("generation")
    config = generation.get("config") if isinstance(generation, dict) else None
    if not isinstance(config, dict):
        raise ValueError(f"{label} 数据缺少 generation.config")
    return config


def _validate_ood_support(provider, contract: dict, label: str) -> None:
    config = _generation_config(provider, label)
    parameter = config.get("ood_parameter")
    tail = config.get("ood_tail")
    bounds = _normalized_config_value(config.get("resolved_ood_scale"))
    if parameter not in MATERIAL_AXES or tail not in {"low", "high"}:
        raise ValueError(f"{label} OOD parameter/tail 非法")
    if not isinstance(bounds, list) or len(bounds) != 2:
        raise ValueError(f"{label} resolved_ood_scale 非法: {bounds!r}")
    low, high = map(float, bounds)
    if low <= 0 or high < low:
        raise ValueError(f"{label} OOD range 非法: {bounds}")
    id_bounds = contract["id_ranges"][AXIS_TO_ID_RANGE[parameter]]
    id_low, id_high = map(float, id_bounds)
    if tail == "low" and not high < id_low:
        raise ValueError(f"{label} low-tail OOD 未完全落在 ID range 以下")
    if tail == "high" and not low > id_high:
        raise ValueError(f"{label} high-tail OOD 未完全落在 ID range 以上")

    for episode in provider.episodes:
        if episode.metadata.get("ood_parameter") != parameter:
            raise ValueError(f"{label} episode OOD parameter 与 generation 不一致")
        if episode.metadata.get("ood_tail") != tail:
            raise ValueError(f"{label} episode OOD tail 与 generation 不一致")
        declared = _normalized_config_value(
            episode.metadata.get("ood_scale_range")
        )
        if declared != bounds:
            raise ValueError(f"{label} episode OOD range 与 generation 不一致")
        scales = _material_axis_scales(episode, contract)
        if not _inside(scales[parameter], bounds):
            raise ValueError(
                f"{label} episode={episode.episode_id!r} 的目标轴 "
                f"{parameter} scale={scales[parameter]:.6g} 不在 OOD "
                f"range={bounds}"
            )
        for axis, scale in scales.items():
            if axis == parameter:
                continue
            axis_bounds = contract["id_ranges"][AXIS_TO_ID_RANGE[axis]]
            if not _inside(scale, axis_bounds):
                raise ValueError(
                    f"{label} episode={episode.episode_id!r} 的非目标轴 "
                    f"{axis} scale={scale:.6g} 超出 ID range={axis_bounds}"
                )


def _validate_counterfactual_protocol(provider, label: str) -> None:
    config = _generation_config(provider, label)
    parameter = config.get("cf_parameter")
    raw_scales = _normalized_config_value(config.get("cf_scales"))
    if parameter not in MATERIAL_AXES:
        raise ValueError(f"{label} cf_parameter 非法: {parameter!r}")
    if not isinstance(raw_scales, list) or len(raw_scales) < 2:
        raise ValueError(f"{label} cf_scales 非法: {raw_scales!r}")
    expected_scales = sorted(float(scale) for scale in raw_scales)
    if (
        len(set(expected_scales)) != len(expected_scales)
        or not all(math.isfinite(scale) and scale > 0 for scale in expected_scales)
        or not any(math.isclose(scale, 1.0, abs_tol=1e-9)
                   for scale in expected_scales)
    ):
        raise ValueError(f"{label} cf_scales 必须唯一、为正且包含 1.0")

    groups = {}
    for episode in provider.episodes:
        group_id = episode.metadata.get("counterfactual_group_id")
        if group_id is None:
            raise ValueError(f"{label} episode 缺少 counterfactual_group_id")
        if episode.metadata.get("counterfactual_parameter") != parameter:
            raise ValueError(f"{label} episode cf_parameter 与 generation 不一致")
        groups.setdefault(str(group_id), []).append(episode)
    for group_id, episodes in groups.items():
        actual_scales = sorted(
            float(episode.metadata.get("counterfactual_scale", float("nan")))
            for episode in episodes
        )
        if len(actual_scales) != len(expected_scales) or any(
            not math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-8)
            for actual, expected in zip(actual_scales, expected_scales)
        ):
            raise ValueError(
                f"{label} group={group_id!r} scales={actual_scales} "
                f"与 generation cf_scales={expected_scales} 不一致"
            )


def validate_dataset_contract(
    provider,
    expected: dict,
    *,
    label: str,
    counterfactual: bool = False,
    ood: bool = False,
) -> dict:
    """与 train contract 对照；CF 只允许 motion 子集且必须零沉降。"""
    actual = extract_dataset_contract(provider, label=label)
    expected_compare = dict(expected)
    actual_compare = dict(actual)
    expected_motions = expected_compare.pop("motions", None)
    actual_motions = actual_compare.pop("motions", None)
    expected_settle = expected_compare.pop("settle_steps", None)
    actual_settle = actual_compare.pop("settle_steps", None)

    if counterfactual and ood:
        raise ValueError("同一 split 不能同时声明 OOD 与 counterfactual")
    if counterfactual:
        if actual_settle != 0:
            raise ValueError(
                f"{label} paired-CF 必须 settle_steps=0，实际为 {actual_settle!r}"
            )
        if not set(actual_motions or ()).issubset(set(expected_motions or ())):
            raise ValueError(
                f"{label} motions={actual_motions} 不是 train motions="
                f"{expected_motions} 的子集"
            )
        if "random" in set(actual_motions or ()):
            raise ValueError(f"{label} paired-CF 不允许 random motion")
    else:
        actual_compare["settle_steps"] = actual_settle
        expected_compare["settle_steps"] = expected_settle
        actual_compare["motions"] = actual_motions
        expected_compare["motions"] = expected_motions

    if actual_compare != expected_compare:
        raise ValueError(
            f"{label} 数据契约与 train checkpoint 不一致:\n"
            f"  expected={expected_compare}\n  actual={actual_compare}"
        )
    if counterfactual:
        _validate_counterfactual_protocol(provider, label)
    elif ood:
        _validate_ood_support(provider, expected, label)
    else:
        _validate_id_support(provider, expected, label)
    return actual
