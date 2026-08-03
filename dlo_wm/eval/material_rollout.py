"""Workshop 用材料条件化闭环 rollout 指标。"""

from __future__ import annotations

import math
import random

import torch

from ..train.trainer import edge_builder_from_material


def _validate_model_dt(model, macro_dt):
    model_dt = getattr(model, "dt", None)
    for attribute in ("model", "base_model"):
        if model_dt is None and hasattr(model, attribute):
            model_dt = getattr(getattr(model, attribute), "dt", None)
    if model_dt is None or not math.isclose(
        float(model_dt), float(macro_dt), rel_tol=1e-6, abs_tol=1e-9
    ):
        raise ValueError(
            f"模型 dt={model_dt!r} 与 episode macro_dt={macro_dt!r} 不一致"
        )


def _chamfer_distance(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    distance = torch.cdist(pred, target)
    return 0.5 * (
        distance.min(dim=1).values.mean()
        + distance.min(dim=0).values.mean()
    )


def _edge_length_violation(state, rest_length):
    length = (state.pos[1:] - state.pos[:-1]).norm(dim=-1)
    return ((length - rest_length).abs() / rest_length.clamp(min=1e-8)).mean()


def _horizon_indices(horizon: int, requested=(1, 5, 10, 20)):
    return sorted({min(max(1, int(step)), horizon) for step in requested})


def _episode_material_features(episode, normalizer):
    features = episode.material.global_features()
    return features if normalizer is None else normalizer.transform(features)


@torch.no_grad()
def evaluate_material_rollout(
    model,
    provider,
    *,
    n_episodes: int,
    horizon: int,
    device="cpu",
    normalizer=None,
    shuffle_material: bool = False,
    seed: int = 0,
    contact_margin_scale: float = 0.5,
    divergence_fraction: float = 0.25,
):
    """评估 ID/OOD rollout；可固定图构造并仅打乱 encoder 的材料输入。"""
    if n_episodes <= 0 or horizon <= 0:
        raise ValueError("n_episodes 和 horizon 必须大于 0")
    model.to(device).eval()
    episodes = [
        provider.sample_episode(T=horizon).to(device).validate()
        for _ in range(n_episodes)
    ]
    short_episodes = [
        episode.episode_id
        for episode in episodes
        if episode.horizon < horizon
    ]
    if short_episodes:
        raise ValueError(
            f"请求 horizon={horizon}，但 episode 长度不足: "
            f"{short_episodes}"
        )
    feature_order = list(range(n_episodes))
    shuffle_offset = 0
    if shuffle_material:
        if n_episodes < 2:
            raise ValueError("material shuffle 至少需要两个 episode")
        # 随机循环移位是严格 derangement：每条 episode 都拿到别人的材料，
        # 不会像普通 shuffle 那样残留固定点而稀释消融效应。
        offset = random.Random(seed).randrange(1, n_episodes)
        shuffle_offset = offset
        feature_order = [
            (index + offset) % n_episodes
            for index in range(n_episodes)
        ]

    steps = _horizon_indices(horizon)
    per_step = {
        key: {step: [] for step in steps}
        for key in (
            "position_rmse",
            "position_nrmse",
            "velocity_rmse",
            "chamfer",
            "chamfer_normalized",
            "tension_mae",
            "edge_length_violation",
        )
    }
    tp = fp = fn = 0
    topology_correct = topology_total = 0
    divergent = 0
    per_episode = []

    for episode_index, episode in enumerate(episodes):
        _validate_model_dt(model, episode.macro_dt)
        feature_episode = episodes[feature_order[episode_index]]
        features = _episode_material_features(feature_episode, normalizer)
        # 图几何仍用当前 episode 的真实半径；shuffle 只扰动 model condition，
        # 避免把“错误构图”混进材料消融。
        builder = edge_builder_from_material(
            episode.material, contact_margin_scale
        )
        pred = model.rollout(
            episode.states[0],
            episode.actions[:horizon],
            builder,
            features,
        )
        episode_log = {
            "episode_id": episode.episode_id,
            "rope_length": float(episode.material.rest_length.sum()),
            "metrics_at_horizon": {},
        }
        episode_tp = episode_fp = episode_fn = 0
        episode_topology_correct = episode_topology_total = 0
        rope_length = float(episode.material.rest_length.sum())
        final_rmse = float("nan")

        for step in range(1, min(horizon, episode.horizon) + 1):
            target = episode.states[step]
            estimate = pred[step]
            pos_rmse = torch.sqrt(
                ((estimate.pos - target.pos) ** 2).mean()
            )
            vel_rmse = torch.sqrt(
                ((estimate.vel - target.vel) ** 2).mean()
            )
            chamfer = _chamfer_distance(estimate.pos, target.pos)
            tension_mae = (
                estimate.tension - target.tension
            ).abs().mean()
            edge_violation = _edge_length_violation(
                estimate,
                episode.material.rest_length.to(estimate.pos.device),
            )
            final_rmse = float(pos_rmse)
            if step in steps:
                values = (
                    pos_rmse,
                    pos_rmse / max(rope_length, 1e-8),
                    vel_rmse,
                    chamfer,
                    chamfer / max(rope_length, 1e-8),
                    tension_mae,
                    edge_violation,
                )
                for key, value in zip(per_step, values):
                    per_step[key][step].append(float(value))
                episode_log["metrics_at_horizon"][str(step)] = {
                    key: float(value)
                    for key, value in zip(per_step, values)
                }

            predicted_contact = estimate.contact > 0.5
            target_contact = target.contact > 0.5
            tp += int((predicted_contact & target_contact).sum())
            fp += int((predicted_contact & ~target_contact).sum())
            fn += int((~predicted_contact & target_contact).sum())
            step_tp = int((predicted_contact & target_contact).sum())
            step_fp = int((predicted_contact & ~target_contact).sum())
            step_fn = int((~predicted_contact & target_contact).sum())
            episode_tp += step_tp
            episode_fp += step_fp
            episode_fn += step_fn
            step_topology_correct = int(
                estimate.topology == target.topology
            )
            topology_correct += step_topology_correct
            topology_total += 1
            episode_topology_correct += step_topology_correct
            episode_topology_total += 1

        bad = (
            not math.isfinite(final_rmse)
            or final_rmse > divergence_fraction * max(rope_length, 1e-8)
        )
        divergent += int(bad)
        episode_log["final_position_rmse"] = final_rmse
        episode_log["diverged"] = bad
        episode_log["self_contact_counts"] = {
            "tp": episode_tp,
            "fp": episode_fp,
            "fn": episode_fn,
        }
        episode_log["topology_correct"] = episode_topology_correct
        episode_log["topology_total"] = episode_topology_total
        per_episode.append(episode_log)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    def summarize(values):
        tensor = torch.tensor(values, dtype=torch.float64)
        return {
            "mean": float(tensor.mean()),
            "std": float(tensor.std(unbiased=False)),
            "n": len(values),
        }

    return {
        "n_episodes": n_episodes,
        "horizon": horizon,
        "shuffle_material": shuffle_material,
        "shuffle_seed": seed if shuffle_material else None,
        "shuffle_offset": shuffle_offset,
        "material_feature_assignment": [
            {
                "episode_index": index,
                "episode_id": episodes[index].episode_id,
                "feature_episode_index": feature_order[index],
                "feature_episode_id": episodes[
                    feature_order[index]
                ].episode_id,
            }
            for index in range(n_episodes)
        ],
        "metrics_at_horizon": {
            key: {str(step): summarize(values)
                  for step, values in by_step.items()}
            for key, by_step in per_step.items()
        },
        "self_contact": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        },
        "topology_accuracy": topology_correct / max(topology_total, 1),
        "rollout_divergence_rate": divergent / n_episodes,
        "episodes": per_episode,
    }


@torch.no_grad()
def evaluate_counterfactual_group(
    model,
    episodes,
    *,
    device="cpu",
    normalizer=None,
    contact_margin_scale=0.5,
    shuffle_material=False,
    seed=0,
):
    """比较一组 paired episode 的 simulator/model 材料效应。"""
    if len(episodes) < 2:
        raise ValueError("反事实评估至少需要两个 episode")
    episodes = [episode.to(device).validate() for episode in episodes]
    group_ids = {
        episode.metadata.get("counterfactual_group_id")
        for episode in episodes
    }
    if len(group_ids) != 1 or None in group_ids:
        raise ValueError("episodes 必须来自同一 counterfactual_group_id")
    parameter_values = {
        episode.metadata.get("counterfactual_parameter")
        for episode in episodes
    }
    if len(parameter_values) != 1 or None in parameter_values:
        raise ValueError("反事实组缺少唯一 counterfactual_parameter")
    parameter = next(iter(parameter_values))
    if parameter not in {
        "K", "E", "G", "linear_density", "radius",
        "mu_self_static", "mu_self_kinetic",
    }:
        raise ValueError(f"未知反事实参数: {parameter!r}")

    reference = episodes[0]
    reference_control_seed = reference.metadata.get("control_seed")
    if reference_control_seed is None:
        raise ValueError("反事实 episode 缺少 control_seed")
    material_values = lambda episode: {
        "K": episode.material.K,
        "E": episode.material.E,
        "G": episode.material.G,
        "linear_density": episode.material.linear_density(),
        "radius": episode.material.node_radius.mean(),
        "mu_self_static": episode.material.mu_self_static,
        "mu_self_kinetic": episode.material.mu_self_kinetic,
    }
    reference_values = material_values(reference)
    for episode in episodes:
        _validate_model_dt(model, episode.macro_dt)
        if episode.horizon != reference.horizon:
            raise ValueError("反事实组 horizon 不一致")
        if episode.task != reference.task:
            raise ValueError("反事实组 task 不一致")
        if episode.metadata.get("control_seed") != reference_control_seed:
            raise ValueError("反事实组 control_seed 不一致")
        for field in ("pos", "vel", "tension", "contact", "topology"):
            if not torch.allclose(
                getattr(episode.states[0], field),
                getattr(reference.states[0], field),
                rtol=1e-6,
                atol=1e-7,
            ):
                raise ValueError(f"反事实组初始 {field} 不一致")
        if not torch.equal(
            episode.material.rest_length,
            reference.material.rest_length,
        ):
            raise ValueError("反事实组 rest_length 不一致")
        values = material_values(episode)
        for name, reference_value in reference_values.items():
            if name == parameter:
                continue
            if not torch.allclose(
                values[name], reference_value, rtol=1e-6, atol=1e-8
            ):
                raise ValueError(
                    f"反事实组除 {parameter} 外还改变了 {name}"
                )
        for step, (action, reference_action) in enumerate(
            zip(episode.actions, reference.actions)
        ):
            if not torch.equal(action.grasp_idx, reference_action.grasp_idx):
                raise ValueError(f"反事实组 action[{step}] grasp_idx 不一致")
            if action.target_pos is None or reference_action.target_pos is None:
                raise ValueError("反事实评估要求 action.target_pos")
            if not torch.allclose(
                action.target_pos, reference_action.target_pos,
                rtol=1e-6, atol=1e-7,
            ):
                raise ValueError(f"反事实组 action[{step}] target_pos 不一致")
            for field in ("grasp_active", "duration"):
                value = getattr(action, field)
                reference_value = getattr(reference_action, field)
                if (value is None) != (reference_value is None):
                    raise ValueError(
                        f"反事实组 action[{step}] {field} 缺失情况不一致"
                    )
                if value is not None and not torch.allclose(
                    value.to(torch.float32),
                    reference_value.to(torch.float32),
                    rtol=1e-6,
                    atol=1e-7,
                ):
                    raise ValueError(
                        f"反事实组 action[{step}] {field} 不一致"
                    )

    scales = [
        float(episode.metadata.get("counterfactual_scale", float("nan")))
        for episode in episodes
    ]
    if not all(math.isfinite(scale) for scale in scales):
        raise ValueError("反事实组缺少有效 counterfactual_scale")
    if len(set(scales)) != len(scales):
        raise ValueError("反事实组 counterfactual_scale 重复")
    reference_index = min(
        range(len(episodes)), key=lambda index: abs(scales[index] - 1.0)
    )
    if not math.isclose(scales[reference_index], 1.0, abs_tol=1e-6):
        raise ValueError("反事实组必须包含 scale=1.0 reference")
    if reference_index != 0:
        episodes[0], episodes[reference_index] = (
            episodes[reference_index], episodes[0]
        )
    baseline_value = material_values(episodes[0])[parameter]
    for episode in episodes:
        declared_scale = float(
            episode.metadata["counterfactual_scale"]
        )
        actual_scale = material_values(episode)[parameter] / baseline_value
        if not torch.isclose(
            actual_scale,
            torch.tensor(
                declared_scale,
                device=actual_scale.device,
                dtype=actual_scale.dtype,
            ),
            rtol=1e-5,
            atol=1e-7,
        ):
            raise ValueError(
                f"反事实 metadata scale={declared_scale} 与实际 "
                f"{parameter} 倍率={float(actual_scale)} 不一致"
            )

    feature_order = list(range(len(episodes)))
    shuffle_offset = 0
    if shuffle_material:
        offset = random.Random(seed).randrange(1, len(episodes))
        shuffle_offset = offset
        feature_order = [
            (index + offset) % len(episodes)
            for index in range(len(episodes))
        ]
    predicted = []
    for index, episode in enumerate(episodes):
        feature_episode = episodes[feature_order[index]]
        features = _episode_material_features(feature_episode, normalizer)
        builder = edge_builder_from_material(
            episode.material, contact_margin_scale
        )
        predicted.append(model.rollout(
            episode.states[0], episode.actions, builder, features
        )[-1].pos)

    reference_gt = episodes[0].states[-1].pos
    reference_pred = predicted[0]
    comparisons = []
    for index in range(1, len(episodes)):
        gt_effect = episodes[index].states[-1].pos - reference_gt
        pred_effect = predicted[index] - reference_pred
        effect_rmse = torch.sqrt(((pred_effect - gt_effect) ** 2).mean())
        gt_flat = gt_effect.flatten()
        pred_flat = pred_effect.flatten()
        error_flat = pred_flat - gt_flat
        cosine = torch.nn.functional.cosine_similarity(
            gt_flat.unsqueeze(0), pred_flat.unsqueeze(0), dim=1
        )[0]
        comparisons.append({
            "reference_episode_id": episodes[0].episode_id,
            "reference_scale": float(
                episodes[0].metadata["counterfactual_scale"]
            ),
            "episode_id": episodes[index].episode_id,
            "scale": float(
                episodes[index].metadata["counterfactual_scale"]
            ),
            "effect_rmse": float(effect_rmse),
            "effect_nrmse": float(
                effect_rmse
                / episodes[index].material.rest_length.sum().clamp(min=1e-8)
            ),
            "relative_effect_error": float(
                error_flat.norm() / gt_flat.norm().clamp(min=1e-8)
            ),
            "effect_cosine": float(cosine),
            "gt_effect_norm": float(gt_flat.norm()),
            "pred_effect_norm": float(pred_flat.norm()),
        })
    count = len(episodes)
    gt_pairwise = torch.zeros(count, count, device=device)
    pred_pairwise = torch.zeros(count, count, device=device)
    unique_pair_errors = []
    for i in range(count):
        for j in range(i + 1, count):
            gt_distance = torch.sqrt(
                ((episodes[i].states[-1].pos
                  - episodes[j].states[-1].pos) ** 2).mean()
            )
            pred_distance = torch.sqrt(
                ((predicted[i] - predicted[j]) ** 2).mean()
            )
            gt_pairwise[i, j] = gt_pairwise[j, i] = gt_distance
            pred_pairwise[i, j] = pred_pairwise[j, i] = pred_distance
            unique_pair_errors.append((pred_distance - gt_distance) ** 2)
    matrix_rmse = torch.sqrt(torch.stack(unique_pair_errors).mean())
    return {
        "counterfactual_group_id": next(iter(group_ids)),
        "counterfactual_parameter": parameter,
        "shuffle_material": shuffle_material,
        "shuffle_seed": seed if shuffle_material else None,
        "shuffle_offset": shuffle_offset,
        "material_feature_assignment": [
            {
                "episode_index": index,
                "episode_id": episodes[index].episode_id,
                "episode_scale": float(
                    episodes[index].metadata["counterfactual_scale"]
                ),
                "feature_episode_index": feature_order[index],
                "feature_episode_id": episodes[
                    feature_order[index]
                ].episode_id,
                "feature_scale": float(
                    episodes[feature_order[index]].metadata[
                        "counterfactual_scale"
                    ]
                ),
            }
            for index in range(len(episodes))
        ],
        "scales": [
            float(episode.metadata["counterfactual_scale"])
            for episode in episodes
        ],
        "comparisons": comparisons,
        "gt_pairwise_distance": gt_pairwise.cpu().tolist(),
        "pred_pairwise_distance": pred_pairwise.cpu().tolist(),
        "pairwise_matrix_rmse": float(matrix_rmse),
        "pairwise_matrix_nrmse": float(
            matrix_rmse
            / episodes[0].material.rest_length.sum().clamp(min=1e-8)
        ),
    }
