"""材料条件化模型训练。

保留旧 ``trainer.train`` 不变；本模块提供 workshop 实验所需的显式材料输入和
真正的闭环多步损失。多步阶段把模型自己的预测状态继续送回模型，而不是仅对
teacher-forced 输入加噪声。
"""

from __future__ import annotations

import random
import math

import torch

from ..data.dataset import make_transition_batch, slice_episode
from ..data.schema import (
    DLOState,
    build_edges,
    node_contact_from_edges,
)
from .losses import world_model_loss
from .trainer import edge_builder_from_material


def _validate_macro_dt(model, macro_dt):
    """拒绝用错误积分步长静默训练。"""
    if macro_dt is None:
        return
    model_dt = getattr(model, "dt", None)
    for attribute in ("model", "base_model"):
        if model_dt is None and hasattr(model, attribute):
            model_dt = getattr(getattr(model, attribute), "dt", None)
    if model_dt is None:
        raise AttributeError("模型未暴露 dt，无法核对 episode.macro_dt")
    if not math.isclose(
        float(model_dt), float(macro_dt), rel_tol=1e-6, abs_tol=1e-9
    ):
        raise ValueError(
            f"模型 dt={float(model_dt):.9g} 与数据 macro_dt="
            f"{float(macro_dt):.9g} 不一致"
        )


def _mean_logs(logs: list[dict[str, float]]) -> dict[str, float]:
    if not logs:
        return {}
    return {
        key: sum(log[key] for log in logs) / len(logs)
        for key in logs[0]
    }


def _sample_cached_episodes(episodes, count, rng):
    """独立 RNG 的确定性 episode 日程；不足一轮时无放回抽取。"""
    if count <= 0:
        raise ValueError("episode 采样数必须大于 0")
    pool = list(episodes)
    if not pool:
        raise ValueError("episode pool 为空")
    selected = []
    while len(selected) < count:
        order = list(range(len(pool)))
        rng.shuffle(order)
        selected.extend(pool[index] for index in order)
    return selected[:count]


def _predicted_state(pred: dict[str, torch.Tensor]) -> DLOState:
    """把连续预测变成下一轮输入；contact 保留概率以维持梯度。"""
    return DLOState(
        pos=pred["pos_next"],
        vel=pred["vel_next"],
        tension=pred["tension"],
        contact=torch.sigmoid(pred["contact_logit"]),
        # topology 不属于节点输入，离散化不会截断动力学主干的梯度。
        topology=pred["topo_logits"].argmax().long(),
    )


def transition_loss(
    model, sample, cfg, device="cpu", normalizer=None
):
    """一个材料条件化 teacher-forced transition 的损失。"""
    state = sample["state_t"].to(device)
    target = sample["state_tp1"].to(device)
    action = sample["action_t"].to(device)
    material = sample.get("material")
    if material is None:
        raise ValueError(
            "材料条件化训练需要 v2 episode；当前 transition 缺少 material"
        )
    material = material.to(device)
    _validate_macro_dt(model, sample.get("macro_dt"))
    pairs = sample.get("cpairs_t")
    if pairs is not None and len(pairs):
        pairs = pairs.to(device)
    else:
        pairs = None
    edge_index, is_contact = build_edges(
        state.num_nodes, pairs, device=device
    )
    model_state = DLOState(
        pos=state.pos,
        vel=state.vel,
        tension=state.tension,
        contact=node_contact_from_edges(
            state.num_nodes,
            edge_index,
            is_contact,
            dtype=state.pos.dtype,
        ),
        topology=state.topology,
    )
    features = material.global_features()
    if normalizer is not None:
        features = normalizer.transform(features)
    pred = model(
        model_state,
        action.to_node_drive(state.num_nodes, current_pos=state.pos),
        edge_index,
        is_contact,
        features,
    )
    return world_model_loss(
        pred,
        target,
        cfg["weights"],
        cfg["tension_limit"],
        cfg["stuck_topo_classes"],
    )


def closed_loop_loss(
    model, episode, cfg, device="cpu", normalizer=None
):
    """在一个 episode 上计算可反传的闭环多步平均损失。"""
    episode = episode.to(device).validate()
    _validate_macro_dt(model, episode.macro_dt)
    horizon = min(
        int(cfg.get("rollout_horizon", episode.horizon)),
        episode.horizon,
    )
    if horizon <= 0:
        raise ValueError("rollout_horizon 必须大于 0")

    state = episode.states[0]
    features = episode.material.global_features()
    if normalizer is not None:
        features = normalizer.transform(features)
    builder = edge_builder_from_material(
        episode.material,
        cfg.get("contact_margin_scale", 0.5),
    )
    losses = []
    logs = []
    for step in range(horizon):
        action = episode.actions[step]
        edge_index, is_contact = builder(state.pos)
        model_state = DLOState(
            pos=state.pos,
            vel=state.vel,
            tension=state.tension,
            contact=node_contact_from_edges(
                state.num_nodes,
                edge_index,
                is_contact,
                dtype=state.pos.dtype,
            ),
            topology=state.topology,
        )
        pred = model(
            model_state,
            action.to_node_drive(state.num_nodes, current_pos=state.pos),
            edge_index,
            is_contact,
            features,
        )
        loss, log = world_model_loss(
            pred,
            episode.states[step + 1],
            cfg["weights"],
            cfg["tension_limit"],
            cfg["stuck_topo_classes"],
        )
        losses.append(loss)
        logs.append(log)
        state = _predicted_state(pred)
    return torch.stack(losses).mean(), _mean_logs(logs)


def train_material_conditioned(
    model,
    provider,
    cfg,
    device="cpu",
    seed=0,
    normalizer=None,
    validation_fn=None,
    early_stop_patience: int | None = None,
    early_stop_min_delta: float = 0.0,
):
    """单步主训练 + 可选真闭环 rollout 更新。

    ``rollout_updates_per_epoch=0`` 可得到严格单步 baseline；设置为正数后，
    每个 epoch 额外采样若干完整 episode 做 push-forward 更新。
    """
    if normalizer is None:
        raise ValueError(
            "材料训练必须显式传入仅由 train split 拟合的 normalizer"
        )
    if validation_fn is None and early_stop_patience is not None:
        raise ValueError("设置 early_stop_patience 时必须提供 validation_fn")
    if early_stop_patience is not None and early_stop_patience <= 0:
        raise ValueError("early_stop_patience 必须大于 0")
    if early_stop_min_delta < 0:
        raise ValueError("early_stop_min_delta 必须大于等于 0")
    model.to(device).train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg.get("weight_decay", 0.0),
    )
    teacher_rng = random.Random(seed)
    rollout_rng = random.Random(seed + 10_000_019)
    cached_episodes = None
    try:
        cached_episodes = tuple(provider.episodes)
    except (AttributeError, NotImplementedError):
        pass
    history = []
    best_validation = float("inf")
    best_epoch = None
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(cfg["epochs"]):
        if cached_episodes is None:
            samples = make_transition_batch(
                provider,
                n_traj=cfg["traj_per_epoch"],
                T=cfg["traj_len"],
            )
        else:
            teacher_episodes = _sample_cached_episodes(
                cached_episodes,
                cfg["traj_per_epoch"],
                teacher_rng,
            )
            samples = make_transition_batch(
                teacher_episodes, T=cfg["traj_len"]
            )
        teacher_rng.shuffle(samples)
        epoch_logs = []
        for sample in samples:
            loss, log = transition_loss(
                model, sample, cfg, device=device,
                normalizer=normalizer,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.get("grad_clip", 1.0)
            )
            optimizer.step()
            epoch_logs.append(log)

        rollout_logs = []
        rollout_weight = float(cfg.get("rollout_weight", 1.0))
        for _ in range(int(cfg.get("rollout_updates_per_epoch", 0))):
            rollout_horizon = int(
                cfg.get("rollout_horizon", cfg["traj_len"])
            )
            if cached_episodes is None:
                episode = provider.sample_episode(T=rollout_horizon)
            else:
                episode = slice_episode(
                    cached_episodes[
                        rollout_rng.randrange(len(cached_episodes))
                    ],
                    T=rollout_horizon,
                )
            loss, log = closed_loop_loss(
                model, episode, cfg, device=device,
                normalizer=normalizer,
            )
            optimizer.zero_grad(set_to_none=True)
            (rollout_weight * loss).backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.get("grad_clip", 1.0)
            )
            optimizer.step()
            rollout_logs.append(log)

        summary = _mean_logs(epoch_logs)
        if rollout_logs:
            summary.update({
                f"rollout_{key}": value
                for key, value in _mean_logs(rollout_logs).items()
            })
        should_stop = False
        if validation_fn is not None:
            model.eval()
            validation_value = float(validation_fn(model))
            model.train()
            if not math.isfinite(validation_value):
                raise RuntimeError(
                    "validation position_NRMSE 不是有限数，拒绝选模"
                )
            summary["val_position_nrmse"] = validation_value
            improved = validation_value < (
                best_validation - early_stop_min_delta
            )
            if improved:
                best_validation = validation_value
                best_epoch = epoch
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            summary["best_val_position_nrmse"] = best_validation
            summary["best_epoch"] = float(best_epoch)
            summary["epochs_without_improvement"] = float(
                epochs_without_improvement
            )
            should_stop = (
                early_stop_patience is not None
                and epochs_without_improvement >= early_stop_patience
            )
        history.append(summary)
        msg = " ".join(
            f"{key}={value:.4f}" for key, value in summary.items()
        )
        print(f"[epoch {epoch:03d}] {msg}")
        if should_stop:
            print(
                f"[early-stop] epoch={epoch} best_epoch={best_epoch} "
                f"best_val_position_nrmse={best_validation:.6g}"
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
        print(
            f"[restore-best] epoch={best_epoch} "
            f"val_position_nrmse={best_validation:.6g}"
        )
    return history
