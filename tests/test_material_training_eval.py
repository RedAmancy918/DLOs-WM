import pytest
import torch
from dataclasses import replace

from dlo_wm.data.dataset import TrajectoryProvider, episode_to_transitions
from dlo_wm.data.normalization import MaterialFeatureNormalizer
from dlo_wm.data.schema import (
    DLOAction,
    DLOEpisode,
    DLOState,
    MaterialCondition,
)
from dlo_wm.eval.material_rollout import (
    evaluate_counterfactual_group,
    evaluate_material_rollout,
)
from dlo_wm.model.gnn import MaterialConditionedDLOWorldModel
from dlo_wm.train.material_trainer import (
    closed_loop_loss,
    train_material_conditioned,
    transition_loss,
)


def _state(pos, velocity=0.0):
    n = pos.shape[0]
    return DLOState(
        pos=pos.clone(),
        vel=torch.full_like(pos, velocity),
        tension=torch.full((n,), 0.1),
        contact=torch.zeros(n),
        topology=torch.tensor(0, dtype=torch.long),
    )


def _episode(material_scale=1.0, episode_id="ep"):
    n = 4
    base = torch.zeros(n, 3)
    base[:, 0] = torch.arange(n) * 0.1
    states = [
        _state(base),
        _state(base + torch.tensor([0.001, 0.0, 0.0]), 0.01),
        _state(base + torch.tensor([0.003, 0.0, 0.0]), 0.02),
    ]
    actions = []
    for step in range(2):
        target = states[step].pos[[3]] + torch.tensor([[0.002, 0.0, 0.0]])
        actions.append(DLOAction(
            grasp_idx=torch.tensor([3]),
            delta=target - states[step].pos[[3]],
            target_pos=target,
        ))
    material = MaterialCondition(
        rest_length=torch.full((n - 1,), 0.1),
        node_mass=torch.full((n,), 0.02 * material_scale),
        node_radius=torch.full((n,), 0.005),
        K=torch.tensor(10.0 * material_scale),
        E=torch.tensor(20.0 * material_scale),
        G=torch.tensor(30.0),
        mu_self_static=torch.tensor(0.3),
        mu_self_kinetic=torch.tensor(0.2),
    )
    return DLOEpisode(
        material=material,
        states=states,
        actions=actions,
        contact_pairs=[torch.empty(0, 2, dtype=torch.long) for _ in states],
        macro_dt=0.1,
        task="unit",
        id=episode_id,
    ).validate()


class _Provider(TrajectoryProvider):
    def __init__(self, episodes):
        self.episodes = episodes
        self.index = 0

    @property
    def num_nodes(self):
        return self.episodes[0].num_nodes

    def sample_episode(self, T=None, **kwargs):
        episode = self.episodes[self.index % len(self.episodes)]
        self.index += 1
        return episode

    def sample_trajectory(self, T=None):
        episode = self.sample_episode(T=T)
        return episode.states, episode.actions, episode.contact_pairs


def _config():
    return {
        "weights": {
            "pos": 1.0,
            "tension": 0.1,
            "contact": 0.1,
            "topo": 0.1,
            "fail": 0.1,
        },
        "tension_limit": 1.0,
        "stuck_topo_classes": [2],
        "rollout_horizon": 2,
        "contact_margin_scale": 0.0,
    }


def test_teacher_forced_and_closed_loop_losses_backpropagate():
    episode = _episode()
    model = MaterialConditionedDLOWorldModel(
        hidden=12, n_message_passing=1, dt=episode.macro_dt
    )
    normalizer = MaterialFeatureNormalizer.fit([episode.material])

    one_step, _ = transition_loss(
        model,
        episode_to_transitions(episode)[0],
        _config(),
        normalizer=normalizer,
    )
    one_step.backward()
    assert torch.isfinite(one_step)

    model.zero_grad(set_to_none=True)
    rollout, _ = closed_loop_loss(
        model, episode, _config(), normalizer=normalizer
    )
    rollout.backward()
    assert torch.isfinite(rollout)
    assert any(
        parameter.grad is not None
        for parameter in model.parameters()
    )

    wrong_dt_model = MaterialConditionedDLOWorldModel(
        hidden=12, n_message_passing=1, dt=0.04
    )
    with pytest.raises(ValueError, match="macro_dt"):
        transition_loss(
            wrong_dt_model,
            episode_to_transitions(episode)[0],
            _config(),
            normalizer=normalizer,
        )


def test_material_rollout_report_and_shuffle_path():
    episodes = [_episode(0.8, "low"), _episode(1.2, "high")]
    provider = _Provider(episodes)
    model = MaterialConditionedDLOWorldModel(
        hidden=12, n_message_passing=1,
        dt=episodes[0].macro_dt,
    )
    normalizer = MaterialFeatureNormalizer.fit(
        [episode.material for episode in episodes]
    )

    report = evaluate_material_rollout(
        model,
        provider,
        n_episodes=2,
        horizon=2,
        normalizer=normalizer,
        shuffle_material=True,
    )

    assert report["shuffle_material"] is True
    assert set(report["metrics_at_horizon"]["position_rmse"]) == {"1", "2"}
    assert report["self_contact"]["tp"] == 0
    assert len(report["episodes"]) == 2
    assert report["shuffle_seed"] == 0
    assert all(
        row["episode_id"] != row["feature_episode_id"]
        for row in report["material_feature_assignment"]
    )

    with pytest.raises(ValueError, match="episode 长度不足"):
        evaluate_material_rollout(
            model,
            _Provider(episodes),
            n_episodes=2,
            horizon=3,
            normalizer=normalizer,
        )


def test_counterfactual_requires_pairing_and_uses_scale_one_reference():
    episodes = []
    for index, scale in enumerate((0.7, 1.0, 1.4)):
        episode = _episode(1.0, f"cf-{index}")
        episode.material = replace(
            episode.material,
            K=torch.tensor(10.0 * scale),
        )
        episode.metadata.update({
            "counterfactual_group_id": "group-1",
            "counterfactual_parameter": "K",
            "counterfactual_scale": scale,
            "control_seed": 123,
        })
        episodes.append(episode.validate())

    model = MaterialConditionedDLOWorldModel(
        hidden=12, n_message_passing=1, dt=0.1
    )
    normalizer = MaterialFeatureNormalizer.fit(
        [episode.material for episode in episodes]
    )
    report = evaluate_counterfactual_group(
        model, episodes, normalizer=normalizer
    )
    assert all(
        row["reference_scale"] == 1.0
        for row in report["comparisons"]
    )
    shuffled = evaluate_counterfactual_group(
        model,
        episodes,
        normalizer=normalizer,
        shuffle_material=True,
        seed=7,
    )
    assert shuffled["shuffle_material"] is True
    assert shuffled["shuffle_seed"] == 7
    assert all(
        row["episode_id"] != row["feature_episode_id"]
        and row["episode_scale"] != row["feature_scale"]
        for row in shuffled["material_feature_assignment"]
    )

    episodes[2].metadata["counterfactual_scale"] = 1.5
    with pytest.raises(ValueError, match="实际 K 倍率"):
        evaluate_counterfactual_group(
            model, episodes, normalizer=normalizer
        )
    episodes[2].metadata["counterfactual_scale"] = 1.4

    episodes[0].metadata["control_seed"] = 999
    with pytest.raises(ValueError, match="control_seed"):
        evaluate_counterfactual_group(
            model, episodes, normalizer=normalizer
        )


def test_training_early_stops_and_restores_best_validation_epoch():
    episode = _episode()
    provider = _Provider([episode])
    model = MaterialConditionedDLOWorldModel(
        hidden=8, n_message_passing=1, dt=episode.macro_dt
    )
    normalizer = MaterialFeatureNormalizer.fit([episode.material])
    cfg = {
        **_config(),
        "epochs": 6,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "grad_clip": 1.0,
        "traj_per_epoch": 1,
        "traj_len": 1,
        "rollout_updates_per_epoch": 0,
        "rollout_weight": 0.0,
    }
    values = iter((3.0, 2.0, 2.5, 2.6, 1.0))
    validation_states = []

    def validation_fn(candidate):
        validation_states.append({
            name: value.detach().clone()
            for name, value in candidate.state_dict().items()
        })
        return next(values)

    history = train_material_conditioned(
        model,
        provider,
        cfg,
        normalizer=normalizer,
        validation_fn=validation_fn,
        early_stop_patience=2,
    )

    assert len(history) == 4
    assert history[-1]["best_epoch"] == 1.0
    assert history[-1]["best_val_position_nrmse"] == 2.0
    assert all(
        torch.equal(value, validation_states[1][name])
        for name, value in model.state_dict().items()
    )
