import dataclasses

import pytest
import torch

from dlo_wm.data.schema import (
    DLOAction,
    DLOEpisode,
    DLOState,
    MaterialCondition,
)
from dlo_wm.data.serialization import (
    FORMAT_VERSION,
    episode_from_dict,
    episode_to_dict,
    load_episode,
    save_episode,
)


def _material(num_nodes=4):
    return MaterialCondition(
        rest_length=torch.full((num_nodes - 1,), 0.5),
        node_mass=torch.full((num_nodes,), 0.1),
        node_radius=torch.full((num_nodes,), 0.01),
        K=torch.tensor(100.0),
        E=torch.tensor(20.0),
        G=torch.tensor(5.0),
        mu_self_static=torch.tensor(0.4),
        mu_self_kinetic=torch.tensor(0.3),
    )


def _state(num_nodes=4, offset=0.0):
    return DLOState(
        pos=torch.arange(num_nodes * 3, dtype=torch.float32).reshape(num_nodes, 3)
        + offset,
        vel=torch.zeros(num_nodes, 3),
        tension=torch.linspace(0.0, 1.0, num_nodes),
        contact=torch.zeros(num_nodes),
        topology=torch.tensor(0, dtype=torch.long),
    )


def _episode():
    action = DLOAction(
        grasp_idx=torch.tensor([1, -1]),
        delta=torch.tensor([[0.1, 0.0, 0.0], [9.0, 9.0, 9.0]]),
        gripper_id=torch.tensor([0, 1]),
        target_pos=torch.tensor([[0.1, 0.2, 0.3], [0.0, 0.0, 0.0]]),
        target_vel=torch.zeros(2, 3),
        grasp_active=torch.tensor([True, False]),
        duration=0.05,
    )
    empty_pairs = torch.empty((0, 2), dtype=torch.long)
    return DLOEpisode(
        material=_material(),
        states=[_state(), _state(offset=0.1)],
        actions=[action],
        contact_pairs=[empty_pairs, torch.tensor([[0, 3]], dtype=torch.long)],
        macro_dt=0.05,
        task="loop",
        seed=17,
        id="episode-000017",
        metadata={"counterfactual_group_id": "group-3", "scales": [0.8, 1.2]},
    )


def _assert_primitive_tree(value):
    assert not dataclasses.is_dataclass(value)
    if isinstance(value, dict):
        for key, item in value.items():
            assert isinstance(key, str)
            _assert_primitive_tree(item)
    elif isinstance(value, list):
        for item in value:
            _assert_primitive_tree(item)
    else:
        assert value is None or isinstance(
            value, (bool, int, float, str, torch.Tensor)
        )


def test_material_condition_shapes_density_and_global_features():
    material = _material().validate(num_nodes=4)

    assert torch.allclose(material.linear_density(), torch.tensor(0.4 / 1.5))
    expected = torch.tensor([
        100.0,
        20.0,
        5.0,
    ]).log()
    expected = torch.cat([
        expected,
        torch.tensor([0.01, 0.4 / 1.5, 0.4, 0.3]),
    ])
    assert material.global_features().shape == (7,)
    assert torch.allclose(material.global_features(), expected)


def test_material_condition_rejects_wrong_native_shapes():
    material = _material()
    material.rest_length = torch.ones(4)

    with pytest.raises(ValueError, match="rest_length"):
        material.validate()


def test_old_delta_action_stays_compatible():
    action = DLOAction(
        torch.tensor([0, 3]),
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),
    )

    drive = action.to_node_drive(4)

    assert torch.equal(drive[0], action.delta[0])
    assert torch.equal(drive[3], action.delta[1])
    assert torch.count_nonzero(drive[1:3]) == 0


def test_inactive_negative_grasp_does_not_write_last_node():
    action = DLOAction(
        grasp_idx=torch.tensor([1, -1]),
        delta=torch.tensor([[1.0, 0.0, 0.0], [8.0, 8.0, 8.0]]),
        grasp_active=torch.tensor([True, False]),
    )

    drive = action.to_node_drive(4)

    assert torch.equal(drive[1], action.delta[0])
    assert torch.equal(drive[-1], torch.zeros(3))


def test_target_position_drive_is_recomputed_from_predicted_position():
    action = DLOAction(
        grasp_idx=torch.tensor([1]),
        delta=torch.tensor([[99.0, 99.0, 99.0]]),
        target_pos=torch.tensor([[2.0, 1.0, 0.0]]),
    )
    predicted_pos = torch.zeros(4, 3)
    predicted_pos[1] = torch.tensor([0.5, 0.25, 0.0])

    drive = action.to_node_drive(4, current_pos=predicted_pos)

    assert torch.equal(drive[1], torch.tensor([1.5, 0.75, 0.0]))
    predicted_pos[1, 0] = 1.0
    next_drive = action.to_node_drive(4, current_pos=predicted_pos)
    assert torch.equal(next_drive[1], torch.tensor([1.0, 0.75, 0.0]))


def test_episode_validates_temporal_contract_without_sdf_state_fields():
    episode = _episode().validate()

    assert episode.horizon == 1
    assert episode.episode_id == episode.id
    assert not hasattr(episode.states[0], "obstacle_sdf")

    episode.actions = []
    with pytest.raises(ValueError, match=r"T\+1/T"):
        episode.validate()


def test_episode_primitive_dict_and_memory_round_trip():
    original = _episode().validate()
    data = episode_to_dict(original)

    assert data["format_version"] == FORMAT_VERSION
    _assert_primitive_tree(data)
    restored = episode_from_dict(data)

    assert restored.id == original.id
    assert restored.task == original.task
    assert restored.metadata == original.metadata
    assert torch.equal(restored.material.rest_length, original.material.rest_length)
    assert torch.equal(restored.states[1].pos, original.states[1].pos)
    assert torch.equal(restored.actions[0].grasp_active,
                       original.actions[0].grasp_active)
    assert torch.equal(restored.contact_pairs[1], original.contact_pairs[1])


def test_episode_disk_round_trip(tmp_path):
    path = tmp_path / "episode-v2.pt"
    original = _episode().validate()

    save_episode(original, path)
    restored = load_episode(path)

    assert restored.id == original.id
    assert torch.equal(restored.material.global_features(),
                       original.material.global_features())
    assert torch.equal(restored.actions[0].duration,
                       original.actions[0].duration)


def test_unknown_format_version_is_explicitly_rejected():
    data = episode_to_dict(_episode())
    data["format_version"] = FORMAT_VERSION + 1

    with pytest.raises(ValueError, match="不支持.*格式版本"):
        episode_from_dict(data)
