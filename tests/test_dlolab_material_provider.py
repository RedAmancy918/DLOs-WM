import torch
import pytest

from dlo_wm.data.dlolab_provider import DLOLabProvider
from dlo_wm.data.material_sampling import (
    MaterialParameters,
    MaterialRandomizationConfig,
)
from dlo_wm.train.trainer import edge_builder_from_material


class _FakeRope:
    """只记录 setter 输入，验证 provider 确实改的是 solver 而非 metadata。"""

    def __init__(self):
        self.calls = {}

    def __getattr__(self, name):
        if not name.startswith("set_"):
            raise AttributeError(name)

        def record(value, *args, **kwargs):
            self.calls[name] = value.clone()

        return record


def _provider():
    provider = DLOLabProvider(
        num_nodes=4,
        interval=0.1,
        segment_mass=0.02,
        segment_radius=0.01,
        steps_interval=20,
    )
    provider._rest_len = torch.tensor([0.11, 0.12, 0.13])
    provider._init_pos = torch.zeros(4, 3)
    provider._rope = _FakeRope()
    return provider


def test_inextensible_provider_rejects_unidentifiable_k_randomization():
    with pytest.raises(ValueError, match="K"):
        DLOLabProvider(
            num_nodes=4,
            use_inextensible=True,
            material_randomization=MaterialRandomizationConfig(
                K_scale=(0.7, 1.4)
            ),
        )


def test_apply_material_updates_every_solver_parameter_and_schema():
    provider = _provider()
    material = MaterialParameters(
        K=12.0,
        E=23.0,
        G=34.0,
        linear_density=0.5,
        radius=0.007,
        mu_self_static=0.4,
        mu_self_kinetic=0.3,
    )

    condition = provider._apply_material(material)

    assert set(provider._rope.calls) == {
        "set_stretching_stiffness",
        "set_bending_stiffness",
        "set_twisting_stiffness",
        "set_segment_mass",
        "set_segment_radius",
        "set_mu_s",
        "set_mu_k",
    }
    assert provider._rope.calls["set_segment_mass"].shape == (1, 4)
    assert provider._rope.calls["set_segment_radius"].shape == (1, 4)
    assert torch.allclose(condition.rest_length, provider._rest_len)
    assert torch.isclose(
        condition.node_mass.sum(),
        torch.tensor(material.linear_density) * provider._rest_len.sum(),
    )
    assert torch.all(condition.node_radius == material.radius)
    assert float(condition.K) == material.K
    assert float(condition.E) == material.E
    assert float(condition.G) == material.G
    assert torch.isclose(
        condition.mu_self_static,
        torch.tensor(material.mu_self_static),
    )
    assert torch.isclose(
        condition.mu_self_kinetic,
        torch.tensor(material.mu_self_kinetic),
    )
    assert provider.stretch_stiffness == material.K
    assert provider.contact_radius == 3.0 * material.radius


def test_action_records_target_velocity_consistently():
    provider = _provider()
    current_pos = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
         [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
    )
    delta = torch.tensor([[0.02, -0.01, 0.0]])
    action = provider._make_action(
        torch.tensor([2]), delta, current_pos
    )

    assert torch.allclose(action.target_pos, current_pos[[2]] + delta)
    assert torch.allclose(
        action.target_vel,
        (action.target_pos - current_pos[[2]]) / action.duration,
    )


def test_planned_targets_are_reproducible_for_counterfactual_pair():
    provider = _provider()
    provider.motion = "loop"
    reference = torch.zeros(4, 3)
    reference[:, 0] = torch.arange(4) * 0.1

    first = provider._plan_motion(
        reference, 6, torch.Generator().manual_seed(17)
    )
    second = provider._plan_motion(
        reference, 6, torch.Generator().manual_seed(17)
    )

    assert first[0] == second[0]
    assert torch.equal(first[1], second[1])


def test_provider_labels_and_rollout_builder_use_identical_contact_rule():
    provider = _provider()
    material = MaterialParameters(
        K=12.0,
        E=23.0,
        G=34.0,
        linear_density=0.5,
        radius=0.007,
        mu_self_static=0.4,
        mu_self_kinetic=0.3,
    )
    condition = provider._apply_material(material)
    pos = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.11, 0.0, 0.0],
        [0.23, 0.0, 0.0],
        [0.0, 0.01, 0.0],
    ])

    labeled_pairs, labeled_nodes = provider._self_contacts(pos)
    edge_index, is_contact = edge_builder_from_material(
        condition, contact_margin_scale=0.5
    )(pos)
    directed = edge_index[:, is_contact.bool()].t()
    rollout_pairs = torch.sort(directed, dim=1).values.unique(dim=0)

    assert torch.equal(labeled_pairs, rollout_pairs)
    assert torch.equal(
        labeled_nodes, torch.tensor([1.0, 0.0, 0.0, 1.0])
    )
