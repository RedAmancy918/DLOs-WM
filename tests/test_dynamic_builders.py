import pytest
import torch

from dlo_wm.data.schema import MaterialCondition, node_contact_from_edges
from dlo_wm.train.trainer import (
    edge_builder_from_contacts,
    edge_builder_from_material,
)


def _material(device="cpu"):
    return MaterialCondition(
        rest_length=torch.full((4,), 0.1, device=device),
        node_mass=torch.full((5,), 0.02, device=device),
        node_radius=torch.full((5,), 0.01, device=device),
        K=torch.tensor(10.0, device=device),
        E=torch.tensor(20.0, device=device),
        G=torch.tensor(30.0, device=device),
        mu_self_static=torch.tensor(0.3, device=device),
        mu_self_kinetic=torch.tensor(0.2, device=device),
    )


def test_material_edge_builder_uses_current_position_and_radius():
    material = _material()
    builder = edge_builder_from_material(
        material, contact_margin_scale=0.0
    )
    far = torch.tensor(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0],
         [0.3, 0.0, 0.0], [0.4, 0.0, 0.0]]
    )
    near = far.clone()
    near[4] = torch.tensor([0.0, 0.015, 0.0])

    _, far_contact = builder(far)
    edges, near_contact = builder(near)

    assert int(far_contact.sum()) == 0
    contact_edges = edges[:, near_contact.bool()].t().tolist()
    assert [0, 4] in contact_edges
    assert [4, 0] in contact_edges
    node_contact = node_contact_from_edges(5, edges, near_contact)
    assert torch.equal(
        node_contact, torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0])
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA")
def test_dynamic_edge_builders_keep_indices_on_cuda():
    pos = torch.zeros(5, 3, device="cuda")
    pos[:, 0] = torch.arange(5, device="cuda") * 0.1

    edge_a, flag_a = edge_builder_from_contacts(5, 0.03)(pos)
    edge_b, flag_b = edge_builder_from_material(
        _material("cuda"), contact_margin_scale=0.0
    )(pos)

    assert edge_a.device.type == flag_a.device.type == "cuda"
    assert edge_b.device.type == flag_b.device.type == "cuda"
