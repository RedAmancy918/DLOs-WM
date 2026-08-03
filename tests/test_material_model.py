import torch

from dlo_wm.data.batch import BatchedGraph
from dlo_wm.data.schema import DLOAction, DLOState, build_edges
from dlo_wm.model.gnn import (
    MATERIAL_INPUT_DIM,
    DLOWorldModel,
    MaterialConditionedDLOWorldModel,
)
from dlo_wm.model.gnn_batched import (
    BatchedMaterialConditionedDLOWorldModel,
)


def _state(num_nodes, offset=0.0):
    pos = torch.zeros(num_nodes, 3)
    pos[:, 0] = torch.linspace(0.0, 0.3, num_nodes) + offset
    vel = torch.randn(num_nodes, 3) * 0.01
    tension = torch.linspace(0.1, 0.2, num_nodes)
    contact = torch.zeros(num_nodes)
    return DLOState(
        pos=pos,
        vel=vel,
        tension=tension,
        contact=contact,
        topology=torch.tensor(0, dtype=torch.long),
    )


def _batched_graph():
    states = [_state(4), _state(6, offset=1.0)]
    node_feat = torch.cat([state.node_features() for state in states])
    pos = torch.cat([state.pos for state in states])
    vel = torch.cat([state.vel for state in states])
    drive = torch.zeros_like(pos)

    edge_parts = []
    contact_parts = []
    batch_parts = []
    ptr = [0]
    offset = 0
    for graph_id, state in enumerate(states):
        edge_index, is_contact = build_edges(state.num_nodes)
        edge_parts.append(edge_index + offset)
        contact_parts.append(is_contact)
        batch_parts.append(torch.full(
            (state.num_nodes,), graph_id, dtype=torch.long
        ))
        offset += state.num_nodes
        ptr.append(offset)

    return BatchedGraph(
        node_feat=node_feat,
        pos=pos,
        vel=vel,
        drive=drive,
        edge_index=torch.cat(edge_parts, dim=1),
        is_contact=torch.cat(contact_parts),
        batch_idx=torch.cat(batch_parts),
        num_graphs=len(states),
        ptr=torch.tensor(ptr, dtype=torch.long),
    )


def _prediction_sum(pred):
    return sum(value.float().sum() for value in pred.values())


def test_old_model_strict_state_dict_still_compatible():
    """新增路径不能改变旧模型的参数结构。"""
    old_model = DLOWorldModel(
        hidden=16, n_message_passing=2, n_topo_classes=3
    )
    reloaded = DLOWorldModel(
        hidden=16, n_message_passing=2, n_topo_classes=3
    )
    reloaded.load_state_dict(old_model.state_dict(), strict=True)


def test_single_graph_material_condition_shapes_backward_and_effect():
    torch.manual_seed(7)
    state = _state(5)
    action = DLOAction(
        grasp_idx=torch.tensor([0, 4]),
        delta=torch.tensor([[0.02, 0.0, 0.0], [0.0, 0.01, 0.0]]),
    )
    edge_index, is_contact = build_edges(state.num_nodes)
    drive = action.to_node_drive(state.num_nodes)
    model = MaterialConditionedDLOWorldModel(
        hidden=24, n_message_passing=2, n_topo_classes=4
    )
    material_a = torch.zeros(MATERIAL_INPUT_DIM)
    material_b = torch.ones(MATERIAL_INPUT_DIM)

    pred_a = model(
        state, drive, edge_index, is_contact, material_a
    )
    pred_b = model(
        state, drive, edge_index, is_contact, material_b
    )

    assert pred_a["acc"].shape == (5, 3)
    assert pred_a["pos_next"].shape == (5, 3)
    assert pred_a["vel_next"].shape == (5, 3)
    assert pred_a["tension"].shape == (5,)
    assert pred_a["contact_logit"].shape == (5,)
    assert pred_a["topo_logits"].shape == (4,)
    assert pred_a["fail_logit"].ndim == 0
    assert not torch.allclose(pred_a["acc"], pred_b["acc"])

    _prediction_sum(pred_a).backward()
    material_grads = [
        parameter.grad for parameter in model.material_enc.parameters()
    ]
    assert any(
        grad is not None and torch.count_nonzero(grad).item() > 0
        for grad in material_grads
    )


def test_rollout_rebuilds_edges_from_each_predicted_position():
    torch.manual_seed(11)
    state = _state(5)
    seen_action_positions = []
    seen_drives = []

    class RecordingAction(DLOAction):
        def to_node_drive(self, num_nodes, current_pos=None):
            assert current_pos is not None
            seen_action_positions.append(current_pos.clone())
            drive = super().to_node_drive(
                num_nodes, current_pos=current_pos
            )
            seen_drives.append(drive.clone())
            return drive

    actions = [
        RecordingAction(
            grasp_idx=torch.tensor([0]),
            delta=torch.zeros(1, 3),
            target_pos=torch.tensor([[0.1, 0.0, 0.0]]),
        ),
        RecordingAction(
            grasp_idx=torch.tensor([4]),
            delta=torch.zeros(1, 3),
            target_pos=torch.tensor([[0.3, 0.1, 0.0]]),
        ),
    ]
    model = MaterialConditionedDLOWorldModel(
        hidden=16, n_message_passing=1
    )
    seen_positions = []

    def edge_builder(pos):
        seen_positions.append(pos.clone())
        return build_edges(pos.shape[0], device=pos.device)

    trajectory = model.rollout(
        state,
        actions,
        edge_builder,
        torch.ones(MATERIAL_INPUT_DIM),
    )

    assert len(trajectory) == 3
    assert len(seen_positions) == 2
    assert torch.equal(seen_positions[0], state.pos)
    assert torch.equal(seen_positions[1], trajectory[1].pos)
    assert torch.equal(seen_action_positions[0], state.pos)
    assert torch.equal(seen_action_positions[1], trajectory[1].pos)
    assert torch.allclose(
        seen_drives[1][4], actions[1].target_pos[0] - trajectory[1].pos[4]
    )


def test_rollout_derives_node_contact_from_current_dynamic_edges():
    state = _state(5)
    # 故意给一个与动态边矛盾的缓存值；forward 必须看到边派生结果。
    state.contact.fill_(0.0)
    action = DLOAction(
        torch.tensor([2]), torch.tensor([[0.01, 0.0, 0.0]])
    )
    model = MaterialConditionedDLOWorldModel(
        hidden=16, n_message_passing=1
    )
    seen_contact = []
    original_forward = model.forward

    def capture(state_input, *args, **kwargs):
        seen_contact.append(state_input.contact.clone())
        return original_forward(state_input, *args, **kwargs)

    model.forward = capture

    def edge_builder(pos):
        return build_edges(
            pos.shape[0], torch.tensor([[0, 4]]), device=pos.device
        )

    model.rollout(
        state,
        [action],
        edge_builder,
        torch.ones(MATERIAL_INPUT_DIM),
    )

    assert torch.equal(
        seen_contact[0], torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0])
    )


def test_batched_material_condition_shapes_backward_and_effect():
    torch.manual_seed(19)
    bg = _batched_graph()
    model = BatchedMaterialConditionedDLOWorldModel(
        hidden=24, n_message_passing=2, n_topo_classes=4
    )
    material_a = torch.zeros(2, MATERIAL_INPUT_DIM)
    material_b = torch.ones(2, MATERIAL_INPUT_DIM)

    pred_a = model(bg, material_a)
    pred_b = model(bg, material_b)

    assert pred_a["acc"].shape == (10, 3)
    assert pred_a["pos_next"].shape == (10, 3)
    assert pred_a["vel_next"].shape == (10, 3)
    assert pred_a["tension"].shape == (10,)
    assert pred_a["contact_logit"].shape == (10,)
    assert pred_a["topo_logits"].shape == (2, 4)
    assert pred_a["fail_logit"].shape == (2,)
    assert not torch.allclose(pred_a["acc"], pred_b["acc"])

    _prediction_sum(pred_a).backward()
    material_grads = [
        parameter.grad for parameter in model.material_enc.parameters()
    ]
    assert any(
        grad is not None and torch.count_nonzero(grad).item() > 0
        for grad in material_grads
    )
