import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from dlo_wm.data.batch import collate_transitions
from dlo_wm.data.cached_provider import CachedTrajectoryProvider
from dlo_wm.data.dataset import (
    TrajectoryProvider,
    episode_to_transitions,
    make_transition_batch,
)
from dlo_wm.data.normalization import MaterialFeatureNormalizer
from dlo_wm.data.provenance import (
    extract_dataset_contract,
    sha256_file,
    validate_dataset_contract,
    validate_manifest_entry,
)
from dlo_wm.data.schema import (
    DLOAction,
    DLOEpisode,
    DLOState,
    MaterialCondition,
)
from dlo_wm.data.serialization import FORMAT_VERSION, episode_to_dict


def _material(num_nodes=4, K=100.0):
    return MaterialCondition(
        rest_length=torch.full((num_nodes - 1,), 0.1),
        node_mass=torch.full((num_nodes,), 0.02),
        node_radius=torch.full((num_nodes,), 0.005),
        K=torch.tensor(K),
        E=torch.tensor(20.0),
        G=torch.tensor(5.0),
        mu_self_static=torch.tensor(0.4),
        mu_self_kinetic=torch.tensor(0.3),
    )


def _state(num_nodes=4, offset=0.0):
    pos = torch.zeros(num_nodes, 3)
    pos[:, 0] = torch.arange(num_nodes, dtype=torch.float32) * 0.1 + offset
    return DLOState(
        pos=pos,
        vel=torch.zeros(num_nodes, 3),
        tension=torch.full((num_nodes,), offset),
        contact=torch.zeros(num_nodes),
        topology=torch.tensor(0, dtype=torch.long),
    )


def _episode(num_nodes=4, horizon=2, K=100.0, episode_id="episode-a"):
    states = [_state(num_nodes, offset=0.01 * t)
              for t in range(horizon + 1)]
    actions = [
        DLOAction(
            grasp_idx=torch.tensor([t % num_nodes]),
            delta=torch.tensor([[0.01, 0.0, 0.0]]),
        )
        for t in range(horizon)
    ]
    contact_pairs = [
        torch.empty((0, 2), dtype=torch.long)
        for _ in range(horizon + 1)
    ]
    return DLOEpisode(
        material=_material(num_nodes, K=K),
        states=states,
        actions=actions,
        contact_pairs=contact_pairs,
        macro_dt=0.04,
        task="loop",
        seed=7,
        id=episode_id,
        metadata={"counterfactual_group_id": "group-1"},
    ).validate()


class _LegacyProvider(TrajectoryProvider):
    """只实现旧契约，用于防止 v2 改造破坏现有训练路径。"""

    def __init__(self, num_nodes=4):
        self._n = num_nodes

    @property
    def num_nodes(self):
        return self._n

    def sample_trajectory(self, T=2):
        episode = _episode(self._n, horizon=T)
        return episode.states, episode.actions, episode.contact_pairs


def test_legacy_provider_and_explicit_episode_adapter_stay_compatible():
    provider = _LegacyProvider()

    samples = make_transition_batch(provider, n_traj=1, T=2)

    assert len(samples) == 2
    assert all(sample["material"] is None for sample in samples)
    assert collate_transitions(samples).material_features is None

    adapted = provider.sample_episode(
        T=1,
        material=_material(),
        macro_dt=0.04,
        episode_id="adapted",
    )
    assert adapted.horizon == 1
    assert adapted.episode_id == "adapted"
    assert adapted.material is not None


def test_cached_provider_reads_v1_and_keeps_tuple_api(tmp_path):
    path = tmp_path / "legacy.pt"
    trajectory = _LegacyProvider().sample_trajectory(T=2)
    torch.save({"trajs": [trajectory], "num_nodes": 4}, path)

    provider = CachedTrajectoryProvider(path)
    states, actions, contact_pairs = provider.sample_trajectory(T=1)

    assert provider.format_version == 1
    assert provider.num_episodes == 1
    assert len(states) == 2
    assert len(actions) == 1
    assert len(contact_pairs) == 2
    with pytest.raises(NotImplementedError, match="v1"):
        provider.sample_episode()


def test_cached_provider_reads_v2_container_and_slices_episode(tmp_path):
    path = tmp_path / "train-v2.pt"
    episodes = [
        _episode(K=100.0, episode_id="episode-a"),
        _episode(K=200.0, episode_id="episode-b"),
    ]
    torch.save({
        "format_version": FORMAT_VERSION,
        "episodes": [episode_to_dict(episode) for episode in episodes],
        "num_nodes": 4,
        "split": "train",
        "generator_config": {"seed": 7},
        "normalization": {"material": "physical"},
    }, path)

    provider = CachedTrajectoryProvider(path, seed=3)
    sampled = provider.sample_episode(T=1)
    states, actions, contact_pairs = provider.sample_trajectory(T=1)

    assert provider.format_version == FORMAT_VERSION
    assert provider.num_episodes == 2
    assert len(provider.episodes) == 2
    assert provider.metadata["split"] == "train"
    assert sampled.horizon == 1
    assert len(sampled.states) == 2
    assert len(sampled.contact_pairs) == 2
    assert len(states) == 2 and len(actions) == 1 and len(contact_pairs) == 2

    transitions = make_transition_batch(provider, n_traj=1, T=1)
    assert len(transitions) == 1
    assert transitions[0]["material"] is not None
    assert transitions[0]["episode_id"] in {"episode-a", "episode-b"}


def test_cached_provider_rejects_unknown_top_level_version(tmp_path):
    path = tmp_path / "future.pt"
    torch.save({
        "format_version": FORMAT_VERSION + 1,
        "episodes": [episode_to_dict(_episode())],
    }, path)

    with pytest.raises(ValueError, match="format_version"):
        CachedTrajectoryProvider(path)


def test_cached_provider_rejects_unversioned_pickled_v2_object(tmp_path):
    path = tmp_path / "unsafe-v2.pt"
    torch.save(_episode(), path)

    with pytest.raises(ValueError, match="weights-only"):
        CachedTrajectoryProvider(path)


def test_manifest_integrity_and_dataset_contract(tmp_path):
    path = tmp_path / "train.pt"
    episode = _episode()
    episode.metadata.update({
        "provider": "DLOLabProvider",
        "use_inextensible": True,
        "steps_interval": 40,
        "settle_steps": 40,
        "tension_scale": 1000.0,
        "contact_mode": "geometry",
        "contact_margin_scale": 0.5,
        "contact_distance_threshold": None,
    })
    generation = {
        "git_commit": "dlo-wm-commit",
        "git_dirty": False,
        "source_tree_sha256": "source-hash",
        "dlolab_git_commit": "dlolab-commit",
        "config": {
            "K": 100.0,
            "E": 20.0,
            "G": 5.0,
            "segment_mass": 0.02,
            "radius": 0.005,
            "mu_static": 0.4,
            "mu_kinetic": 0.3,
            "horizon": 2,
            "num_nodes": 4,
            "steps_interval": 40,
            "max_disp": 0.02,
            "lift_height": 0.011,
            "fold_back_frac": 0.2,
            "tension_scale": 1000.0,
            "motions": ["loop"],
            "seed": 7,
            "seed_offsets": {"train": 0},
            "id_k_scale": [0.7, 1.4],
            "id_e_scale": [0.7, 1.4],
            "id_density_scale": [0.8, 1.2],
            "id_radius_scale": [0.9, 1.1],
        },
    }
    torch.save({
        "format_version": FORMAT_VERSION,
        "episodes": [episode_to_dict(episode)],
        "num_nodes": 4,
        "split": "train",
        "generation": generation,
    }, path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "format_version": FORMAT_VERSION,
        "files": {"train": str(path)},
        "sha256": {"train": sha256_file(path)},
    }), encoding="utf-8")

    integrity = validate_manifest_entry(path, "train")
    provider = CachedTrajectoryProvider(path)
    contract = extract_dataset_contract(provider, label="train")
    validate_dataset_contract(provider, contract, label="train")

    assert integrity["sha256"] == sha256_file(path)
    assert contract["action_protocol"]["max_disp"] == 0.02
    assert contract["generation_source"]["source_tree_sha256"] == "source-hash"

    bad_episode = deepcopy(provider.episodes[0])
    bad_episode.material.K = torch.tensor(1000.0)
    bad_provider = SimpleNamespace(
        episodes=(bad_episode,),
        num_nodes=provider.num_nodes,
        metadata=provider.metadata,
    )
    with pytest.raises(ValueError, match="超出 ID range"):
        validate_dataset_contract(bad_provider, contract, label="train")

    ood_episode = deepcopy(episode)
    ood_episode.id = "ood-a"
    ood_episode.material.K = torch.tensor(170.0)
    ood_episode.metadata.update({
        "ood_parameter": "K",
        "ood_tail": "high",
        "ood_scale_range": [1.6, 2.0],
    })
    ood_generation = deepcopy(generation)
    ood_generation["config"].update({
        "ood_parameter": "K",
        "ood_tail": "high",
        "resolved_ood_scale": [1.6, 2.0],
    })
    ood_provider = SimpleNamespace(
        episodes=(ood_episode,),
        num_nodes=4,
        metadata={"generation": ood_generation},
    )
    validate_dataset_contract(
        ood_provider, contract, label="OOD", ood=True
    )
    ood_episode.metadata["ood_tail"] = "low"
    with pytest.raises(ValueError, match="tail"):
        validate_dataset_contract(
            ood_provider, contract, label="OOD", ood=True
        )
    ood_episode.metadata["ood_tail"] = "high"

    cf_episodes = []
    for index, scale in enumerate((0.7, 1.0, 1.4)):
        cf_episode = deepcopy(episode)
        cf_episode.id = f"cf-{index}"
        cf_episode.material.K = torch.tensor(100.0 * scale)
        cf_episode.metadata.update({
            "settle_steps": 0,
            "counterfactual_group_id": "cf-group",
            "counterfactual_parameter": "K",
            "counterfactual_scale": scale,
        })
        cf_episodes.append(cf_episode)
    cf_generation = deepcopy(generation)
    cf_generation["config"].update({
        "cf_parameter": "K",
        "cf_scales": [0.7, 1.0, 1.4],
    })
    cf_provider = SimpleNamespace(
        episodes=tuple(cf_episodes),
        num_nodes=4,
        metadata={"generation": cf_generation},
    )
    validate_dataset_contract(
        cf_provider,
        contract,
        label="counterfactual",
        counterfactual=True,
    )
    incomplete_cf = SimpleNamespace(
        episodes=tuple(cf_episodes[:-1]),
        num_nodes=4,
        metadata={"generation": cf_generation},
    )
    with pytest.raises(ValueError, match="cf_scales"):
        validate_dataset_contract(
            incomplete_cf,
            contract,
            label="counterfactual",
            counterfactual=True,
        )

    with path.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="SHA256"):
        validate_manifest_entry(path, "train")


def test_collate_material_rows_match_graph_ids():
    episode_a = _episode(num_nodes=3, horizon=1, K=100.0,
                         episode_id="a")
    episode_b = _episode(num_nodes=5, horizon=1, K=400.0,
                         episode_id="b")
    samples = (episode_to_transitions(episode_a)
               + episode_to_transitions(episode_b))

    batch = collate_transitions(samples)
    expected = torch.stack([
        episode_a.material.global_features(),
        episode_b.material.global_features(),
    ])

    assert torch.equal(batch.ptr, torch.tensor([0, 3, 8]))
    assert torch.equal(batch.material_features, expected)
    assert torch.equal(batch.batch_idx[:3], torch.zeros(3, dtype=torch.long))
    assert torch.equal(batch.batch_idx[3:], torch.ones(5, dtype=torch.long))
    # 模型使用 material_features[batch_idx] 广播，验证每个节点拿到所属图的行。
    assert torch.equal(
        batch.material_features[batch.batch_idx],
        torch.cat([
            expected[0].expand(3, -1),
            expected[1].expand(5, -1),
        ]),
    )

    normalizer = MaterialFeatureNormalizer.fit(
        [episode_a.material, episode_b.material]
    )
    normalized = collate_transitions(
        samples, material_normalizer=normalizer
    )
    assert torch.allclose(
        normalized.material_features.mean(dim=0),
        torch.zeros(7),
        atol=1e-6,
    )


def test_collate_rejects_mixed_material_presence():
    with_material = episode_to_transitions(_episode(horizon=1))[0]
    without_material = dict(with_material, material=None)

    with pytest.raises(ValueError, match="不能混合"):
        collate_transitions([with_material, without_material])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA")
def test_collate_accepts_samples_already_on_cuda():
    episode = _episode(horizon=1).to("cuda")
    batch = collate_transitions(
        episode_to_transitions(episode), device="cuda"
    )

    assert batch.node_feat.device.type == "cuda"
    assert batch.edge_index.device.type == "cuda"
    assert batch.material_features.device.type == "cuda"
