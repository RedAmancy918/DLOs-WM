import torch

from dlo_wm.data.material_sampling import (
    MaterialParameters,
    MaterialRandomizationConfig,
    counterfactual_material_sweep,
    sample_material_parameters,
)


def _base():
    return MaterialParameters(
        K=5e4,
        E=1e5,
        G=1e4,
        linear_density=0.1,
        radius=0.005,
        mu_self_static=0.3,
        mu_self_kinetic=0.25,
    )


def test_sampling_is_reproducible_and_bounded():
    cfg = MaterialRandomizationConfig()
    a = sample_material_parameters(_base(), cfg, torch.Generator().manual_seed(7))
    b = sample_material_parameters(_base(), cfg, torch.Generator().manual_seed(7))
    assert a == b
    assert 0.7 * _base().K <= a.K <= 1.4 * _base().K
    assert 0.7 * _base().E <= a.E <= 1.4 * _base().E
    assert a.G == _base().G
    assert a.mu_self_static == _base().mu_self_static
    assert a.mu_self_kinetic == _base().mu_self_kinetic


def test_counterfactual_sweep_changes_only_one_parameter():
    variants = counterfactual_material_sweep(_base(), "E", (0.5, 2.0))
    assert [v.E for v in variants] == [0.5 * _base().E, 2.0 * _base().E]
    for variant in variants:
        assert variant.K == _base().K
        assert variant.G == _base().G
        assert variant.linear_density == _base().linear_density
