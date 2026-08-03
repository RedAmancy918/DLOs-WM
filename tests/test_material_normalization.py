import torch
import pytest

from dlo_wm.data.normalization import MaterialFeatureNormalizer
from dlo_wm.data.schema import MaterialCondition


def _material(scale):
    return MaterialCondition(
        rest_length=torch.full((3,), 0.1),
        node_mass=torch.full((4,), 0.02 * scale),
        node_radius=torch.full((4,), 0.005 * scale),
        K=torch.tensor(10.0 * scale),
        E=torch.tensor(20.0 * scale),
        G=torch.tensor(30.0),
        mu_self_static=torch.tensor(0.3),
        mu_self_kinetic=torch.tensor(0.2),
    )


def test_fit_transform_and_round_trip():
    materials = [_material(0.5), _material(1.0), _material(2.0)]
    normalizer = MaterialFeatureNormalizer.fit(materials)
    transformed = torch.stack([
        normalizer.transform_material(material) for material in materials
    ])

    assert torch.allclose(transformed.mean(0), torch.zeros(7), atol=1e-6)
    # G 与摩擦在这个训练集固定，必须得到 0 而不是数值爆炸。
    assert torch.equal(transformed[:, [2, 5, 6]], torch.zeros(3, 3))

    # 训练中未变化的 G/摩擦即使评估时变化，也不能经过未训练的随机权重。
    out_of_scope = materials[0].global_features().clone()
    out_of_scope[[2, 5, 6]] += torch.tensor([2.0, 0.5, 0.5])
    transformed_ood = normalizer.transform(out_of_scope)
    assert torch.equal(transformed_ood[[2, 5, 6]], torch.zeros(3))

    loaded = MaterialFeatureNormalizer.from_dict(normalizer.to_dict())
    assert torch.equal(loaded.mean, normalizer.mean)
    assert torch.equal(loaded.std, normalizer.std)

    bad = normalizer.to_dict()
    bad["feature_names"] = list(reversed(bad["feature_names"]))
    with pytest.raises(ValueError, match="特征契约"):
        MaterialFeatureNormalizer.from_dict(bad)
