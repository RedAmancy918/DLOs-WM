"""Episode 级材料参数采样。

该模块只负责“采什么参数”，不依赖 Genesis。仿真 provider 负责把采样结果
真正写入求解器，并在 episode metadata 中记录最终生效值。这样可以单独测试
随机化分布，也便于构造只改变一个参数的成组反事实实验。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch


@dataclass(frozen=True)
class MaterialParameters:
    """均匀 DLO 的图级材料参数（SI 单位由数据集文档固定）。"""

    K: float
    E: float
    G: float
    linear_density: float
    radius: float
    mu_self_static: float
    mu_self_kinetic: float

    def __post_init__(self):
        for name in ("K", "E", "G", "linear_density", "radius"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} 必须是有限正数，得到 {value}")
        for name in ("mu_self_static", "mu_self_kinetic"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} 必须是有限非负数，得到 {value}")
        if self.mu_self_kinetic > self.mu_self_static:
            raise ValueError("动摩擦系数不应大于静摩擦系数")

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class MaterialRandomizationConfig:
    """相对 base 参数的采样范围。

    K/E/G 在 log-space 采样，其余正参数在线性空间采样。Workshop 第一版
    默认固定 G 和摩擦，只随机化可由当前 centerline 状态充分观察的参数。
    """

    K_scale: tuple[float, float] = (0.7, 1.4)
    E_scale: tuple[float, float] = (0.7, 1.4)
    G_scale: tuple[float, float] = (1.0, 1.0)
    density_scale: tuple[float, float] = (0.8, 1.2)
    radius_scale: tuple[float, float] = (0.9, 1.1)
    mu_static_scale: tuple[float, float] = (1.0, 1.0)
    mu_kinetic_scale: tuple[float, float] = (1.0, 1.0)

    def __post_init__(self):
        for name, bounds in asdict(self).items():
            low, high = bounds
            if low <= 0 or high < low:
                raise ValueError(f"{name} 范围非法: {bounds}")


def _sample_scale(
    bounds: tuple[float, float],
    generator: torch.Generator,
    *,
    log_space: bool,
) -> float:
    low, high = bounds
    if low == high:
        return float(low)
    u = float(torch.rand((), generator=generator, device="cpu"))
    if log_space:
        return math.exp(math.log(low) + u * (math.log(high) - math.log(low)))
    return low + u * (high - low)


def sample_material_parameters(
    base: MaterialParameters,
    config: MaterialRandomizationConfig,
    generator: torch.Generator,
) -> MaterialParameters:
    """独立采样一组 episode 级材料参数。"""

    mu_s = base.mu_self_static * _sample_scale(
        config.mu_static_scale, generator, log_space=False
    )
    mu_k = base.mu_self_kinetic * _sample_scale(
        config.mu_kinetic_scale, generator, log_space=False
    )
    # 独立采样可能令 mu_k > mu_s；物理约束在这里显式投影，而不是静默交给 solver。
    mu_k = min(mu_k, mu_s)
    return MaterialParameters(
        K=base.K * _sample_scale(config.K_scale, generator, log_space=True),
        E=base.E * _sample_scale(config.E_scale, generator, log_space=True),
        G=base.G * _sample_scale(config.G_scale, generator, log_space=True),
        linear_density=base.linear_density
        * _sample_scale(config.density_scale, generator, log_space=False),
        radius=base.radius
        * _sample_scale(config.radius_scale, generator, log_space=False),
        mu_self_static=mu_s,
        mu_self_kinetic=mu_k,
    )


def counterfactual_material_sweep(
    base: MaterialParameters,
    parameter: str,
    scales: list[float] | tuple[float, ...],
) -> list[MaterialParameters]:
    """生成只改变一个参数的材料组，供 paired counterfactual 评估。"""

    allowed = {
        "K",
        "E",
        "G",
        "linear_density",
        "radius",
        "mu_self_static",
        "mu_self_kinetic",
    }
    if parameter not in allowed:
        raise ValueError(f"不支持的反事实参数: {parameter}")
    result = []
    values = base.to_dict()
    for scale in scales:
        if scale <= 0:
            raise ValueError(f"scale 必须为正数，得到 {scale}")
        changed = dict(values)
        changed[parameter] = values[parameter] * scale
        if parameter == "mu_self_static":
            changed["mu_self_kinetic"] = min(
                changed["mu_self_kinetic"], changed["mu_self_static"]
            )
        elif parameter == "mu_self_kinetic":
            changed["mu_self_kinetic"] = min(
                changed["mu_self_kinetic"], changed["mu_self_static"]
            )
        result.append(MaterialParameters(**changed))
    return result
