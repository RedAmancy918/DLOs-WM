"""材料图级特征的训练集归一化。"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .schema import (
    MATERIAL_FEATURE_DIM,
    MATERIAL_FEATURE_NAMES,
    MaterialCondition,
)


@dataclass
class MaterialFeatureNormalizer:
    """只用训练 split 拟合的逐维标准化器。"""

    mean: torch.Tensor
    std: torch.Tensor
    eps: float = 1e-8

    def __post_init__(self):
        if self.mean.ndim != 1 or self.std.shape != self.mean.shape:
            raise ValueError("mean/std 必须是同 shape 的一维张量")
        if self.mean.numel() != MATERIAL_FEATURE_DIM:
            raise ValueError(
                f"材料归一化维度必须为 {MATERIAL_FEATURE_DIM}"
            )
        if not torch.isfinite(self.mean).all():
            raise ValueError("mean 包含非有限值")
        if not torch.isfinite(self.std).all() or (self.std < 0).any():
            raise ValueError("std 必须是有限非负数")

    @classmethod
    def fit(
        cls,
        materials: list[MaterialCondition],
        eps: float = 1e-8,
    ) -> "MaterialFeatureNormalizer":
        if not materials:
            raise ValueError("至少需要一个材料条件来拟合归一化器")
        features = torch.stack([
            material.global_features().detach().cpu().float()
            for material in materials
        ])
        # unbiased=False 使单 episode 冒烟数据也得到有限统计量。
        return cls(
            mean=features.mean(dim=0),
            std=features.std(dim=0, unbiased=False),
            eps=eps,
        )

    def transform(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != self.mean.numel():
            raise ValueError(
                f"材料特征最后一维应为 {self.mean.numel()}，"
                f"实际为 {features.shape[-1]}"
            )
        mean = self.mean.to(device=features.device, dtype=features.dtype)
        std = self.std.to(device=features.device, dtype=features.dtype)
        active = std > self.eps
        scale = torch.where(active, std, torch.ones_like(std))
        normalized = (features - mean) / scale
        # 训练中固定的维度没有可辨识权重。即使评估文件意外改变该值，也必须
        # 继续置 0，而不是让随机初始化的未训练权重制造虚假的“条件响应”。
        return torch.where(active, normalized, torch.zeros_like(normalized))

    def transform_material(self, material: MaterialCondition) -> torch.Tensor:
        return self.transform(material.global_features())

    def to_dict(self) -> dict:
        return {
            "mean": self.mean.detach().cpu(),
            "std": self.std.detach().cpu(),
            "eps": self.eps,
            "feature_names": list(MATERIAL_FEATURE_NAMES),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "MaterialFeatureNormalizer":
        required = {"mean", "std", "feature_names"}
        missing = required - set(payload)
        if missing:
            raise ValueError(f"normalization 缺少字段: {sorted(missing)}")
        feature_names = tuple(payload["feature_names"])
        if feature_names != MATERIAL_FEATURE_NAMES:
            raise ValueError(
                "checkpoint 材料特征契约不匹配: "
                f"{feature_names!r} != {MATERIAL_FEATURE_NAMES!r}"
            )
        return cls(
            mean=torch.as_tensor(payload["mean"]).float(),
            std=torch.as_tensor(payload["std"]).float(),
            eps=float(payload.get("eps", 1e-8)),
        )
