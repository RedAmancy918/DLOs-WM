"""DLOEpisode 的版本化磁盘序列化。

磁盘内容只由字典、列表、基础标量、``None`` 和普通张量组成，不把
dataclass 实例直接交给 pickle。这样 schema 演进时可以先检查版本，再做
显式迁移或拒绝读取。
"""

from __future__ import annotations

from os import PathLike
from typing import Any

import torch

from .schema import DLOAction, DLOEpisode, DLOState, MaterialCondition


FORMAT_VERSION = 2


def material_to_dict(material: MaterialCondition) -> dict[str, Any]:
    """把材料条件转换为不含 Python 自定义对象的字典。"""
    material.validate()
    return {
        "rest_length": material.rest_length,
        "node_mass": material.node_mass,
        "node_radius": material.node_radius,
        "K": material.K,
        "E": material.E,
        "G": material.G,
        "mu_self_static": material.mu_self_static,
        "mu_self_kinetic": material.mu_self_kinetic,
    }


def material_from_dict(data: dict[str, Any]) -> MaterialCondition:
    """从材料字典恢复并验证 MaterialCondition。"""
    _require_dict(data, "material")
    material = MaterialCondition(
        rest_length=data["rest_length"],
        node_mass=data["node_mass"],
        node_radius=data["node_radius"],
        K=data["K"],
        E=data["E"],
        G=data["G"],
        mu_self_static=data["mu_self_static"],
        mu_self_kinetic=data["mu_self_kinetic"],
    )
    return material.validate()


def state_to_dict(state: DLOState) -> dict[str, torch.Tensor]:
    """把一帧状态转换为张量字典。"""
    return {
        "pos": state.pos,
        "vel": state.vel,
        "tension": state.tension,
        "contact": state.contact,
        "topology": state.topology,
    }


def state_from_dict(data: dict[str, Any]) -> DLOState:
    """从状态字典恢复旧接口兼容的 DLOState。"""
    _require_dict(data, "state")
    return DLOState(
        pos=data["pos"],
        vel=data["vel"],
        tension=data["tension"],
        contact=data["contact"],
        topology=data["topology"],
    )


def action_to_dict(action: DLOAction) -> dict[str, Any]:
    """把动作及其可选控制字段转换为字典。"""
    return {
        "grasp_idx": action.grasp_idx,
        "delta": action.delta,
        "gripper_id": action.gripper_id,
        "target_pos": action.target_pos,
        "target_vel": action.target_vel,
        "grasp_active": action.grasp_active,
        "duration": action.duration,
    }


def action_from_dict(data: dict[str, Any]) -> DLOAction:
    """从动作字典恢复 DLOAction；缺少新字段时按 None 处理。"""
    _require_dict(data, "action")
    return DLOAction(
        grasp_idx=data["grasp_idx"],
        delta=data["delta"],
        gripper_id=data.get("gripper_id"),
        target_pos=data.get("target_pos"),
        target_vel=data.get("target_vel"),
        grasp_active=data.get("grasp_active"),
        duration=data.get("duration"),
    )


def episode_to_dict(episode: DLOEpisode) -> dict[str, Any]:
    """生成顶层带格式版本的 primitive episode 字典。"""
    episode.validate()
    _validate_primitive(episode.metadata, "metadata")
    return {
        "format_version": FORMAT_VERSION,
        "material": material_to_dict(episode.material),
        "states": [state_to_dict(state) for state in episode.states],
        "actions": [action_to_dict(action) for action in episode.actions],
        "contact_pairs": list(episode.contact_pairs),
        "macro_dt": float(episode.macro_dt),
        "task": episode.task,
        "seed": episode.seed,
        "id": episode.id,
        "metadata": episode.metadata,
    }


def episode_from_dict(data: dict[str, Any]) -> DLOEpisode:
    """检查格式版本后恢复完整 episode，未知版本不会静默兼容。"""
    _require_dict(data, "episode")
    if "format_version" not in data:
        raise ValueError("episode 缺少 format_version")
    version = data["format_version"]
    if version != FORMAT_VERSION:
        raise ValueError(
            f"不支持的 episode 格式版本 {version!r}，当前仅支持 {FORMAT_VERSION}")

    metadata = data.get("metadata", {})
    _validate_primitive(metadata, "metadata")
    episode = DLOEpisode(
        material=material_from_dict(data["material"]),
        states=[state_from_dict(state) for state in data["states"]],
        actions=[action_from_dict(action) for action in data["actions"]],
        contact_pairs=list(data["contact_pairs"]),
        macro_dt=float(data["macro_dt"]),
        task=data.get("task", ""),
        seed=data.get("seed", 0),
        id=data.get("id", ""),
        metadata=metadata,
    )
    return episode.validate()


def save_episode(episode: DLOEpisode,
                 path: str | PathLike[str]) -> None:
    """把单条 v2 episode 保存到磁盘。"""
    torch.save(episode_to_dict(episode), path)


def load_episode(path: str | PathLike[str],
                 map_location: str | torch.device | None = "cpu") -> DLOEpisode:
    """从磁盘安全读取 primitive 字典并恢复单条 v2 episode。"""
    data = torch.load(path, map_location=map_location, weights_only=True)
    return episode_from_dict(data)


def _require_dict(value: Any, name: str) -> None:
    """为损坏文件给出比后续索引异常更清楚的提示。"""
    if not isinstance(value, dict):
        raise TypeError(f"{name} 必须是字典")


def _validate_primitive(value: Any, path: str) -> None:
    """递归拒绝 metadata 中需要自定义 pickle 类才能恢复的对象。"""
    if value is None or isinstance(value, (bool, int, float, str, torch.Tensor)):
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_primitive(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} 的字典键必须是字符串")
            _validate_primitive(item, f"{path}.{key}")
        return
    raise TypeError(f"{path} 包含不支持的序列化类型 {type(value).__name__}")
