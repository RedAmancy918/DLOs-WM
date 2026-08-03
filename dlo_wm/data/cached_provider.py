"""
从磁盘读取预生成轨迹的 TrajectoryProvider。

DLO-Lab 仿真慢（一条 loop 轨迹 ~十几秒），不能在训练循环里每个 epoch 重新仿真。
做法：用 gen_dataset.py 预生成一批轨迹存盘，训练时用本 provider 秒级读取。
接口与 SyntheticRope / DLOLabProvider 完全一致，trainer/rollout 不用改。
"""
from __future__ import annotations
import pickle
import random
import torch

from .dataset import TrajectoryProvider, slice_episode
from .schema import DLOEpisode
from .serialization import FORMAT_VERSION, episode_from_dict


class CachedTrajectoryProvider(TrajectoryProvider):
    def __init__(self, path: str, seed: int = 0):
        # v2 先走 weights-only 安全加载；只有明确的旧 {trajs,...} 文件才
        # 回退到 pickle 兼容路径。map_location 固定 CPU，避免 GPU 缓存无法载入。
        try:
            blob = torch.load(
                path, map_location="cpu", weights_only=True
            )
        except pickle.UnpicklingError as safe_error:
            legacy_blob = torch.load(
                path, map_location="cpu", weights_only=False
            )
            if not (
                isinstance(legacy_blob, dict) and "trajs" in legacy_blob
            ):
                raise ValueError(
                    "非 v1 缓存包含 weights-only 不允许的 Python 对象"
                ) from safe_error
            blob = legacy_blob
        self._rng = random.Random(seed)
        self._trajs = None
        self._episodes = None
        self._metadata = {}

        if isinstance(blob, dict) and "trajs" in blob:
            # v1：保持原有 {trajs, num_nodes} 存档完全可读。
            self._trajs = blob["trajs"]
            self._n = int(blob["num_nodes"])
            self._format_version = 1
            if not self._trajs:
                raise ValueError(f"empty dataset: {path}")
            return

        self._episodes, self._metadata = self._load_v2(blob, path)
        self._format_version = FORMAT_VERSION
        first_n = self._episodes[0].states[0].num_nodes
        for episode in self._episodes:
            episode.validate()
            if episode.states[0].num_nodes != first_n:
                raise ValueError(
                    "CachedTrajectoryProvider 当前要求同一数据集的"
                    " episode 节点数一致"
                )
        declared_n = self._metadata.get("num_nodes")
        if declared_n is not None and int(declared_n) != first_n:
            raise ValueError(
                f"数据集声明 num_nodes={declared_n}，"
                f"但 episode 实际为 {first_n}"
            )
        self._n = first_n

    @staticmethod
    def _load_v2(blob, path):
        """读取单 episode 或带 split 信息的多 episode 容器。"""
        if not isinstance(blob, dict):
            raise ValueError(
                f"不支持的缓存数据类型 {type(blob).__name__}: {path}"
            )

        version = blob.get("format_version")
        if version != FORMAT_VERSION:
            raise ValueError(
                f"不支持的顶层 format_version={version!r}；"
                f"当前仅支持 v1 trajs 或 v{FORMAT_VERSION} episodes"
            )

        if "episodes" in blob:
            raw_episodes = blob["episodes"]
            # 容器中除 episodes 外的字段是 split/生成配置元数据。
            metadata = {
                key: value for key, value in blob.items()
                if key not in {"format_version", "episodes"}
            }
        else:
            # serialization.episode_to_dict 生成的单 episode 顶层字典。
            raw_episodes = [blob]
            metadata = {}

        if not isinstance(raw_episodes, (list, tuple)) or not raw_episodes:
            raise ValueError(f"empty v{FORMAT_VERSION} episode dataset: {path}")

        episodes = []
        for raw_episode in raw_episodes:
            if isinstance(raw_episode, dict):
                episode = episode_from_dict(raw_episode)
            else:
                raise ValueError(
                    "v2 episodes 的每一项必须是版本化 primitive 字典，"
                    f"实际为 {type(raw_episode).__name__}"
                )
            episode.validate()
            episodes.append(episode)
        return tuple(episodes), metadata

    @property
    def num_nodes(self) -> int:
        return self._n

    @property
    def num_episodes(self) -> int:
        """缓存中的 episode 数，v1 轨迹也按 episode 计数。"""
        return len(self._episodes if self._episodes is not None else self._trajs)

    @property
    def episodes(self) -> tuple[DLOEpisode, ...]:
        """
        v2 episode 的只读视图，便于训练代码顺序遍历 split。

        v1 没有材料信息，不伪造 episode，而是明确报错。
        """
        if self._episodes is None:
            raise AttributeError("v1 缓存不包含可遍历的 DLOEpisode")
        return self._episodes

    @property
    def metadata(self) -> dict:
        """返回顶层 split/生成配置的浅拷贝。"""
        return dict(self._metadata)

    @property
    def format_version(self) -> int:
        return self._format_version

    def sample_episode(self, T: int | None = None) -> DLOEpisode:
        if self._episodes is None:
            raise NotImplementedError(
                "v1 缓存没有材料条件，无法构造 DLOEpisode"
            )
        episode = self._rng.choice(self._episodes)
        return slice_episode(episode, T=T)

    def sample_trajectory(self, T: int | None = None):
        if self._episodes is not None:
            episode = self.sample_episode(T=T)
            return episode.states, episode.actions, episode.contact_pairs

        states, actions, cpairs = self._rng.choice(self._trajs)
        if T is not None and T < len(actions):
            states = states[: T + 1]
            actions = actions[: T]
            cpairs = cpairs[: T + 1]
        return states, actions, cpairs
