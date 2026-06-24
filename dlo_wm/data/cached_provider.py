"""
从磁盘读取预生成轨迹的 TrajectoryProvider。

DLO-Lab 仿真慢（一条 loop 轨迹 ~十几秒），不能在训练循环里每个 epoch 重新仿真。
做法：用 gen_dataset.py 预生成一批轨迹存盘，训练时用本 provider 秒级读取。
接口与 SyntheticRope / DLOLabProvider 完全一致，trainer/rollout 不用改。
"""
from __future__ import annotations
import random
import torch

from .dataset import TrajectoryProvider


class CachedTrajectoryProvider(TrajectoryProvider):
    def __init__(self, path: str, seed: int = 0):
        blob = torch.load(path, weights_only=False)
        self._trajs = blob["trajs"]          # list[(states, actions, cpairs)]
        self._n = blob["num_nodes"]
        self._rng = random.Random(seed)
        if not self._trajs:
            raise ValueError(f"empty dataset: {path}")

    @property
    def num_nodes(self) -> int:
        return self._n

    def sample_trajectory(self, T: int | None = None):
        states, actions, cpairs = self._rng.choice(self._trajs)
        if T is not None and T < len(actions):
            states = states[: T + 1]
            actions = actions[: T]
            cpairs = cpairs[: T + 1]
        return states, actions, cpairs
