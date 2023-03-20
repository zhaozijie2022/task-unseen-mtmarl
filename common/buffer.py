#  采样transition [obs_past, action_past, reward_past, hidden_past,
#                   obs, action, reward, hidden,
#                   obs_next, done
from typing import Dict, List, Tuple, Any
import numpy as np


class ReplayBuffer:
    def __init__(self, obs_dim, action_dim, hidden_dim_en, max_size):
        self._obs_past = np.zeros((max_size, obs_dim))
        self._action_past = np.zeros((max_size, action_dim))
        self._reward_past = np.zeros((max_size, 1))
        self._hidden_past = np.zeros((max_size, hidden_dim_en))

        self._obs = np.zeros((max_size, obs_dim))
        self._action = np.zeros((max_size, action_dim))
        self._reward = np.zeros((max_size, 1))
        self._hidden = np.zeros((max_size, hidden_dim_en))

        self._obs_next = np.zeros((max_size, obs_dim))
        self._done = np.zeros((max_size, 1))

        self.max_size = max_size
        self._top = 0
        self.size = 0

    def clear(self):
        self._top = 0  # 指针, 指向下一个要写入的位置
        self.size = 0  # 当前buffer大小

    def add(self, obs_past, action_past, reward_past, hidden_past,
            obs, action, reward, hidden, obs_next, done):
        # 压入一个transition, 输入 np.ndarray
        self._obs_past[self._top] = obs_past  # ndarray只有一个索引时, 自动指代行索引
        self._action_past[self._top] = action_past
        self._reward_past[self._top] = reward_past
        self._hidden_past[self._top] = hidden_past

        self._obs[self._top] = obs
        self._action[self._top] = action
        self._reward[self._top] = reward
        self._hidden[self._top] = hidden

        self._obs_next[self._top] = obs_next
        self._done[self._top] = done

        self._top = (self._top + 1) % self.max_size  # 满了就从头开始覆盖
        if self.size < self.max_size:
            self.size += 1

    def add_traj(self, traj: Dict[str, np.ndarray]):
        # 压入一条traj, traj就是sampler.obtain_samples()的返回的trajs_n[i]
        for (obs_past, action_past, reward_past, hidden_past,
             obs, action, reward, hidden, obs_next, done) in zip(
                traj["obs_past"], traj["action_past"], traj["reward_past"], traj["hidden_past"],
                traj["obs"], traj["action"], traj["reward"], traj["hidden"],
                traj["obs_next"], traj["done"],
        ):
            self.add(obs_past, action_past, reward_past, hidden_past,
                     obs, action, reward, hidden, obs_next, done)

    def sample_batch(self, indices):
        # 给定indices, 采样transition
        return dict(
            obs_past=self._obs_past[indices],
            action_past=self._action_past[indices],
            reward_past=self._reward_past[indices],
            hidden_past=self._hidden_past[indices],

            obs=self._obs[indices],
            action=self._action[indices],
            reward=self._reward[indices],
            hidden=self._hidden[indices],

            obs_next=self._obs_next[indices],
            done=self._done[indices],
        )
