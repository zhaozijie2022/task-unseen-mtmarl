import numpy as np
import torch
import os
import time
from mpe.lib4occupy import make_env


scenario_name = "occupy"
max_ep_len = 100
num_eps = 10


if __name__ == '__main__':
    env = make_env(scenario_name, num=3)
    obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
    obs_n = env.reset()
    episode_step = 0
    num_ep = 0
    while True:
        # action_n = [env.action_space[i].sample() for i in range(env.n)]
        action_n = []
        for i in range(env.n):
            tmp = np.random.uniform(0, 1, size=5)
            tmp = np.exp(tmp) / np.sum(np.exp(tmp))
            action_n.append(tmp)
        time.sleep(0.05)
        env.render()
        new_obs_n, rew_n, done_n, _ = env.step(action_n)
        obs_n = new_obs_n
        episode_step += 1
        terminal = (episode_step >= max_ep_len) or all(done_n)
        if terminal:
            num_ep += 1
            print("Eps %d, Sps: %d" % (num_ep, episode_step))
            episode_step = 0
            obs_n = env.reset()
        if num_ep >= num_eps:
            break

















