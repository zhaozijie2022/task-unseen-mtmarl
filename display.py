import os
import time
from tqdm import tqdm
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


from pearl.utils import create_dir
from mpe.lib4occupy import make_env
from pearl.marl.maddpg import MADDPGAgent
from pearl.sampler import rollout, sample_batch




num_agents = 2
scenario_name = "occupy"
latent_dim = 4  # z的维度
hidden_dim_act = 8
hidden_dim_critic = 16
hidden_dim_en = 8
device = torch.device(torch.device("cpu"))
max_buffer_size = 500000
max_step = 50 + 1

load_models_path = "./occupy/03_20_15_08/models/"
if __name__ == '__main__':
    env = make_env(scenario_name=scenario_name, num=num_agents)
    obs_dim = env.observation_space[0].shape[0]  # obs_dim: 2 * 3 + 2 = 8
    action_dim = env.action_space[0].n
    agents = [MADDPGAgent(num_agents=num_agents,
                          agent_id=i,
                          obs_dim=obs_dim,
                          action_dim=action_dim,
                          latent_dim=latent_dim,
                          hidden_dim_act=hidden_dim_act,
                          hidden_dim_critic=hidden_dim_critic,
                          hidden_dim_en=hidden_dim_en,
                          max_buffer_size=max_buffer_size,
                          device=device,
                          ) for i in range(num_agents)]
    for agent in agents:
        agent.load_model(load_models_path)

    n = env.n
    while True:
        obs_past_n = [np.zeros((obs_dim,)) for _ in range(n)]
        action_past_n = [np.zeros((action_dim,)) for _ in range(n)]
        reward_past_n = [np.zeros((1,)) for _ in range(n)]
        hidden_past_n = [np.zeros((hidden_dim_en,)) for _ in range(n)]

        obs_n = env.reset()
        action_n = [np.zeros((action_dim,)) for _ in range(n)]
        reward_n = [np.zeros((1,)) for _ in range(n)]
        hidden_n = [np.zeros((hidden_dim_en,)) for _ in range(n)]

        obs_next_n = [np.zeros((obs_dim,)) for _ in range(n)]
        # endregion

        cur_step = 0
        while True:
            # region S1.1 与环境交互, 将数据存储到xxx_n中
            with torch.no_grad():
                for i, agent in enumerate(agents):
                    if cur_step == 0:
                        latent, hidden = agent.get_prior_latent(obs=obs_n[i],
                                                                hidden=hidden_n[i])
                    else:
                        latent, hidden = agent.get_post_latent(obs=obs_past_n[i],
                                                               action=action_past_n[i],
                                                               reward=reward_past_n[i],
                                                               hidden=hidden_past_n[i])
                    # latent = latent.cpu().numpy().squeeze()  # (1, latent_dim) -> (latent_dim,)
                    action = agent.get_action(obs=obs_n[i], latent=latent)
                    action_n[i] = action.cpu().numpy().squeeze()  # (1, action_dim) -> (action_dim,)
                    hidden_n[i] = hidden.cpu().numpy().squeeze()  # (1, hidden_dim) -> (hidden_dim,)

            time.sleep(0.05)
            env.render()
            obs_next_n, reward_n, done_n, info_n = env.step(action_n)
            # print(done_n)
            reward_n = [np.array(reward) for reward in reward_n]  # list of float -> list of np.ndarray

            cur_step += 1
            obs_past_n = obs_n[:]
            action_past_n = action_n[:]
            reward_past_n = reward_n[:]
            hidden_past_n = hidden_n[:]
            obs_n = obs_next_n[:]

            if (cur_step == max_step) or all(done_n):
                obs_past_n = [np.zeros((obs_dim,)) for _ in range(n)]
                action_past_n = [np.zeros((action_dim,)) for _ in range(n)]
                reward_past_n = [np.zeros((1,)) for _ in range(n)]
                hidden_past_n = [np.zeros((hidden_dim_en,)) for _ in range(n)]

                obs_n = env.reset()
                action_n = [np.zeros((action_dim,)) for _ in range(n)]
                reward_n = [np.zeros((1,)) for _ in range(n)]
                hidden_n = [np.zeros((hidden_dim_en,)) for _ in range(n)]

                obs_next_n = [np.zeros((obs_dim,)) for _ in range(n)]



