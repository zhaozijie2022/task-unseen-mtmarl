import os
import time
from tqdm import tqdm
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


from common.utils import create_dir
from mpe.lib4occupy import make_env
from marl.maddpg import MADDPGAgent
from common.sampler import rollout, sample_batch



# region S1. 初始化参数
# 1.环境相关 --make_env
num_agents = 2
# num_landmarks = 3
scenario_name = "occupy"

# 2.Agent相关 --MADDPGAgent
latent_dim = 4  # z的维度
hidden_dim_act = 8
hidden_dim_critic = 16
hidden_dim_en = 8
device = torch.device(torch.device("cpu"))
max_buffer_size = 500000  # 500k
actor_lr = 1e-4
critic_lr = 1e-3
encoder_lr = 5e-4
kl_lambda = 0.1

# 3.采样相关 --Sampler
max_step = 50  # 每个episode的最大step数

# 4.训练相关 -- Trainer
gamma = 0.99
batch_size = 1024
num_episodes = 60000  # 本次训练采集多少episode
# train_rate = 1  # 每隔多少个episode训练一次
print_rate = 100
save_rate = 1000
plot_rate = 5000
save_buffer_rate = 10000

# 5.实验相关 -- Experiment
# load_policy_path = "./results/"
# load_buffer_path = "./results/"

# endregion

if __name__ == '__main__':
    save_models_path, save_figures_path, save_buffers_path = create_dir(scenario_name)
    # region S2. 初始化env, agents
    env = make_env(scenario_name=scenario_name, num=num_agents)
    obs_dim = env.observation_space[0].shape[0]  # obs_dim: 2 * 3 + 2 = 8
    action_dim = env.action_space[0].n
    agents = [MADDPGAgent(num_agents=num_agents,
                          agent_id=i,
                          obs_dim=obs_dim,
                          action_dim=action_dim,
                          latent_dim=latent_dim,
                          hidden_dim_act=8,
                          hidden_dim_critic=16,
                          hidden_dim_en=8,
                          max_buffer_size=max_buffer_size,
                          device=device,
                          gamma=gamma,
                          actor_lr=actor_lr,
                          critic_lr=critic_lr,
                          encoder_lr=encoder_lr,
                          kl_lambda=kl_lambda,
                          batch_size=batch_size,
                          ) for i in range(num_agents)]
    # endregion

    episode_rewards = []  # 本次训练每个episode的reward
    episode_rps = []  # 本次训练每个episode的reward per step
    cur_step = 0
    # cur_episode = 0
    t_start = time.time()

    print("Start training...")
    # while cur_episode < num_episodes:
    for cur_episode in tqdm(range(num_episodes)):
        traj_n = rollout(env=env, agents=agents, max_step=max_step)

        for i in range(env.n):
            agents[i].buffer.add_traj(traj_n[i])

        cur_step += len(traj_n[0]["obs"])
        # cur_episode += 1
        episode_rewards.append(traj_n[0]["reward"].sum())
        episode_rps.append(traj_n[0]["reward"].sum() / len(traj_n[0]["obs"]))

        for agent_id, agent in enumerate(agents):
            agent.train(agents, cur_step)

        if (cur_episode + 1) % print_rate == 0:
            print("\nEps: %d, Sps: %.2fk, Reward: %.2f, RpS: %.2f, Time: %.2f"
                  % (cur_episode + 1,
                     round(cur_step / 1000, 2),
                     float(np.mean(episode_rewards[-print_rate:])),
                     float(np.mean(episode_rps[-print_rate:])),
                     time.time() - t_start))
            t_start = time.time()

        if (cur_episode + 1) % save_rate == 0:
            for i in range(env.n):
                agents[i].save_model(save_models_path)

        if (cur_episode + 1) % save_buffer_rate == 0:
            for i in range(env.n):
                agents[i].save_buffer(save_buffers_path)







