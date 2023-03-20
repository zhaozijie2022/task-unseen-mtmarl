from typing import Dict, List, Tuple
import torch
import numpy as np


def rollout(env, agents, max_step) -> List[Dict[str, np.ndarray]]:
    """获得1条轨迹"""
    n = env.n
    obs_dim = agents[0].obs_dim
    action_dim = agents[0].action_dim
    hidden_dim_en = agents[0].hidden_dim_en
    max_step += 1
    # region S0.1 初始化traj_xxx_n: List[List[np.ndarray]] (用于存储历史数据)
    traj_obs_past_n = [[] for _ in range(n)]
    traj_action_past_n = [[] for _ in range(n)]
    traj_reward_past_n = [[] for _ in range(n)]
    traj_hidden_past_n = [[] for _ in range(n)]

    traj_obs_n = [[] for _ in range(n)]
    traj_action_n = [[] for _ in range(n)]
    traj_hidden_n = [[] for _ in range(n)]
    traj_reward_n = [[] for _ in range(n)]

    traj_obs_next_n = [[] for _ in range(n)]
    traj_done_n = [[] for _ in range(n)]
    # endregion

    # region S0.2 初始化xxx_n: List[np.ndarray] (用于存储当前时刻的数据)
    obs_past_n = [np.zeros((obs_dim,)) for _ in range(n)]
    action_past_n = [np.zeros((action_dim,)) for _ in range(n)]
    reward_past_n = [np.zeros((1,)) for _ in range(n)]
    hidden_past_n = [np.zeros((hidden_dim_en,)) for _ in range(n)]

    obs_n = env.reset()
    action_n = [np.zeros((action_dim,)) for _ in range(n)]
    reward_n = [np.zeros((1,)) for _ in range(n)]
    hidden_n = [np.zeros((hidden_dim_en,)) for _ in range(n)]

    obs_next_n = [np.zeros((obs_dim,)) for _ in range(n)]
    done_n = [False for _ in range(n)]
    # endregion

    cur_step = 0
    while not (all(done_n) or cur_step >= max_step):
        # region S1.1 与环境交互, 将数据存储到xxx_n中
        with torch.no_grad():
            for i, agent in enumerate(agents):
                if cur_step == 0:
                    latent, hidden = agent.get_prior_latent(obs=obs_past_n[i],
                                                            hidden=hidden_past_n[i])
                else:
                    latent, hidden = agent.get_post_latent(obs=obs_past_n[i],
                                                           action=action_past_n[i],
                                                           reward=reward_past_n[i],
                                                           hidden=hidden_past_n[i])
                # latent = latent.cpu().numpy().squeeze()  # (1, latent_dim) -> (latent_dim,)
                action = agent.get_action(obs=obs_n[i], latent=latent)
                action_n[i] = action.cpu().numpy().squeeze()  # (1, action_dim) -> (action_dim,)
                hidden_n[i] = hidden.cpu().numpy().squeeze()  # (1, hidden_dim) -> (hidden_dim,)

        obs_next_n, reward_n, done_n, info_n = env.step(action_n)
        reward_n = [np.array(reward) for reward in reward_n]  # list of float -> list of np.ndarray
        # endregion

        # region S1.2 将当前时刻的数据存储到traj_xxx_n中
        for i in range(n):
            traj_obs_past_n[i].append(obs_past_n[i])
            traj_action_past_n[i].append(action_past_n[i])
            traj_reward_past_n[i].append(reward_past_n[i])
            traj_hidden_past_n[i].append(hidden_past_n[i])

            traj_obs_n[i].append(obs_n[i])
            traj_action_n[i].append(action_n[i])
            traj_reward_n[i].append(reward_n[i])
            traj_hidden_n[i].append(hidden_n[i])

            traj_obs_next_n[i].append(obs_next_n[i])
            traj_done_n[i].append(done_n[i])
        # endregion

        # region S1.3 更新xxx_n与xxx_past_n
        cur_step += 1
        obs_past_n = obs_n[:]
        action_past_n = action_n[:]
        reward_past_n = reward_n[:]
        hidden_past_n = hidden_n[:]
        obs_n = obs_next_n[:]
        # endregion
        # print(cur_step)

    # region S2.1 将traj_xxx_n中的数据转换为np.ndarray
    traj_n = []
    for i in range(n):
        traj_n.append(dict(
            obs_past=np.array(traj_obs_past_n[i][1:]),
            action_past=np.array(traj_action_past_n[i][1:]),
            reward_past=np.array(traj_reward_past_n[i][1:]),
            hidden_past=np.array(traj_hidden_past_n[i][1:]),

            obs=np.array(traj_obs_n[i][1:]),
            action=np.array(traj_action_n[i][1:]),
            reward=np.array(traj_reward_n[i][1:]),
            hidden=np.array(traj_hidden_n[i][1:]),

            obs_next=np.array(traj_obs_next_n[i][1:]),
            done=np.array(traj_done_n[i][1:]),
        ))
    # endregion
    return traj_n


def sample_batch(agents, batch_size):
    buffer_size = agents[0].buffer.size
    # 理论上, agents.buffer[same_task_id]._size应该相等
    indices = np.random.randint(0, buffer_size, batch_size)
    samples_n = []
    for agent in agents:
        samples_n.append(agent.buffer.sample_batch(indices))
    # 返回list of dict, 类型和obtain_samples返回的trajs_n相同
    return samples_n

# region class Sampler 废稿
# class Sampler:
#     # 所有agent共用一个Sampler
#     def __init__(self, env, agents, max_step, device) -> None:
#         self.env = env  # list of gym.Env
#         self.agents: List[MADDPGAgent] = agents  # list of MARL Agent, attr: a-c net, ReplayBuffer
#         self.max_step = max_step + 1  # max step of 1 episode
#         self.device = device
#         self.n = env.n  # number of agents
#
#         self.obs_dim = env.observation_space[0].shape[0]  # obs_dim of each agent
#         self.action_dim = env.action_space[0].n  # action_dim of each agent
#         self.hidden_dim_en = agents[0].hidden_dim_en  # hidden_dim of each agent
#         self.latent_dim = agents[0].latent_dim  # latent_dim of each agent
#
#     def obtain_samples(self, max_samples):
#         # 采样max_samples个transitions,
#         trajs_n = [{
#             "obs_past": [], "action_past": [], "reward_past": [], "hidden_past": [],
#             "obs": [], "action": [], "reward": [], "hidden": [],
#             "obs_next": [], "done": []
#         } for _ in range(self.n)]
#
#         _next = 0
#
#         while _next < max_samples:
#             traj_n = self.rollout()
#             # traj_n[i]是第i个agent的traj, 是一个dict, traj_n[i]['obs']是一个list of array, 每个array是一个episode上的obs轨迹
#             for i in range(self.n):
#                 for key in trajs_n[i].keys():
#                     trajs_n[i][key].append(traj_n[i][key])
#             _next += len(traj_n[0]["obs"])
#
#         # print(_next)
#
#         for i in range(self.env.n):
#             for key in trajs_n[i].keys():
#                 trajs_n[i][key] = np.concatenate(trajs_n[i][key], axis=0)
#
#         return _next, trajs_n
#
#     def rollout(self) -> List[Dict[str, np.ndarray]]:
#         # rollout用于获得一个episode上的traj (到done或max_step)
#         # traj_n: List[Dict[str, np.ndarray]], len = num_agents
#         # traj_n[i] = {"xxx": np.ndarray}, np.ndarray.shape := max_step (or others) * xxx_dim
#
#         # region 初始化traj_xxx_n: List[List[np.ndarray]] (用于存储历史数据)
#         traj_obs_past_n = [[] for _ in range(self.n)]
#         traj_action_past_n = [[] for _ in range(self.n)]
#         traj_reward_past_n = [[] for _ in range(self.n)]
#         traj_hidden_past_n = [[] for _ in range(self.n)]
#
#         traj_obs_n = [[] for _ in range(self.n)]
#         traj_action_n = [[] for _ in range(self.n)]
#         traj_hidden_n = [[] for _ in range(self.n)]
#         traj_reward_n = [[] for _ in range(self.n)]
#
#         traj_obs_next_n = [[] for _ in range(self.n)]
#         traj_done_n = [[] for _ in range(self.n)]
#         # endregion
#
#         # region 初始化xxx_n: List[np.ndarray] (用于存储当前时刻的数据)
#         obs_past_n = [np.zeros((self.obs_dim,)) for _ in range(self.n)]
#         action_past_n = [np.zeros((self.action_dim,)) for _ in range(self.n)]
#         reward_past_n = [np.zeros((1,)) for _ in range(self.n)]
#         hidden_past_n = [np.zeros((self.hidden_dim_en,)) for _ in range(self.n)]
#
#         obs_n = self.env.reset()
#         action_n = [np.zeros((self.action_dim,)) for _ in range(self.n)]
#         reward_n = [np.zeros((1,)) for _ in range(self.n)]
#         hidden_n = [np.zeros((self.hidden_dim_en,)) for _ in range(self.n)]
#
#         obs_next_n = [np.zeros((self.obs_dim,)) for _ in range(self.n)]
#         done_n = [False for _ in range(self.n)]
#         # endregion
#
#         cur_step = 0
#         while not (all(done_n) or cur_step >= self.max_step):
#             with torch.no_grad():
#                 for i, agent in enumerate(self.agents):
#                     if cur_step == 0:
#                         latent, hidden = agent.get_prior_latent(obs=obs_n[i],
#                                                                 hidden=hidden_n[i])
#                     else:
#                         latent, hidden = agent.get_post_latent(obs=obs_past_n[i],
#                                                                action=action_past_n[i],
#                                                                reward=reward_past_n[i],
#                                                                hidden=hidden_past_n[i])
#                     # latent = latent.cpu().numpy().squeeze()  # (1, latent_dim) -> (latent_dim,)
#                     action = agent.get_action(obs=obs_n[i], latent=latent)
#                     action_n[i] = action.cpu().numpy().squeeze()  # (1, action_dim) -> (action_dim,)
#                     hidden_n[i] = hidden.cpu().numpy().squeeze()  # (1, hidden_dim) -> (hidden_dim,)
#
#             obs_next_n, reward_n, done_n, info_n = self.env.step(action_n)
#             reward_n = [np.array(reward) for reward in reward_n]  # list of float -> list of np.ndarray
#
#             for i in range(self.n):
#                 traj_obs_past_n[i].append(obs_past_n[i])
#                 traj_action_past_n[i].append(action_past_n[i])
#                 traj_reward_past_n[i].append(reward_past_n[i])
#                 traj_hidden_past_n[i].append(hidden_past_n[i])
#
#                 traj_obs_n[i].append(obs_n[i])
#                 traj_action_n[i].append(action_n[i])
#                 traj_reward_n[i].append(reward_n[i])
#                 traj_hidden_n[i].append(hidden_n[i])
#
#                 traj_obs_next_n[i].append(obs_next_n[i])
#                 traj_done_n[i].append(done_n[i])
#
#             cur_step += 1
#             obs_past_n = obs_n
#             action_past_n = action_n
#             reward_past_n = reward_n
#             hidden_past_n = hidden_n
#             # print(cur_step)
#
#         traj_n = []
#         for i in range(self.n):
#             traj_n.append(dict(
#                 obs_past=np.array(traj_obs_past_n[i][1:]),
#                 action_past=np.array(traj_action_past_n[i][1:]),
#                 reward_past=np.array(traj_reward_past_n[i][1:]),
#                 hidden_past=np.array(traj_hidden_past_n[i][1:]),
#
#                 obs=np.array(traj_obs_n[i]),
#                 action=np.array(traj_action_n[i]),
#                 reward=np.array(traj_reward_n[i]),
#                 hidden=np.array(traj_hidden_n[i]),
#
#                 obs_next=np.array(traj_obs_next_n[i]),
#                 done=np.array(traj_done_n[i]),
#             ))
#         return traj_n
#
#     def sample_batch(self, batch_size):
#         # 同步地在每个agent的ReplayBuffer中采样batch_size个transitions
#         buffer_size = self.agents[0].buffer._size
#         # 理论上, agents.buffer[same_task_id]._size应该相等
#         indices = np.random.randint(0, buffer_size, batch_size)
#         sample_n = []
#         for agent in self.agents:
#             sample_n.append(agent.buffer.sample_batch(indices))
#         # 返回list of dict, 类型和obtain_samples返回的trajs_n相同
#         return sample_n
# endregion


