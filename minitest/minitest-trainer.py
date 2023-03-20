# the test code for pearl.train.py

from mpe.lib4occupy import make_env
from pearl.marl.maddpg import MADDPGAgent
from pearl.sampler import rollout, sample_batch

import numpy as np
import torch
import torch.nn.functional as F

# region S1. 初始化参数
num_train_tasks = 1
num_agents = 3
num_landmarks = 3
latent_dim = 5  # z的维度
hidden_dim_act = 8
hidden_dim_critic = 8
hidden_dim_en = 8
max_step = 200  # 每个episode的最大step数
device = torch.device(torch.device("cuda"))
max_samples = 600  # 采样的最大样本数

kl_lambda = 0.1
batch_size = 10
gamma = 0.99
# endregion


# region S2. 初始化env, agents, sampler
env = make_env(scenario_name="occupy", num=num_agents)
obs_dim = env.observation_space[0].shape[0]  # obs_dim: 2 * 3 + 2 = 8
act_dim = env.action_space[0].n
agents = [MADDPGAgent(num_agents=num_agents,
                      agent_id=i,
                      obs_dim=obs_dim,
                      action_dim=act_dim,
                      latent_dim=latent_dim,
                      hidden_dim_act=8,
                      hidden_dim_critic=16,
                      hidden_dim_en=8,
                      max_buffer_size=10000,
                      device=device) for i in range(num_agents)]

# endregion

# region S3. 采样max_samples个样本并压入buffer
samples_n = rollout(env, agents, max_step=max_step)

for i in range(env.n):
    agents[i].buffer.add_traj(samples_n[i])

for i in range(env.n):
    print("agent%d: " % i + str(agents[i].buffer._size) + '/' + str(agents[i].buffer._max_size))
# endregion




trans_n = sample_batch(agents, batch_size=batch_size)
# trans_n: List[Dict[str, np.ndarray]]
# len(List) = num_agents; len(Dict) = 8; np.ndarray.shape = (batch_size, obs_dim)

for trans in trans_n:
    for key in trans.keys():
        trans[key] = torch.tensor(trans[key], dtype=torch.float32, device=device)
        # trans[key].shape = (batch_size, xxx_dim or 1)

agent_id = 0
agent = agents[agent_id]
# region S1. encoder_loss的kl散度部分的误差反传, 媒介是latent,
# task_encoder_optimizer.zero_grad(), encoder_loss.backward(), 不要进行encoder的step
# S1需要计算主encoder的kl_div, 主encoder的输出latent, 目标encoder的输出target_latent
kl_div, latent = agent.task_encoder.get_kl_div(obs=trans_n[agent_id]["obs_past"],
                                               action=trans_n[agent_id]["action_past"],
                                               reward=trans_n[agent_id]["reward_past"],
                                               hidden=trans_n[agent_id]["hidden_past"])
target_latent, _ = agent.target_task_encoder.get_post_latent(obs=trans_n[agent_id]["obs_past"],
                                                             action=trans_n[agent_id]["action_past"],
                                                             reward=trans_n[agent_id]["reward_past"],
                                                             hidden=trans_n[agent_id]["hidden_past"])
encoder_loss = kl_div * kl_lambda
agent.task_encoder_optimizer.zero_grad()
encoder_loss.backward(retain_graph=True)
# endregion

# region S2. 计算critic_loss, target_latent计算target_q(无梯度), latent计算q_value
# 注意计算target_q的过程中使用的latent是target_encoder的输出
# 计算q_next的latent的主encoder的输出
# critic_optimizer.zero_grad(), critic_loss.backward(), critic_optimizer.step()
with torch.no_grad():
    action_next_n = [torch.zeros(batch_size, act_dim, device=device) for _ in range(env.n)]
    for j, agent_j in enumerate(agents):
        action_next_n[j] = agent.target_actor.forward(torch.cat([trans_n[j]["obs"], target_latent], dim=1))
    action_next_n.insert(0, action_next_n.pop(agent_id))

    obs_next_n = [trans_n[j]["obs_next"] for j in range(env.n)]
    obs_next_n.insert(0, obs_next_n.pop(agent_id))

    q_next = agent.target_critic.forward(torch.cat([torch.cat(obs_next_n, dim=1),
                                                    torch.cat(action_next_n, dim=1),
                                                    target_latent], dim=1)).detach()

    target_q = (trans_n[agent_id]["reward"] + (1 - trans_n[agent_id]["done"]) * gamma * q_next).detach()

obs_n_critic = [trans_n[j]["obs"] for j in range(env.n)]
obs_n_critic.insert(0, obs_n_critic.pop(agent_id))

action_n_critic = [trans_n[j]["action"] for j in range(env.n)]
action_n_critic.insert(0, action_n_critic.pop(agent_id))
q_value = agent.critic.forward(torch.cat([torch.cat(obs_n_critic, dim=1),
                                          torch.cat(action_n_critic, dim=1),
                                          latent], dim=1))
critic_loss = F.mse_loss(q_value, target_q)
agent.critic_optimizer.zero_grad()
critic_loss.backward(retain_graph=True)
agent.critic_optimizer.step()
# endregion

# region S3. 计算actor_loss, 选择action时要用主encoder, 评价action时要用target_encoder
# actor_optimizer.zero_grad(), actor_loss.backward(), actor_optimizer.step(), encoder_optimizer.step()

obs_n_actor = [trans_n[j]["obs"] for j in range(env.n)]
obs_n_actor.insert(0, obs_n_actor.pop(agent_id))

action_n_actor = [trans_n[j]["action"] for j in range(env.n)]
action_n_actor[agent_id] = agent.actor.forward(torch.cat([trans_n[agent_id]["obs"], latent], dim=1))
action_n_actor.insert(0, action_n_actor.pop(agent_id))

actor_loss = - agent.critic.forward(torch.cat([torch.cat(obs_n_actor, dim=1),
                                               torch.cat(action_n_actor, dim=1),
                                               target_latent], dim=1)).mean()
agent.actor_optimizer.zero_grad()
actor_loss.backward()
agent.actor_optimizer.step()
agent.task_encoder_optimizer.step()
# endregion








