from typing import List, Dict, Tuple, Any
# 在marl中进行numpy与tensor的转换, 需要标注下数据类型
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle

import os

from common.networks import MLP, VaeEncoder
from common.buffer import ReplayBuffer
from common.sampler import sample_batch


class Actor(MLP):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, num_hidden_layers: int):
        super(Actor, self).__init__(input_dim, output_dim, hidden_dim, num_hidden_layers)

    def forward(self, x):
        return F.tanh(super(Actor, self).forward(x))

class MADDPGAgent:
    def __init__(
            self,
            num_agents: int,
            agent_id: int,
            obs_dim: int,
            action_dim: int,
            latent_dim: int,
            hidden_dim_act: int,
            hidden_dim_critic: int,
            hidden_dim_en: int,
            device: torch.device,
            max_buffer_size=100000,
            gamma: float = 0.99,
            actor_lr: float = 3e-4,
            critic_lr: float = 3e-4,
            encoder_lr: float = 3e-4,
            kl_lambda: float = 0.1,
            batch_size: int = 256,
            tau: float = 0.01,
    ):
        self.num_agents = num_agents
        self.agent_id = agent_id
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.hidden_dim_act = hidden_dim_act
        self.hidden_dim_critic = hidden_dim_critic
        self.hidden_dim_en = hidden_dim_en
        self.device = device
        self.gamma = gamma
        self.max_buffer_size = max_buffer_size
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.encoder_lr = encoder_lr
        self.kl_lambda = kl_lambda
        self.batch_size = batch_size
        self.tau = tau
        self._train_epochs = 0  # 截止现在, 训练了多少次

        self.actor = Actor(input_dim=obs_dim + latent_dim,
                         output_dim=action_dim,
                         hidden_dim=hidden_dim_act,
                         num_hidden_layers=2).to(device)
        self.critic = MLP(input_dim=(obs_dim + action_dim) * num_agents + latent_dim,
                          output_dim=1,
                          hidden_dim=hidden_dim_critic,
                          num_hidden_layers=2).to(device)
        self.target_actor = Actor(input_dim=obs_dim + latent_dim,
                                output_dim=action_dim,
                                hidden_dim=hidden_dim_act,
                                num_hidden_layers=2).to(device)
        self.target_critic = MLP(input_dim=(obs_dim + action_dim) * num_agents + latent_dim,
                                 output_dim=1,
                                 hidden_dim=hidden_dim_critic,
                                 num_hidden_layers=2).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        self.task_encoder = VaeEncoder(obs_dim=obs_dim,
                                       action_dim=action_dim,
                                       hidden_dim_en=hidden_dim_en,
                                       latent_dim=latent_dim,
                                       device=device).to(device)
        self.target_task_encoder = VaeEncoder(obs_dim=obs_dim,
                                              action_dim=action_dim,
                                              hidden_dim_en=hidden_dim_en,
                                              latent_dim=latent_dim,
                                              device=device).to(device)
        self.task_encoder_optimizer = torch.optim.Adam(self.task_encoder.parameters(), lr=self.encoder_lr)

        self.buffer = ReplayBuffer(obs_dim=obs_dim,
                                   action_dim=action_dim,
                                   hidden_dim_en=hidden_dim_en,
                                   max_size=max_buffer_size)

    def soft_update_target_networks(self, tau):
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        for target_param, param in zip(self.target_task_encoder.parameters(), self.task_encoder.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def hard_update_target_networks(self):
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_task_encoder.load_state_dict(self.task_encoder.state_dict())

    def get_prior_latent(self,
                         obs: np.ndarray,  # shape: batch_size * obs_dim
                         hidden,  # shape: batch_size * hidden_dim_en
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
        obs = torch.tensor(obs, dtype=torch.float32).view(-1, self.obs_dim).to(self.device)
        hidden = torch.tensor(hidden, dtype=torch.float32).view(-1, self.hidden_dim_en).to(self.device)
        latent, hidden = self.task_encoder.get_prior_latent(obs, hidden)
        return latent, hidden

    def get_post_latent(self,
                        obs: np.ndarray,  # shape: batch_size * obs_dim
                        action: np.ndarray,  # shape: batch_size * action_dim
                        reward,  # shape: batch_size,
                        hidden,  # shape: batch_size * hidden_dim_en
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
        obs = torch.tensor(obs, dtype=torch.float32).view(-1, self.obs_dim).to(self.device)
        action = torch.tensor(action, dtype=torch.float32).view(-1, self.action_dim).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float32).view(-1, 1).to(self.device)
        hidden = torch.tensor(hidden, dtype=torch.float32).view(-1, self.hidden_dim_en).to(self.device)

        latent, hidden = self.task_encoder.get_post_latent(obs, hidden, action, reward)
        return latent, hidden

    def get_action(self,
                   obs: np.ndarray,  # shape: batch_size * obs_dim
                   latent,  # shape: batch_size * latent_dim
                   ) -> torch.Tensor:
        obs = torch.tensor(obs, dtype=torch.float32).view(-1, self.obs_dim).to(self.device)
        # latent = torch.tensor(latent, dtype=torch.float32).view(-1, self.latent_dim).to(self.device)
        action = self.actor.forward(torch.cat([obs, latent], dim=1))
        return action

    def train(self, agents, cur_steps):
        if self.buffer.size < self.buffer.max_size:
            return

        self._train_epochs += 1
        if not cur_steps < 100 * self._train_epochs:
            return

        # region S0. 从buffer中采样
        trans_n = sample_batch(agents=agents, batch_size=self.batch_size)
        # trans_n: List[Dict[str, np.ndarray]]
        # len(List) = num_agents; len(Dict) = 8; np.ndarray.shape = (batch_size, obs_dim)

        for trans in trans_n:
            for key in trans.keys():
                trans[key] = torch.tensor(trans[key], dtype=torch.float32, device=self.device)
                # trans[key].shape = (batch_size, xxx_dim or 1)
        # endregion

        # region S1. encoder_loss的kl散度部分的误差反传, 媒介是latent,
        # task_encoder_optimizer.zero_grad(), encoder_loss = ,  encoder_loss.backward(), 不要进行encoder的step
        # S1需要计算主encoder的kl_div, 主encoder的输出latent, 目标encoder的输出target_latent
        kl_div, latent = self.task_encoder.get_kl_div(obs=trans_n[self.agent_id]["obs_past"],
                                                      action=trans_n[self.agent_id]["action_past"],
                                                      reward=trans_n[self.agent_id]["reward_past"],
                                                      hidden=trans_n[self.agent_id]["hidden_past"])
        with torch.no_grad():
            target_latent, _ = self.target_task_encoder.get_post_latent(obs=trans_n[self.agent_id]["obs_past"],
                                                                        action=trans_n[self.agent_id]["action_past"],
                                                                        reward=trans_n[self.agent_id]["reward_past"],
                                                                        hidden=trans_n[self.agent_id]["hidden_past"])
        encoder_loss = kl_div * self.kl_lambda
        self.task_encoder_optimizer.zero_grad()
        encoder_loss.backward(retain_graph=True)
        # endregion

        # region S3. 计算actor_loss, 选择action时要用主encoder, 评价action时要用target_encoder
        # actor_optimizer.zero_grad(), actor_loss.backward(), actor_optimizer.step(), encoder_optimizer.step()

        obs_n_actor = [trans_n[j]["obs"] for j in range(self.num_agents)]
        obs_n_actor.insert(0, obs_n_actor.pop(self.agent_id))

        action_n_actor = [trans_n[j]["action"] for j in range(self.num_agents)]
        action_n_actor[self.agent_id] = self.actor.forward(torch.cat([trans_n[self.agent_id]["obs"], latent], dim=1))
        action_n_actor.insert(0, action_n_actor.pop(self.agent_id))

        actor_loss = - self.critic.forward(torch.cat([torch.cat(obs_n_actor, dim=1),
                                                      torch.cat(action_n_actor, dim=1),
                                                      target_latent], dim=1)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_optimizer.step()
        # endregion

        # region S2. 计算critic_loss, target_latent计算target_q(无梯度), latent计算q_value
        # 注意计算target_q的过程中使用的latent是target_encoder的输出
        # 计算q_next的latent的主encoder的输出
        # critic_optimizer.zero_grad(), critic_loss.backward(), critic_optimizer.step()
        with torch.no_grad():
            action_next_n = [torch.zeros(self.batch_size, self.action_dim, device=self.device) for _ in range(self.num_agents)]
            for j, agent_j in enumerate(agents):
                action_next_n[j] = self.target_actor.forward(torch.cat([trans_n[j]["obs"], target_latent], dim=1))
            action_next_n.insert(0, action_next_n.pop(self.agent_id))

            obs_next_n = [trans_n[j]["obs_next"] for j in range(self.num_agents)]
            obs_next_n.insert(0, obs_next_n.pop(self.agent_id))

            q_next = self.target_critic.forward(torch.cat([torch.cat(obs_next_n, dim=1),
                                                           torch.cat(action_next_n, dim=1),
                                                           target_latent], dim=1)).detach()

            target_q = (trans_n[self.agent_id]["reward"] +
                        (1 - trans_n[self.agent_id]["done"]) * self.gamma * q_next).detach()

        obs_n_critic = [trans_n[j]["obs"] for j in range(self.num_agents)]
        obs_n_critic.insert(0, obs_n_critic.pop(self.agent_id))

        action_n_critic = [trans_n[j]["action"] for j in range(self.num_agents)]
        action_n_critic.insert(0, action_n_critic.pop(self.agent_id))
        q_value = self.critic.forward(torch.cat([torch.cat(obs_n_critic, dim=1),
                                                 torch.cat(action_n_critic, dim=1),
                                                 latent], dim=1))
        critic_loss = F.mse_loss(q_value, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.task_encoder_optimizer.step()
        # endregion

        # region S4. 更新target_encoder, target_critic
        self.soft_update_target_networks(self.tau)

    def save_model(self, save_path):
        save_path = os.path.join(save_path, "agent_%d" % self.agent_id)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        model_dict = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "task_encoder": self.task_encoder.state_dict(),
            "target_actor": self.target_actor.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "target_task_encoder": self.target_task_encoder.state_dict(),
        }
        for key in model_dict.keys():
            tmp_save_path = os.path.join(save_path, key + ".pth")
            torch.save(model_dict[key], tmp_save_path)

    def load_model(self, load_path):
        load_path = os.path.join(load_path, "agent_%d" % self.agent_id)
        model_dict = {
            "actor": self.actor,
            "critic": self.critic,
            "task_encoder": self.task_encoder,
            "target_actor": self.target_actor,
            "target_critic": self.target_critic,
            "target_task_encoder": self.target_task_encoder,
        }
        for key in model_dict.keys():
            tmp_load_path = os.path.join(load_path, key + ".pth")
            model_dict[key].load_state_dict(torch.load(tmp_load_path, map_location=self.device))

    def save_buffer(self, save_path):
        save_path = os.path.join(save_path, "buffer_" + "agent_%d" % self.agent_id + ".pkl")
        with open(save_path, "wb") as f:
            pickle.dump(self.buffer, f)

    def load_buffer(self, load_path):
        load_path = os.path.join(load_path, "buffer_" + "agent_%d" % self.agent_id + ".pkl")
        with open(load_path, "rb") as f:
            self.buffer = pickle.load(f)

    # def get_q_value(self,
    #                 obs_n: List[np.ndarray],  # len=agent_num, obs_n[i].shape = (batch_size, obs_dim)
    #                 action_n: List[np.ndarray],  # len=agent_num, action_n[i].shape = (batch_size, action_dim)
    #                 latent: np.ndarray  # shape: batch_size * latent_dim
    #                 ) -> torch.Tensor:
    #     obs_n.insert(0, obs_n.pop(self.agent_id))  # 把本agent的obs放到第一个, 其余的相对位置不变
    #     obs_n = np.concatenate(obs_n, axis=1)  # shape: batch_size * (obs_dim * agent_num)
    #
    #     action_n.insert(0, action_n.pop(self.agent_id))  # 把本agent的action放到第一个, 其余的相对位置不变
    #     action_n = np.concatenate(action_n, axis=1)  # shape: batch_size * (action_dim * agent_num)
    #
    #     obs = torch.tensor(obs_n, dtype=torch.float32).view(-1, self.obs_dim * self.num_agents).to(self.device)
    #     action = torch.tensor(action_n, dtype=torch.float32).view(-1, self.action_dim * self.num_agents).to(self.device)
    #     latent = torch.tensor(latent, dtype=torch.float32).view(-1, self.latent_dim).to(self.device)
    #     q_value = self.critic.forward(torch.cat([obs, action, latent], dim=1))
    #     return q_value







