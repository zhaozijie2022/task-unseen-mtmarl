
# 数据输入输出networks.py都是tensor.float32, 不设计和numpy的转换
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_hidden_layers: int = 2,
        hidden_activation: torch.nn.functional = F.relu,
        init_w: float = 3e-3,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_activation = hidden_activation

        # Set fully connected layers
        self.fc_layers = nn.ModuleList()
        self.hidden_layers = [hidden_dim] * num_hidden_layers  # 此时的hidden_layers只是一个int列表
        in_layer = input_dim

        for i, hidden_layer in enumerate(self.hidden_layers):
            fc_layer = nn.Linear(in_layer, hidden_layer)
            in_layer = hidden_layer  # 上一层的输出维度就是下一层的输入维度
            self.__setattr__("fc_layer{}".format(i), fc_layer)  # 创建一个名为fc_layeri的属性, 其值为nn.Linear
            self.fc_layers.append(fc_layer)  # 此时的self.fc_layers已装填好nn.Linear

        # 定义输出层
        self.last_fc_layer = nn.Linear(hidden_dim, output_dim)
        self.last_fc_layer.weight.data.uniform_(-init_w, init_w)
        self.last_fc_layer.bias.data.uniform_(-init_w, init_w)

    def forward(self, x):
        for fc_layer in self.fc_layers:
            x = self.hidden_activation(fc_layer(x))
        x = self.last_fc_layer(x)
        return x


class SequentialRNN(nn.Module):
    def __init__(self, input_dim,  output_dim,  hidden_dim,
                 hidden_activation: torch.nn.functional = F.relu,
                 num_hidden_layers=2,init_w=3e-3):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.hidden_activation = hidden_activation

        # 若隐层数量大于1, 则1个在RNN前, 其余在后
        self.input_fc = nn.Linear(input_dim, hidden_dim)
        if self.num_hidden_layers > 1:
            self.__setattr__("input_fc", self.input_fc)
            self.gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=1)
            self.output_fc = nn.ModuleList()
            for i in range(num_hidden_layers - 1):
                fc_layer = nn.Linear(hidden_dim, hidden_dim)
                self.__setattr__("output_fc{}".format(i), fc_layer)
                self.output_fc.append(fc_layer)
        else:
            self.gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=1)

        self.last_fc_layer = nn.Linear(hidden_dim, output_dim)
        self.last_fc_layer.weight.data.uniform_(-init_w, init_w)
        self.last_fc_layer.bias.data.uniform_(-init_w, init_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape = (batch_size, seq_len, input_dim)
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        x = x.view(batch_size * seq_len, -1)
        # 将batch_size和seq_len合并, 使得x的shape为(batch_size * seq_len, input_dim)
        x = self.hidden_activation(self.input_fc(x))

        x = x.view(seq_len, batch_size, -1)
        x, _ = self.gru(x)
        x = x[-1, :, :]  # 取最后一个时刻的输出

        if self.num_hidden_layers > 1:
            for fc_layer in self.output_fc:
                x = self.hidden_activation(fc_layer(x))

        x = self.last_fc_layer(x)
        return x


class HiddenRNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        hidden_activation: torch.nn.functional = F.relu,
        num_hidden_layers: int = 2,
        init_w: float = 3e-3,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.hidden_activation = hidden_activation

        # 若隐层数量大于1, 则1个在RNN前, 其余在后
        self.input_fc = nn.Linear(input_dim, hidden_dim)
        if self.num_hidden_layers > 1:
            self.__setattr__("input_fc", self.input_fc)
            self.gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=1)
            self.output_fc = nn.ModuleList()
            for i in range(num_hidden_layers - 1):
                fc_layer = nn.Linear(hidden_dim, hidden_dim)
                self.__setattr__("output_fc{}".format(i), fc_layer)
                self.output_fc.append(fc_layer)
        else:
            self.gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=1)

        self.last_fc_layer = nn.Linear(hidden_dim, output_dim)
        self.last_fc_layer.weight.data.uniform_(-init_w, init_w)
        self.last_fc_layer.bias.data.uniform_(-init_w, init_w)

    def forward(self, x, h):
        x = self.hidden_activation(self.input_fc(x))
        x, h = x.unsqueeze(0), h.unsqueeze(0)
        x, h = self.gru(x, h)
        x, h = x.squeeze(0), h.squeeze(0)
        if self.num_hidden_layers > 1:
            for fc_layer in self.output_fc:
                x = self.hidden_activation(fc_layer(x))
        x = self.last_fc_layer(x)
        return x, h


class VaeEncoder(HiddenRNN):
    # 所有任务共享同一个encoder, 输入trans和hidden, 给出z_mean和z_var
    def __init__(self,
                 obs_dim,
                 action_dim,
                 hidden_dim_en,
                 latent_dim,
                 device):
        super().__init__(input_dim=obs_dim + action_dim + 1,
                         output_dim=latent_dim * 2,
                         hidden_dim=hidden_dim_en)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim_en = hidden_dim_en
        self.latent_dim = latent_dim
        self.device = device

    def get_post_latent(self, obs, hidden, action, reward):
        """更新z的后验分布, 并采样得到z, 返回sampled_z和hidden"""
        transition = torch.cat([obs, action, reward], dim=1)  # all data.shape = (batch_size, dim)

        posterior_z, hidden = self.forward(transition, hidden)  # shape: (-1, self.latent_dim * 2)
        z_mean = torch.unbind(posterior_z[:, :self.latent_dim])
        z_var = torch.unbind(F.softplus(posterior_z[:, self.latent_dim:]))
        # torch.unbind将tensor按照dim=0拆分成多个tensor, 即每行一个z_mean和z_var
        dists = []
        for mean, var in zip(z_mean, z_var):
            dist = torch.distributions.Normal(mean, torch.sqrt(var))
            dists.append(dist)
        latent = [dist.rsample() for dist in dists]
        return torch.stack(latent).to(self.device), hidden

    def get_prior_latent(self, obs, hidden):
        """返回先验分布的z, 不改变hidden"""
        dist = torch.distributions.Normal(torch.zeros(self.latent_dim), torch.ones(self.latent_dim))
        latent = [dist.rsample() for _ in range(obs.shape[0])]
        return torch.stack(latent).to(self.device), hidden

    def get_kl_div(self, obs, hidden, action, reward) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回KL散度"""
        transition = torch.cat([obs, action, reward], dim=1)
        posterior_z, _ = self.forward(transition, hidden)
        z_mean = torch.unbind(posterior_z[:, :self.latent_dim])
        z_var = torch.unbind(F.softplus(posterior_z[:, self.latent_dim:]))
        prior = torch.distributions.Normal(
            torch.zeros(self.latent_dim).to(self.device),
            torch.ones(self.latent_dim).to(self.device),
        )
        posteriors = []
        for mean, var in zip(z_mean, z_var):
            dist = torch.distributions.Normal(mean, torch.sqrt(var))
            posteriors.append(dist)

        kl_div = [torch.distributions.kl.kl_divergence(posterior, prior) for posterior in posteriors]
        latent = [dist.rsample() for dist in posteriors]
        return torch.stack(kl_div).sum().to(self.device), torch.stack(latent).to(self.device)























