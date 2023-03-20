import torch
import numpy as np
import time
import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
# 因为使用了imp, 在python3.4中就被废弃了, 隐藏warning


def make_env(scenario_name, num=3):
    """环境部分"""
    from mpe.environment import MultiAgentEnv
    import mpe.scenarios as scenarios
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world(num=num)
    # create multiagent environment
    env = MultiAgentEnv(world=world,
                        reset_callback=scenario.reset_world,
                        reward_callback=scenario.reward,
                        observation_callback=scenario.observation,
                        done_callback=scenario.done)
    return env


# region
def parse_args_exp(exp_name="mt-occupy",
                   num_train_tasks=100,
                   num_test_tasks=30,
                   latent_dim=5):
    """实验参数, experiment, pearl, sac"""
    import argparse
    parser = argparse.ArgumentParser("The parameters for the experiment")
    args = parser.parse_args()
    args.exp_name = exp_name
    args.num_train_tasks = num_train_tasks
    args.num_test_tasks = num_test_tasks
    args.latent_dim = latent_dim
    return args


def parse_args_algo(num_epochs=1000,
                    num_sample_tasks=5,
                    num_init_samples=2000,
                    num_prior_samples=400,
                    num_posterior_samples=600,
                    num_meta_grads=1500,
                    meta_batch_size=16,
                    batch_size=100,
                    max_buffer_size=100000):
    import argparse
    parser = argparse.ArgumentParser("The parameters for Algorithm: PEARL")
    args = parser.parse_args()
    args.num_epochs = num_epochs
    args.num_sample_tasks = num_sample_tasks
    args.num_init_samples = num_init_samples
    args.num_prior_samples = num_prior_samples
    args.num_posterior_samples = num_posterior_samples
    args.num_meta_grads = num_meta_grads
    args.meta_batch_size = meta_batch_size
    args.batch_size = batch_size
    args.max_buffer_size = max_buffer_size
    return args


def parse_args_net(gamma=0.99,
                   kl_lambda=0.1,
                   batch_size=256,
                   qf_lr=3e-4,
                   policy_lr=3e-4,
                   vf_lr=3e-4,):
    import argparse
    parser = argparse.ArgumentParser("The parameters for Agent: MARL")
    args = parser.parse_args()
    args.gamma = gamma
    args.kl_lambda = kl_lambda
    args.batch_size = batch_size
    args.qf_lr = qf_lr
    args.policy_lr = policy_lr
    args.vf_lr = vf_lr
    return args
# endregion










