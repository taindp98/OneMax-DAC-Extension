from onemax_dac.configs import parse_args
from onemax_dac.dac import (
    QNetwork,
    Agent,
    ReplayBuffer,
    Logger,
    get_time_str,
    seed_everything,
    rename_state_dict,
)
from onemax_dac.theory_env import OneMax
from torch import nn
import numpy as np
import os
import yaml
import torch
from onemax_dac.dac.trainer import OneMaxDAC


def main():
    training_config, policy_config, env_config = parse_args()

    global_rng = np.random.default_rng(training_config.seed)

    onemax_env = OneMax(
        n=env_config.problem_size,
        state_dim=env_config.state_dim,
        action_choices=env_config.action_choices,
        reward_choice=env_config.reward_choice,
        init_obj_rate=env_config.init_obj_rate,
        rng=global_rng,
    )
    q_net = QNetwork(
        state_dim=onemax_env.state_dim,
        action_dim=onemax_env.action_dim,
        net_arch=policy_config.net_arch,
        activation_fn=getattr(nn, policy_config.activation_fn),
    )
    q_net.load_state_dict(rename_state_dict(torch.load(training_config.eval_fpath)))

    seed_everything(training_config.seed)


if __name__ == "__main__":
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_default_dtype(torch.float32)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    main()
