from onemax_dac.configs import parse_args
from onemax_dac.dac import (
    QNetwork,
    Agent,
    ReplayBuffer,
    Logger,
    get_time_str,
    seed_everything,
)
from onemax_dac.envs import OneMax
from torch import nn
import numpy as np
import os
import yaml
import torch
from onemax_dac.dac.trainer import OneMaxDAC

def main():
    # Parse the command-line arguments
    training_config, policy_config, env_config = parse_args()
    
    # Print the configurations to verify
    print(f"Training Config: {training_config}")
    print(f"Policy Config: {policy_config}")
    print(f"Environment Config: {env_config}")

    global_rng = np.random.default_rng(training_config.seed)

    onemax_env = OneMax(
        n=env_config.problem_size,
        state_dim=env_config.state_dim,
        action_choices=env_config.action_choices,
        reward_choice=env_config.reward_choice,
        init_obj_rate=env_config.init_obj_rate,
        rng = global_rng
    )
    q_online = QNetwork(
        state_dim=onemax_env.state_dim,
        action_dim=onemax_env.action_dim,
        net_arch=policy_config.net_arch,
        activation_fn=getattr(nn, policy_config.activation_fn),
    )
    q_target = QNetwork(
        state_dim=onemax_env.state_dim,
        action_dim=onemax_env.action_dim,
        net_arch=policy_config.net_arch,
        activation_fn=getattr(nn, policy_config.activation_fn),
    )
    replay_buffer = ReplayBuffer(
        max_size = training_config.buffer_size,
        rng=global_rng
    )
    agent = Agent(
        env=onemax_env,
        replay_buffer=replay_buffer,
        rng=global_rng
    )
    save_dir = os.path.join(
        training_config.output_dir, "checkpoints", get_time_str(), f"seed_{training_config.seed}"
    )
    os.makedirs(save_dir, exist_ok=True)
    # Dump both configurations
    config_dict = {
        "TrainingConfig": training_config.to_dict(),
        "PolicyConfig": policy_config.to_dict(),
        "EnvConfig": env_config.to_dict(),
    }

    with open(f"{save_dir}/config.yml", "w") as file:
        yaml.dump(config_dict, file, default_flow_style=False)

    logger = Logger(
        config={
            "training_config": training_config,
            "policy_config": policy_config,
            "env_config": env_config,
        },
        use_wandb=training_config.wandb,
        save_dir=save_dir,
    )
    trainer = OneMaxDAC(
        agent=agent,
        q_online = q_online,
        q_target = q_target,
        logger=logger,
        rng = global_rng,
        ckpt_dir=save_dir,
        training_config=training_config
    )
    # Here you can continue with the logic to initialize the agent and start training
    # agent = RLAgent(training_config, policy_config, env_config)
    seed_everything(training_config.seed)
    trainer.learn(
        verbose=1
    )

if __name__ == "__main__":
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_default_dtype(torch.float32)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    main()