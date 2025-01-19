import argparse
from typing import List
from onemax_dac.configs.training_config import TrainingConfig
from onemax_dac.configs.policy_config import PolicyConfig
from onemax_dac.configs.env_config import EnvConfig


def none_or_str(value):
    if value.lower() == "none":
        return None
    return value


def parse_args():
    parser = argparse.ArgumentParser(description="Train RL Agent with Custom Configurations")

    # TrainingConfig arguments
    parser.add_argument(
        "--max_steps", type=int, default=1000000, help="Max number of steps for training"
    )
    parser.add_argument(
        "--buffer_size", type=int, default=1000000, help="Size of the replay buffer"
    )
    parser.add_argument(
        "--epsilon_start", type=float, default=1.0, help="Starting epsilon value for exploration"
    )
    parser.add_argument(
        "--epsilon_end", type=float, default=0.2, help="Ending epsilon value for exploration"
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=10000,
        help="Number of warmup steps before training starts",
    )
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size for training")
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate for the agent"
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor (gamma)")
    parser.add_argument("--tau", type=float, default=0.01, help="Soft update parameter (tau)")
    parser.add_argument(
        "--loss_fn", type=str, default="MSE", choices=["MSE", "Huber"], help="Loss function"
    )
    parser.add_argument("--eval_interval", type=int, default=2000, help="Interval for evaluation")
    parser.add_argument(
        "--n_eval_episodes", type=int, default=100, help="Number of episodes to run for evaluation"
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs", help="Directory for output files"
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="cpu",
        choices=["cpu", "gpu"],
        help="Training accelerator",
    )
    parser.add_argument(
        "--num_workers", type=int, default=1, help="Number of workers for data loading"
    )
    parser.add_argument("--wandb", action="store_true", help="Enable logging to Weights & Biases")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--fixed_shift",
        type=none_or_str,
        default=None,
        help="Fixed shift value for reward calculation",
    )

    # PolicyConfig arguments
    parser.add_argument("--policy_name", type=str, default="DDQN", help="Policy name")
    parser.add_argument(
        "--net_arch",
        type=int,
        nargs="+",
        default=[50, 50],
        help="Network architecture (list of layers)",
    )
    parser.add_argument(
        "--activation_fn",
        type=str,
        default="ReLU",
        choices=["ReLU", "Tanh", "Sigmoid"],
        help="Activation function",
    )

    # EnvConfig arguments
    parser.add_argument(
        "--problem_size",
        type=int,
        required=True,
        help="Size of the problem (depends on environment)",
    )
    parser.add_argument("--state_dim", type=int, default=2, help="Dimension of the state space")
    parser.add_argument(
        "--discrete_action", type=bool, default=True, help="Whether the action space is discrete"
    )
    parser.add_argument(
        "--action_choices", type=int, nargs="+", default=[], help="List of possible action choices"
    )
    parser.add_argument(
        "--reward_choice", type=str, default="imp_minus_evals_shifted", help="Reward choice type"
    )
    parser.add_argument(
        "--init_obj_rate", type=none_or_str, default=0.5, help="Initial object value"
    )

    # Parse the arguments
    args = parser.parse_args()

    # Instantiate the configuration classes with the parsed arguments
    training_config = TrainingConfig(
        max_steps=args.max_steps,
        buffer_size=args.buffer_size,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        warmup_steps=args.warmup_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        tau=args.tau,
        loss_fn=args.loss_fn,
        eval_interval=args.eval_interval,
        n_eval_episodes=args.n_eval_episodes,
        output_dir=args.output_dir,
        accelerator=args.accelerator,
        num_workers=args.num_workers,
        wandb=args.wandb,
        seed=args.seed,
        fixed_shift=args.fixed_shift,
    )

    policy_config = PolicyConfig(
        policy_name=args.policy_name,
        net_arch=args.net_arch,
        activation_fn=args.activation_fn,
    )

    env_config = EnvConfig(
        problem_size=args.problem_size,
        state_dim=args.state_dim,
        discrete_action=args.discrete_action,
        action_choices=args.action_choices,
        reward_choice=args.reward_choice,
        seed=args.seed,
        init_obj_rate=args.init_obj_rate,
    )

    return training_config, policy_config, env_config
