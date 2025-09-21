import argparse
import os
import shutil
import time
import numpy as np
import yaml

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from onemax_dac.utils import make_env, read_config, plot_hittings, get_time_str
from onemax_dac.evals import (
    PPOSingleParamEval,
    PPOMultiParamEval,
)


class EvalCallback(BaseCallback):
    """
    A custom callback that evaluates the agent every `eval_interval` steps.
    """

    def __init__(
        self,
        agent,
        eval_env,
        bench_params,
        eval_env_params,
        eval_interval: int,
        n_eval_episodes: int,
        best_model_save_path: str,
        result_path: str,
        n_cpus: int = 1,
        **kwargs,
    ):
        super(EvalCallback, self).__init__(**kwargs)
        self.eval_env = eval_env
        self.eval_interval = eval_interval
        self.n_eval_episodes = n_eval_episodes
        self.best_mean_reward = -np.inf
        self.best_model_save_path = best_model_save_path
        self.result_path = result_path
        self.evaluator = None
        self.bench_params = bench_params
        self.eval_env_params = eval_env_params
        if isinstance(eval_env.action_choices[0][0], list):
            self.evaluator = PPOMultiParamEval(
                agent=agent,
                eval_env=self.eval_env,
                bench_params=self.bench_params,
                eval_env_params=self.eval_env_params,
                n_eval_episodes_per_instance=self.n_eval_episodes,
                log_path=f"{self.best_model_save_path}",
                n_cpus=n_cpus,
            )
        else:
            self.evaluator = PPOSingleParamEval(
                agent=agent,
                eval_env=self.eval_env,
                bench_params=self.bench_params,
                eval_env_params=self.eval_env_params,
                n_eval_episodes_per_instance=self.n_eval_episodes,
                log_path=f"{self.best_model_save_path}",
                n_cpus=n_cpus,
            )

    def _on_step(self) -> bool:
        """
        This method will be called in the model's `learn` method.
        We evaluate the agent every `eval_interval` steps.
        """
        if (self.n_calls + 1) % self.eval_interval == 0:
            self.evaluator.eval(self.n_calls + 1)
        return True


def main():
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", "-o", type=str, default="output", help="output folder")
    parser.add_argument("--setting-file", "-s", type=str, help="yml file with all settings")
    parser.add_argument("--seed", type=int, default=1, help="random seed for reproducibility")
    parser.add_argument("--ent-coef", "-e", type=float, default=0.0, help="entropy coefficient")
    parser.add_argument(
        "--n-cpus",
        "-c",
        type=int,
        default=1,
        help="number of CPUs to use for evaluation",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="batch size for training")
    parser.add_argument("--clip-range", type=float, default=0.2, help="clipping range for PPO")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda parameter")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument(
        "--learning-rate", "--lr", type=float, default=0.0003, help="learning rate"
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=10,
        help="number of epochs for policy optimization",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=2048,
        help="number of steps to run for each environment per update",
    )
    parser.add_argument(
        "--reward-norm",
        action="store_true",
        help="normalize the rewards during training",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=500000,
        help="total timesteps for training",
    )
    args = parser.parse_args()

    # Get configuration from train_conf_ppo.yml
    config_yml_fn = args.setting_file
    exp_params, bench_params, agent_params, train_env_params, eval_env_params = read_config(
        config_yml_fn
    )

    if exp_params["n_cores"] > 1:
        print("WARNING: n_cores>1 is not yet supported")

    date_str, time_str = get_time_str()

    # Define defaults to check against
    DEFAULTS = {
        "batch_size": 64,
        "clip_range": 0.2,
        "gae_lambda": 0.95,
        "gamma": 0.99,
        "learning_rate": 0.0003,
        "n_epochs": 10,
        "n_steps": 2048,
        "ent_coef": 0.0,
        "reward_norm": False,
        "total_timesteps": exp_params["n_steps"],
    }

    batch_size = args.batch_size
    clip_range = args.clip_range
    gae_lambda = args.gae_lambda
    gamma = args.gamma
    learning_rate = args.learning_rate
    policy_kwargs = {}
    use_sde = True if not bench_params["discrete_action"] else False
    ent_coef = args.ent_coef
    n_epochs = args.n_epochs
    n_steps = args.n_steps
    n_cpus = args.n_cpus
    reward_norm = args.reward_norm
    total_timesteps = args.total_timesteps

    # Check which parameters were overridden and build suffix
    overridden_params = []
    param_mapping = {
        "batch_size": batch_size,
        "clip_range": clip_range,
        "gae_lambda": gae_lambda,
        "gamma": gamma,
        "learning_rate": learning_rate,
        "n_epochs": n_epochs,
        "n_steps": n_steps,
        "ent_coef": ent_coef,
        "reward_norm": reward_norm,
        "total_timesteps": total_timesteps,
    }

    for param, current_val in param_mapping.items():
        if current_val != DEFAULTS[param]:
            # Format the parameter name and value for directory naming
            param_short = param.replace("_", "").replace("learning_rate", "lr")
            overridden_params.append(f"{param_short}{current_val}")

    # Create directory suffix from overridden parameters
    override_suffix = "_".join(overridden_params) if overridden_params else "default"

    exp_name = config_yml_fn.split("/")[-1].split(".")[0]
    out_dir = os.path.join(args.out_dir, f"{exp_name}/{override_suffix}/{time_str}_{args.seed}")
    if os.path.isdir(out_dir) is False:
        os.makedirs(out_dir, exist_ok=True)
        shutil.copyfile(args.setting_file, os.path.join(out_dir, "config.yml"))
        ## dump the config to the output folder
        with open(os.path.join(out_dir, "train_args.yml"), "w") as f:
            yaml.dump(
                {
                    "batch_size": batch_size,
                    "clip_range": clip_range,
                    "gae_lambda": gae_lambda,
                    "gamma": gamma,
                    "learning_rate": learning_rate,
                    "ent_coef": ent_coef,
                    "n_epochs": n_epochs,
                    "policy_kwargs": policy_kwargs,
                    "use_sde": use_sde,
                    "n_steps": n_steps,
                    "reward_norm": reward_norm,
                },
                f,
            )

    if reward_norm:
        env = DummyVecEnv(
            [
                lambda: Monitor(
                    make_env(bench_params, train_env_params),
                    os.path.join(out_dir, f"monitor_{args.seed}"),
                )
            ]
        )
        env = VecNormalize(env, norm_reward=True)
    else:
        env = make_env(bench_params, train_env_params)
        env = Monitor(env, os.path.join(out_dir, f"monitor_{args.seed}"))

    # Create the evaluation environment
    eval_env = make_env(bench_params, eval_env_params)
    assert agent_params["name"] == "ppo", "Only PPO is supported for now"
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        batch_size=batch_size,
        clip_range=clip_range,
        gae_lambda=gae_lambda,
        gamma=gamma,
        learning_rate=learning_rate,
        ent_coef=ent_coef,
        n_epochs=n_epochs,
        policy_kwargs=policy_kwargs,
        use_sde=use_sde,
        n_steps=n_steps,
    )
    # Use the custom callback to evaluate agent's performance after a certain number of steps
    eval_callback = EvalCallback(
        agent=model,
        eval_env=eval_env,
        bench_params=bench_params,
        eval_env_params=eval_env_params,
        eval_interval=exp_params["eval_interval"],
        n_eval_episodes=exp_params["eval_n_episodes"],
        best_model_save_path=out_dir,
        result_path=os.path.join(out_dir, "eval_infos.gzip"),
        n_cpus=n_cpus,
    )
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    model.save(os.path.join(out_dir, "ppo_final"))

    plot_hittings(results_fpath=os.path.join(out_dir, "evaluations.npz"))
    total_time = time.time() - start_time
    print(f"Total runtime: {total_time}")


if __name__ == "__main__":
    main()
