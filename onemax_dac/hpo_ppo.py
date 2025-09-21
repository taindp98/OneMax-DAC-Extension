# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging

import hydra
import stable_baselines3
from omegaconf import DictConfig, OmegaConf
from stable_baselines3.common.monitor import Monitor
from onemax_dac.utils import make_env, get_time_str, seed_everything
from onemax_dac.train_ppo import EvalCallback
import os
import numpy as np
import json

log = logging.getLogger(__name__)

LIST_TUNING_BENCHS = [
    {
        "name": "OLLGATheoryPPO",
        "discrete_action": True,
        "action_choices": [[1, 2, 4, 8, 16, 32, 64]],
        "problem": "OneMax",
        "instance_set_path": "om_ollga_100_medium.csv",
        "observation_description": "n, f(x)",
        "reward_choice": "imp_minus_evals_scaling",
        "seed": 0,
    },
    {
        "name": "OLLGATheoryPPO",
        "discrete_action": True,
        "action_choices": [[1, 2, 4, 8, 16, 32, 64, 128]],
        "problem": "OneMax",
        "instance_set_path": "om_ollga_200_medium.csv",
        "observation_description": "n, f(x)",
        "reward_choice": "imp_minus_evals_scaling",
        "seed": 0,
    },
    {
        "name": "OLLGATheoryPPO",
        "discrete_action": True,
        "action_choices": [[1, 2, 4, 8, 16, 32, 64, 128, 256]],
        "problem": "OneMax",
        "instance_set_path": "om_ollga_300_medium.csv",
        "observation_description": "n, f(x)",
        "reward_choice": "imp_minus_evals_scaling",
        "seed": 0,
    },
]


def analyse_results(
    result_path: str,
):
    eval_data = np.load(result_path, allow_pickle=True)
    eval_runtime_means = [ls[0] for ls in eval_data["eval_runtime_means"]]
    best_ert = np.min(eval_runtime_means)
    ## AUC
    i = 0
    instance_set = eval_data["instance_set"].item()
    inst_id = eval_data["inst_ids"][i]
    instance = instance_set[inst_id]
    opt_mean = eval_data["optimal_runtime_means"][i]
    n = instance["size"]
    cutoff = 0.8 * n * n
    eval_runtime_means = [v if abs(v) <= cutoff else cutoff for v in eval_runtime_means]
    eval_runtime_means = (
        eval_runtime_means - opt_mean + 1e-3
    )  # plus 1 to avoid log(0) (when mean = opt_mean)
    ## negative values are not possible
    eval_runtime_means = [v for v in eval_runtime_means if v >= 0]
    eval_runtime_means = [np.log(v) for v in eval_runtime_means]
    log_auc = np.trapz(eval_runtime_means)
    ## norm AUC by number evaluations
    log_auc = log_auc / len(eval_runtime_means)
    log_auc = log_auc / n
    ## norm ert by the n^2
    best_ert = best_ert / (n * n)
    return best_ert, log_auc


@hydra.main(config_path="configs", config_name="hpo_ppo_7")
def train_ppo(cfg: DictConfig):
    log.info(OmegaConf.to_yaml(cfg))

    log.info(
        f"Training {cfg.algorithm.agent_class} Agent on {cfg.env_name} for {cfg.algorithm.total_timesteps} steps"
    )
    exp_params = OmegaConf.to_container(cfg.experiment, resolve=True)

    metrics = []
    for bench_params in LIST_TUNING_BENCHS:
        seed = 123
        seed_everything(seed)
        train_env_params = None
        eval_env_params = None
        # Write results to a log file
        exp_name = bench_params["instance_set_path"].split(".")[0]
        date_str, time_str = get_time_str()
        out_dir = os.path.join(cfg.out_dir, f"{exp_name}/{date_str}/{time_str}_{seed}")
        os.makedirs(out_dir, exist_ok=True)
        env = make_env(bench_params, train_env_params)
        env = Monitor(env, os.path.join(out_dir, "monitor"))

        # Create the evaluation environment
        eval_env = make_env(bench_params, eval_env_params)

        agent_class = getattr(stable_baselines3, cfg.algorithm.agent_class)

        if cfg.load:
            model = agent_class.load(cfg.load, env=env, **cfg.algorithm.model_kwargs)
        else:
            model = agent_class(cfg.algorithm.policy_model, env, **cfg.algorithm.model_kwargs)

        callback = EvalCallback(
            agent=model,
            eval_env=eval_env,
            bench_params=bench_params,
            eval_env_params=eval_env_params,
            eval_interval=exp_params["eval_interval"],
            n_eval_episodes=exp_params["eval_n_episodes"],
            best_model_save_path=out_dir,
            result_path=os.path.join(out_dir, "eval_infos.gzip"),
            n_cpus=cfg.experiment.n_cores,
        )

        model.learn(
            total_timesteps=cfg.algorithm.total_timesteps,
            reset_num_timesteps=False,
            callback=callback,
        )

        result_path = os.path.join(out_dir, "evaluations.npz")
        best_ert, log_auc = analyse_results(result_path=result_path)

        ## dump best ERT and AUC to the log file
        with open(os.path.join(out_dir, "tuning_results.json"), "w") as f:
            json.dump(
                {
                    "problem": bench_params["instance_set_path"].split(".")[0],
                    "total_timesteps": cfg.algorithm.total_timesteps,
                    "trial": OmegaConf.to_container(cfg.algorithm.model_kwargs, resolve=True),
                    "best_ert": best_ert,
                    "log_auc": log_auc,
                },
                f,
            )
        metrics.append(best_ert + log_auc)
    avg_metric = np.mean(metrics)
    return avg_metric


if __name__ == "__main__":
    train_ppo()
