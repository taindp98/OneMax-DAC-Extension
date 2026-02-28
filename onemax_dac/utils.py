import sys
import yaml
import os
import torch
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import io
from PIL import Image

from dacbench.benchmarks import (
    RLSTheoryBenchmark,
    OLLGATheoryBenchmark,
    OLLGAFactTheoryBenchmark,
    OLLGATheoryPPOBenchmark,
    OLLGAFactL1TheoryBenchmark,
)

default_exp_params = {
    "n_cores": 1,
    "n_episodes": 1e6,
    "n_steps": 1e6,
    "eval_interval": 2000,
    "n_eval_episodes_per_instance": 50,
    "save_agent_at_every_eval": False,
    "seed": 123,
    "eval_mode": "formula",
    "use_cuda": False,
    "log_level": 1,
}

default_bench_params = {
    "name": "Theory",
    "alias": "evenly_spread",
    "discrete_action": True,
    "action_choices": [1, 17, 33],
    "problem": "LeadingOnes",
    "instance_set_path": "lo_rls_50_random.csv",
    "observation_description": "n,f(x)",
    "reward_choice": "imp_minus_evals",
    "seed": 123,
}

default_eval_env_params = {
    "reward_choice": "minus_evals",
    "cutoff": 1e5,
}


def read_config(config_yml_fn: str = "output/config.yml"):
    with open(config_yml_fn, "r") as f:
        params = yaml.safe_load(f)

    for key in default_exp_params:
        if key not in params["experiment"]:
            params["experiment"][key] = default_exp_params[key]

    for key in default_bench_params:
        if key not in params["bench"]:
            params["bench"][key] = default_bench_params[key]

    train_env_params = eval_env_params = None
    if "train_env" in params:
        train_env_params = params["train_env"]
    if "eval_env" in params:
        eval_env_params = params["eval_env"]
        for key in default_eval_env_params:
            if key not in eval_env_params:
                eval_env_params[key] = default_eval_env_params[key]
    return (
        params["experiment"],
        params["bench"],
        params["agent"],
        train_env_params,
        eval_env_params,
    )


def make_env(bench_params, env_config=None, test_env=False):
    """
    env_config will override bench_params
    """
    bench_class = globals()[bench_params["name"] + "Benchmark"]

    params = bench_params.copy()
    del params["name"]
    if env_config:
        for name, val in env_config.items():
            params[name] = val

    # pprint(params)
    bench = bench_class(config=params)
    env = bench.get_environment(test_env)
    # env = wrappers.FlattenObservation(env)
    return env


def object_to_dict(obj):
    if isinstance(obj, dict):
        return {k: object_to_dict(v) for k, v in obj.items()}
    elif hasattr(obj, "__dict__"):
        return {k: object_to_dict(v) for k, v in obj.__dict__.items()}
    elif isinstance(obj, list):
        return [object_to_dict(i) for i in obj]
    else:
        return obj


class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                value = Config(value)
            setattr(self, key, value)


def load_config(config_path):
    with open(config_path, "r") as file:
        config_dict = yaml.safe_load(file)
    return Config(config_dict)


def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_time_str():
    """
    Get the current time as a string
    """
    now = datetime.now()
    date_str = now.strftime("%Y%m%d")
    time_str = now.strftime("%H%M%S")
    return date_str, time_str


def object_to_dict(obj):
    if isinstance(obj, dict):
        return {k: object_to_dict(v) for k, v in obj.items()}
    elif hasattr(obj, "__dict__"):
        return {k: object_to_dict(v) for k, v in obj.__dict__.items()}
    elif isinstance(obj, list):
        return [object_to_dict(i) for i in obj]
    else:
        return obj


def plot_hittings(results_fpath: str) -> torch.Tensor:
    eval_data = np.load(results_fpath, allow_pickle=True)
    instance_set = eval_data["instance_set"].item()
    i = 0
    inst_id = eval_data["inst_ids"][i]
    instance = instance_set[inst_id]

    n = instance["size"]

    # optimal runtime
    opt_mean = eval_data["optimal_runtime_means"][i]
    opt_std = eval_data["optimal_runtime_stds"][i]
    eval_timesteps = eval_data["eval_timesteps"]
    # mean/std of learnt policies
    eval_runtime_means = [ls[i] for ls in eval_data["eval_runtime_means"]]
    eval_runtime_stds = [ls[i] for ls in eval_data["eval_runtime_stds"]]
    # those values should be positive
    eval_runtime_means = [np.absolute(v) for v in eval_runtime_means]
    eval_runtime_stds = [np.absolute(v) for v in eval_runtime_stds]

    # where we hit the optimal (full run)
    close_to_theory = np.where(eval_runtime_means <= opt_mean + 0.25 * opt_std)[0]

    # cap inf values (for plotting only)
    inf_val = (opt_mean + opt_std) * 2

    # some values are not inf but also very large, we will set them as inf
    inf_val = min(inf_val, max(eval_runtime_means))
    # replace inf with inf_val (for plotting only)
    inf_concept = np.inf
    eval_runtime_means = np.asarray(
        [v if v != inf_concept else inf_val for v in eval_runtime_means]
    )
    eval_runtime_stds = np.asarray([v if v != np.inf else 0 for v in eval_runtime_stds])

    # smoothness
    # sd_diff = np.std(np.diff(eval_runtime_means)).round(2)
    # avg_diff = np.mean(np.diff(eval_runtime_means)).round(2)
    # lag one autocorrelation
    # lagone = pd.Series(eval_runtime_means).autocorr(lag=1).round(2)
    # fig = plt.figure(figsize=(10, 4))
    plt.figure(figsize=(10, 6), facecolor="white")
    # plot the optimal
    color_opt = "sandybrown"
    _ = plt.step(
        eval_timesteps,
        [opt_mean] * len(eval_timesteps),
        where="post",
        label="theory",
        ls="--",
        color=color_opt,
    )
    u = [opt_mean + opt_std] * len(eval_timesteps)
    l = [opt_mean - opt_std] * len(eval_timesteps)
    _ = plt.fill_between(eval_timesteps, u, l, alpha=0.2, step="post", color=color_opt)

    eval_runtime_means = np.clip(eval_runtime_means, 0, inf_val)
    eval_runtime_stds = np.clip(eval_runtime_stds, 0, inf_val)

    # RL
    color_rl = "tab:blue"
    _ = plt.step(eval_timesteps, eval_runtime_means, where="post", label="RL", color=color_rl)
    u = eval_runtime_means + eval_runtime_stds
    u = [
        min(u[v], inf_val) for v in range(len(u))
    ]  # make sure we don't jump over inf_val in the plot
    l = eval_runtime_means - eval_runtime_stds
    _ = plt.fill_between(eval_timesteps, u, l, alpha=0.2, step="post", color=color_rl)

    # plot points where we are close to the optimal
    _ = plt.scatter(
        eval_timesteps[close_to_theory],
        eval_runtime_means[close_to_theory],
        color="green",
    )

    hittings_ratio = len(close_to_theory) / len(eval_timesteps)

    _ = plt.xlim([0, max(eval_timesteps)])
    _ = plt.ylim([max(min(l), 0), inf_val + 100])
    _ = plt.legend()
    text = f"# Hittings: {len(close_to_theory)}/{len(eval_timesteps)} (={hittings_ratio:.2f})"

    _ = plt.title(text)

    # Save the plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()  # Close the figure to avoid overlap

    image = Image.open(buf)
    image = np.array(image)
    if image.shape[2] == 4:  # If the image has an alpha channel, remove it
        image = image[:, :, :3]
    figure = torch.tensor(image).permute(2, 0, 1).unsqueeze(0)  # Convert to (1, C, H, W) format

    figure = figure.squeeze(0)
    # Convert the tensor to a NumPy array
    figure_np = figure.permute(1, 2, 0).cpu().numpy()
    # Convert the NumPy array to a PIL image
    pil_image = Image.fromarray(figure_np)
    pil_image.save(os.path.join(os.path.dirname(results_fpath), "hittings.png"))
