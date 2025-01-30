from datetime import datetime
import torch
import random
import os
import numpy as np
import json
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from prettytable import PrettyTable


def soft_update(target, source, tau: float = 0.01):
    """
    Simple Helper for updating target-network parameters
    :param target: target network
    :param source: policy network
    :param tau: weight to regulate how strongly to update (1 -> copy over weights)
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def get_time_str():
    """
    Get the current time as a string
    """
    now = datetime.now()
    date_str = now.strftime("%Y%m%d")
    time_str = now.strftime("%H%M%S")
    time_str = f"{date_str}_{time_str}"
    return time_str


def to_tensor(ndarray, device=torch.device("cpu")):
    """
    Convert a numpy array to a PyTorch tensor
    """
    tensor = torch.tensor(ndarray, dtype=torch.float32)
    return tensor.to(device)


def seed_everything(seed=42):
    """
    Seed everything for reproducibility
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def plot_policies(problem_size: int, policies: dict, save_dir: str, segment: float = 0.5):
    """
    Plot the policies of the different methods
    Args:
        problem_size: The size of the problem
        policies: The policies of the different methods
        save_dir: The directory to save the plot
        segment: The segment of the policy to plot
    """
    sns.set(style="white")
    plt.figure(figsize=(5, 4))
    x = np.arange(problem_size)
    x = x[int(len(x) * (1 - segment)) :]
    colors = sns.color_palette("tab10", len(policies))
    for i, (m_name, policy) in enumerate(list(policies.items())):
        segment_policy = policy[int(len(policy) * (1 - segment)) :]
        plt.plot(
            x,
            segment_policy,
            label=m_name,
            c=colors[i],
        )
    plt.title(f"n={problem_size}")
    plt.xlabel("Fitness")
    plt.ylabel("$\lambda$")
    plt.grid(True)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/policy.pdf", dpi=600)


def plot_learning_curve(
    json_logdata: dict, save_dir: str, baseline: str = "discrete_theory", display_hr: bool = False
):
    """
    Plot the learning curve of the DAC agent
    Args:
        json_logdata: The log data
        save_dir: The directory to save the plot
        baseline: The baseline to compare the learning curve with
        display_hr: Display the hitting rate
    """
    sns.set(style="white")
    plt.figure(figsize=(5, 4), facecolor="white")

    assert baseline in ["continuous_theory", "discrete_theory"]
    trainval_data = pd.DataFrame(json_logdata["trainval"])
    test_data = pd.DataFrame(json_logdata["test"])
    baseline_mean = test_data[test_data["method"] == baseline]["mean_runtime"].values[0]
    baseline_std = test_data[test_data["method"] == baseline]["std_runtime"].values[0]

    eval_timesteps = trainval_data["step"].values
    eval_timesteps = eval_timesteps / 1000
    eval_runtime_means = np.array(trainval_data["runtimes"].tolist()).mean(axis=1)
    eval_runtime_stds = np.array(trainval_data["runtimes"].tolist()).std(axis=1)

    close_to_theory = np.where(eval_runtime_means <= baseline_mean + 0.25 * baseline_std)[0]

    eval_runtime_means = np.asarray([v for v in eval_runtime_means])
    eval_runtime_stds = np.asarray([v if v != np.inf else 0 for v in eval_runtime_stds])

    color_opt = "tab:red"
    _ = plt.step(
        eval_timesteps,
        [baseline_mean] * len(eval_timesteps),
        where="post",
        label=baseline,
        ls="--",
        color=color_opt,
        linewidth=1.5,
    )
    u = [baseline_mean + 0.25 * baseline_std] * len(eval_timesteps)
    l = [baseline_mean - 0.25 * baseline_std] * len(eval_timesteps)
    _ = plt.fill_between(eval_timesteps, u, l, alpha=0.2, step="post", color=color_opt)

    color_rl = "tab:blue"
    _ = plt.plot(
        eval_timesteps,
        eval_runtime_means,
        label="dac",
        color=color_rl,
        linewidth=1.5,
    )

    _ = plt.scatter(
        eval_timesteps[close_to_theory],
        eval_runtime_means[close_to_theory],
        color="green",
    )
    if display_hr:
        segment_hittings = []
        ratios = [1.0, 0.5, 0.25]
        for ratio in ratios:
            ert_means = eval_runtime_means[int(len(eval_runtime_means) * (1 - ratio)) :]
            num_hittings = len(np.where(ert_means <= baseline_mean + 0.25 * baseline_std)[0])
            ratio = num_hittings / len(ert_means)
            segment_hittings.append([ratio, num_hittings, len(ert_means)])

        table = PrettyTable()
        table.field_names = ["Training Period", "Hitting Rate"]
        y_labels = ["0-100", "50-100", "75-100"]
        for i, ratio in enumerate(ratios):
            text_hitting_ratio = f"{segment_hittings[i][0]:.2f}"
            text_division = f"{segment_hittings[i][1]}/{segment_hittings[i][2]}"
            table.add_row([y_labels[i], f"{text_division} (={text_hitting_ratio})"])

        table_text = table.get_string()

        plt.text(
            0.37,
            0.445,
            table_text,
            fontsize=9,
            transform=plt.gcf().transFigure,
            ha="left",
            va="top",
            fontfamily="monospace",
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3", alpha=0.5),
        )

    _ = plt.xlim([0, max(eval_timesteps)])
    _ = plt.ylim(
        [0.8 * min((min(eval_runtime_means), baseline_mean)), 1.2 * max(eval_runtime_means)]
    )
    plt.xlabel("Step (in thousands)")
    plt.ylabel("ERT")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(loc="upper right", prop={"size": 12, "weight": "normal"})
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/learning_curve.pdf", dpi=600)


def rename_state_dict(state_dict):
    """
    Renames the keys in a state_dict based on a given mapping.

    Args:
        state_dict (dict): The original state_dict with mismatched keys.
        rename_map (dict): A dictionary mapping old keys to new keys.

    Returns:
        dict: A new state_dict with renamed keys.
    """
    # Define the key mapping based on your error message
    rename_map = {
        "fc1.weight": "q_net.0.weight",
        "fc1.bias": "q_net.0.bias",
        "fc2.weight": "q_net.2.weight",
        "fc2.bias": "q_net.2.bias",
        "fc3.weight": "q_net.4.weight",
        "fc3.bias": "q_net.4.bias",
    }
    new_state_dict = {}
    for old_key, value in state_dict.items():
        new_key = rename_map.get(old_key, old_key)  # Rename if in map, else keep original
        new_state_dict[new_key] = value
    return new_state_dict
