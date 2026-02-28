import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import io
import torch
import pandas as pd
import seaborn as sns

sns.set_style("white")


def calculate_auc(n, eval_runtime_means, opt_mean, exp):
    """
    Calculate the area under the curve (AUC) of the log-scaled gap between the eval runtime means and the optimal mean.
    The gap is calculated as eval_runtime_means - opt_mean, and we take the log of the gap to calculate the AUC.
    """
    in_inf = np.inf
    eval_runtime_means = [v if abs(v) <= 0.8 * n * n else in_inf for v in eval_runtime_means]
    # print(f"eval_runtime_means: {eval_runtime_means}")
    out_inf = 1e20 * 0.8 * n * n
    means = [
        eval_runtime_means[i] if eval_runtime_means[i] != in_inf else out_inf
        for i in range(len(eval_runtime_means))
    ]
    means = means - opt_mean + 1e-3  # plus 1 to avoid log(0) (when mean = opt_mean)
    ## negative values are not possible
    means = [v for v in means if v >= 0]
    means = [np.log(v) for v in means]
    log_auc = np.trapz(means)
    return log_auc


def calculate_smoothed_auc(n, eval_runtime_means, opt_mean, exp, smooth_level: float = 0.3):
    in_inf = np.inf
    eval_runtime_means = [v if abs(v) <= 0.8 * n * n else in_inf for v in eval_runtime_means]
    # print(f"eval_runtime_means: {eval_runtime_means}")
    out_inf = 1e20 * 0.8 * n * n
    means = [
        eval_runtime_means[i] if eval_runtime_means[i] != in_inf else out_inf
        for i in range(len(eval_runtime_means))
    ]
    df = pd.DataFrame({"Index": np.arange(len(means)), "Runtime Means": means})
    smoothed_means = df["Runtime Means"].rolling(window=20, min_periods=1).mean()
    smoothed_means = (
        smoothed_means - opt_mean + 1e-3
    )  # plus 1 to avoid log(0) (when mean = opt_mean)
    ## negative values are not possible
    smoothed_means = [v for v in smoothed_means if v >= 0]
    smoothed_means = [np.log(v) for v in smoothed_means]
    log_auc = np.trapz(smoothed_means)
    return log_auc


def get_method_auc(results_fpath, exp, norm: bool = True, ratio: float = 1.0):
    eval_data = np.load(results_fpath, allow_pickle=True)
    instance_set = eval_data["instance_set"].item()
    i = 0
    inst_id = eval_data["inst_ids"][i]
    instance = instance_set[inst_id]
    opt_mean = eval_data["optimal_runtime_means"][i]
    n = instance["size"]
    eval_runtime_means = [ls[i] for ls in eval_data["eval_runtime_means"]]
    eval_runtime_means = [np.absolute(v) for v in eval_runtime_means]

    ## get samples from the last ratio of the evaluations
    eval_runtime_means = eval_runtime_means[int(len(eval_runtime_means) * (1 - ratio)) :]
    # eval_runtime_means = eval_runtime_means[
    #     : int(len(eval_runtime_means) * ratio)
    # ]
    smooth_auc = calculate_smoothed_auc(n, eval_runtime_means, opt_mean=opt_mean, exp=exp)

    log_auc = calculate_auc(n, eval_runtime_means, opt_mean=opt_mean, exp=exp)
    return log_auc, smooth_auc


def get_hittings(results_fpath, ratio: float = 1.0):
    eval_data = np.load(results_fpath, allow_pickle=True)
    # optimal runtime
    i = 0
    opt_mean = eval_data["optimal_runtime_means"][i]
    opt_std = eval_data["optimal_runtime_stds"][i]
    eval_timesteps = eval_data["eval_timesteps"]
    # mean/std of learnt policies
    eval_runtime_means = [ls[i] for ls in eval_data["eval_runtime_means"]]
    eval_runtime_stds = [ls[i] for ls in eval_data["eval_runtime_stds"]]
    # those values should be positive
    eval_runtime_means = [np.absolute(v) for v in eval_runtime_means]
    # eval_runtime_stds = [np.absolute(v) for v in eval_runtime_stds]

    ## get samples from the last ratio of the evaluations
    eval_runtime_means = eval_runtime_means[int(len(eval_runtime_means) * (1 - ratio)) :]
    # eval_runtime_stds = eval_runtime_stds[int(len(eval_runtime_stds) * (1 - ratio)) :]
    # where we hit the optimal (full run)
    num_hittings = len(np.where(eval_runtime_means <= opt_mean + 0.25 * opt_std)[0])
    return num_hittings


def get_gap(results_fpath, problem_size):
    cutoff = 0.8 * int(problem_size) * int(problem_size)
    eval_data = np.load(results_fpath, allow_pickle=True)
    optimal_runtime = eval_data["optimal_runtime_means"][0]
    last_results_fpath = results_fpath.replace("evaluations.npz", "evaluations_last.npz")
    if os.path.exists(last_results_fpath):
        last_eval_data = np.load(
            results_fpath.replace("evaluations.npz", "evaluations_last.npz"),
            allow_pickle=True,
        )
        best_mean_runtime = np.min(last_eval_data["eval_runtime_means"])
    else:
        eval_runtime_means = [ls[0] for ls in eval_data["eval_runtime_means"]]
        best_mean_runtime = np.min(eval_runtime_means)
    if best_mean_runtime == cutoff:
        best_mean_runtime = 1e6
    gap = best_mean_runtime - optimal_runtime
    gap = gap if gap > 0 else 0
    gap = gap * 100 / optimal_runtime
    return gap


def plot_polices(
    results_fpath: str, off_env_eval: bool = False, verbose: bool = False
) -> torch.Tensor:
    eval_data = np.load(results_fpath, allow_pickle=True)
    instance_set = eval_data["instance_set"].item()
    i = 0
    inst_id = eval_data["inst_ids"][i]
    instance = instance_set[inst_id]

    n = instance["size"]

    # optimal policy
    optimal_policy = eval_data["optimal_policies"][i]
    optimal_runtime = eval_data["optimal_runtime_means"][i]

    # mean runtime of learnt policies
    eval_runtime_means = [ls[i] for ls in eval_data["eval_runtime_means"]]

    # best policy
    best_mean_runtime_id = np.argmin(eval_runtime_means)
    best_mean_runtime = np.min(eval_runtime_means)

    best_policy = eval_data["eval_policies"][best_mean_runtime_id][i][
        :, 0
    ]  ## shape: (train_steps, 1, problem_size)
    best_gap = (best_mean_runtime - optimal_runtime) / optimal_runtime

    if off_env_eval:
        last_eval_data = np.load(
            results_fpath.replace("evaluations", "evaluations_last"), allow_pickle=True
        )
        last_runtime_mean_id = np.argmin(last_eval_data["eval_runtime_means"])
        last_mean_runtime = np.min(last_eval_data["eval_runtime_means"])
        last_policy = last_eval_data["eval_policies"][last_runtime_mean_id][:, 0]
        last_gap = (last_mean_runtime - optimal_runtime) / optimal_runtime

    # plot the best policy
    plt.figure(figsize=(10, 6), facecolor="white")

    if len(best_policy.shape) != 1:
        best_policy = best_policy[:, 1]
    plt.plot(range(n), optimal_policy, label="theory", c="r")
    plt.plot(range(n), best_policy, label="RL[eval]", c="b")
    if off_env_eval:
        if len(last_policy.shape) != 1:
            last_policy = last_policy[:, 1]
        plt.plot(range(n), last_policy, label="RL[test]", c="orange", linestyle="--")
    plt.legend()
    if off_env_eval:
        text = f"Opt / Best: {optimal_runtime:.2f} / {best_mean_runtime:.2f} ({best_gap*100:.2f})% | Opt / Last: {optimal_runtime:.2f} / {last_mean_runtime:.2f} ({last_gap*100:.2f}%)"

    else:
        text = f"Opt / Best: {optimal_runtime:.2f} / {best_mean_runtime:.2f} ({best_gap*100:.2f})%"
    plt.title(text)
    if verbose:
        print(text)
    else:
        # Save the plot to a BytesIO object
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close()  # Close the figure to avoid overlap
        image = Image.open(buf)
        image = np.array(image)
        if image.shape[2] == 4:  # If the image has an alpha channel, remove it
            image = image[:, :, :3]
        image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0)  # Convert to (1, C, H, W) format

        return image


def plot_hittings(
    results_fpath: str, off_env_eval: bool = False, verbose: bool = False
) -> torch.Tensor:
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
    inf_concept = (0.8 * n * n - 1) if off_env_eval else np.inf
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
    if verbose:
        print(text)
    else:
        # Save the plot to a BytesIO object
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close()  # Close the figure to avoid overlap

        image = Image.open(buf)
        image = np.array(image)
        if image.shape[2] == 4:  # If the image has an alpha channel, remove it
            image = image[:, :, :3]
        image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0)  # Convert to (1, C, H, W) format

        return image


def plot_results(
    results_fpath: str, off_env_eval: bool = False, verbose: bool = False
) -> torch.Tensor:
    policies_img = plot_polices(results_fpath, off_env_eval, verbose)
    hittings_img = plot_hittings(results_fpath, off_env_eval, verbose)
    return policies_img, hittings_img
