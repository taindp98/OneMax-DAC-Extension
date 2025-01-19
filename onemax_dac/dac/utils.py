from datetime import datetime
import torch
import random
import os
import numpy as np
import json
import pandas as pd

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
    # Format the date and time separately
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

# def plot_policy(results_fpath: str) -> torch.Tensor:
#     evals_data = json.load(open(results_fpath))
#     evals_data = pd.DataFrame(evals_data)

#     # optimal policy
#     optimal_policy = eval_data["optimal_policies"][i]
#     optimal_runtime = eval_data["optimal_runtime_means"][i]

#     # mean runtime of learnt policies
#     eval_runtime_means = [ls[i] for ls in eval_data["eval_runtime_means"]]

#     # best policy
#     best_mean_runtime_id = np.argmin(eval_runtime_means)
#     best_mean_runtime = np.min(eval_runtime_means)

#     best_policy = eval_data["eval_policies"][best_mean_runtime_id][
#         i
#     ]  ## shape: (train_steps, 1, problem_size)
#     best_gap = (best_mean_runtime - optimal_runtime) / optimal_runtime

#     if off_env_eval:
#         last_eval_data = np.load(
#             results_fpath.replace("evaluations", "evaluations_last"), allow_pickle=True
#         )
#         last_runtime_mean_id = np.argmin(last_eval_data["eval_runtime_means"])
#         last_mean_runtime = np.min(last_eval_data["eval_runtime_means"])
#         last_policy = last_eval_data["eval_policies"][last_runtime_mean_id]
#         last_gap = (last_mean_runtime - optimal_runtime) / optimal_runtime

#     # plot the best policy
#     plt.figure(figsize=(10, 6), facecolor="white")

#     if len(best_policy.shape) != 1:
#         best_policy = best_policy[:, 1]
#     plt.plot(range(n), optimal_policy, label="optimal", c="r")
#     plt.plot(range(n), best_policy, label="best", c="b")
#     if off_env_eval:
#         if len(last_policy.shape) != 1:
#             last_policy = last_policy[:, 1]
#         plt.plot(range(n), last_policy, label="last", c="orange", linestyle="--")
#     plt.legend()
#     if off_env_eval:
#         text = f"Opt / Best: {optimal_runtime:.2f} / {best_mean_runtime:.2f} ({best_gap*100:.2f})% | Opt / Last: {optimal_runtime:.2f} / {last_mean_runtime:.2f} ({last_gap*100:.2f}%)"

#     else:
#         text = f"Opt / Best: {optimal_runtime:.2f} / {best_mean_runtime:.2f} ({best_gap*100:.2f})%"
#     plt.title(text)
#     if verbose:
#         print(text)
#     else:
#         # Save the plot to a BytesIO object
#         buf = io.BytesIO()
#         plt.savefig(buf, format="png")
#         buf.seek(0)
#         plt.close()  # Close the figure to avoid overlap
#         image = Image.open(buf)
#         image = np.array(image)
#         if image.shape[2] == 4:  # If the image has an alpha channel, remove it
#             image = image[:, :, :3]
#         image = (
#             torch.tensor(image).permute(2, 0, 1).unsqueeze(0)
#         )  # Convert to (1, C, H, W) format

#         return image