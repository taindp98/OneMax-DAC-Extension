from datetime import datetime
import torch
import random
import os
import numpy as np

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