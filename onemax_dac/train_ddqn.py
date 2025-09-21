import pickle
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple
import time
import os
import json
from tqdm import tqdm
import shutil
from onemax_dac.plot import plot_results
from onemax_dac.loggers import Logger
from joblib import Parallel, delayed
from onemax_dac.evals import DDQNFactEval, ollga_mp_single_run
import argparse
from onemax_dac.utils import (
    make_env,
    read_config,
    seed_everything,
    get_time_str,
    load_config,
    object_to_dict,
)
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")


def soft_update(target, source, tau):
    """
    Soft update the parameters of the target network using the source network.
    Args:
        target (nn.Module): Target network to update.
        source (nn.Module): Source (policy) network to copy parameters from.
        tau (float): Interpolation factor (1.0 = hard copy, <1.0 = soft update).
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    """
    Hard update: copy all parameters from source to target network.
    Args:
        target (nn.Module): Target network to update.
        source (nn.Module): Source network to copy parameters from.
    """
    soft_update(target, source, 1.0)


class BranchingQNetwork(nn.Module):
    """
    Branching Q-Network for multi-dimensional discrete action spaces.
    Implements dueling architecture optionally.
    Ref: https://arxiv.org/pdf/1711.08946.pdf
    Github: https://github.com/MoMe36/BranchingDQN
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_actions: int,
        net_arch: list = [50, 50],
        use_dueling: bool = True,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_actions = n_actions
        self.use_dueling = use_dueling

        self.model = nn.Sequential(
            nn.Linear(state_dim, net_arch[0]),
            nn.ReLU(),
            nn.Linear(net_arch[0], net_arch[1]),
            nn.ReLU(),
        )

        if self.use_dueling:
            self.value_head = nn.Linear(net_arch[1], 1)
        self.adv_heads = nn.ModuleList(
            [nn.Linear(net_arch[1], action_dim) for i in range(n_actions)]
        )

    def forward(self, x):
        """
        Forward pass through the network.
        Args:
            x (torch.Tensor): Input state tensor.
        Returns:
            torch.Tensor: Q-values for each action branch.
        """
        out = self.model(x)
        advs = torch.stack([l(out) for l in self.adv_heads], dim=1)
        if self.use_dueling:
            value = self.value_head(out)
            q_val = value.unsqueeze(2) + advs - advs.mean(2, keepdim=True)
        else:
            q_val = advs  # Without dueling, directly use the advantage values
        return q_val


class ReplayBuffer:
    """
    Experience Replay Buffer for storing transitions during training.
    Used for standard DQN/DDQN learning.
    """

    def __init__(self, max_size, rng=np.random.default_rng()):
        self._data = namedtuple(
            "ReplayBuffer",
            ["states", "actions", "next_states", "rewards", "terminal_flags"],
        )
        self._data = self._data(
            states=[], actions=[], next_states=[], rewards=[], terminal_flags=[]
        )
        self._size = 0
        self._max_size = max_size
        self.rng = rng

    def add_transition(self, state, action, next_state, reward, done):
        """
        Add a transition to the buffer.
        Args:
            state: Current state.
            action: Action taken.
            next_state: Next state after action.
            reward: Reward received.
            done: Terminal flag.
        """
        self._data.states.append(state)
        self._data.actions.append(action)
        self._data.next_states.append(next_state)

        self._data.rewards.append(reward)
        self._data.terminal_flags.append(done)
        self._size += 1
        if self._size > self._max_size:
            self._data.states.pop(0)
            self._data.actions.pop(0)
            self._data.next_states.pop(0)
            self._data.rewards.pop(0)
            self._data.terminal_flags.pop(0)

    def random_next_batch(self, batch_size):
        """
        Sample a random batch of transitions from the buffer.
        Args:
            batch_size (int): Number of samples to return.
        Returns:
            Tuple of arrays: (states, actions, next_states, rewards, terminal_flags)
        """
        batch_indices = self.rng.choice(len(self._data.states), batch_size)
        batch_states = np.array([self._data.states[i] for i in batch_indices])
        batch_actions = np.array([self._data.actions[i] for i in batch_indices])
        batch_next_states = np.array([self._data.next_states[i] for i in batch_indices])
        batch_rewards = np.array([self._data.rewards[i] for i in batch_indices])

        batch_terminal_flags = np.array([self._data.terminal_flags[i] for i in batch_indices])
        return (
            batch_states,
            batch_actions,
            batch_next_states,
            batch_rewards,
            batch_terminal_flags,
        )

    def save(self, path):
        """
        Save the replay buffer to disk.
        Args:
            path (str): Directory to save buffer file.
        """
        with open(os.path.join(path, "rpb.pkl"), "wb") as fh:
            pickle.dump(list(self._data), fh)

    def load(self, path):
        """
        Load the replay buffer from disk.
        Args:
            path (str): Directory to load buffer file from.
        """
        with open(os.path.join(path, "rpb.pkl"), "rb") as fh:
            data = pickle.load(fh)
        self._data = namedtuple(
            "ReplayBuffer",
            ["states", "actions", "next_states", "rewards", "terminal_flags"],
        )
        self._data.states = data[0]
        self._data.actions = data[1]
        self._data.next_states = data[2]
        self._data.rewards = data[3]
        self._data.terminal_flags = data[4]
        self._size = len(data[0])

    def get_reward_stats(self):
        """
        Get all rewards stored in the buffer.
        Returns:
            List of rewards.
        """
        return self._data.rewards


class DDQN:
    """
    Factored Deep Q-Network (DDQN) agent for multi-dimensional discrete action spaces.
    Handles training, evaluation, and model management.

    Main features:
    - Supports hard and soft target network updates
    - Dueling architecture for Q-network
    - Adaptive reward shifting for stability
    - Experience replay buffer for sample efficiency
    - Logging and evaluation utilities

    Key methods:
        __init__: Initializes agent, networks, buffer, environments, and logging.
        update_target_network: Updates target Q-network (hard/soft).
        tt: Converts numpy arrays to torch tensors on correct device.
        save_rpb/load_rpb: Save/load replay buffer to/from disk.
        act: Epsilon-greedy action selection.
        update_epsilon: Linear epsilon decay for exploration.
        train: Main training loop, including evaluation and logging.
        __repr__: String representation of the agent.
        save_model/load: Save/load Q-network weights.
        get_actions_for_all_states: Get greedy actions for all states of size n.
        test: Evaluate agent and plot results.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_actions: int,
        env: gym.Env,
        eval_env: gym.Env = None,
        extra_eval_env: gym.Env = None,
        out_dir: str = "outputs/checkpoints",
        gamma: float = 0.99,
        loss_function=F.mse_loss,
        lr: float = 1e-3,
        use_double_dqn=True,
        seed: int = 42,
        config: str = None,
        n_cpus=-1,
        bench_params: dict = {},
        eval_env_params: dict = {},
        use_dueling: bool = True,
        net_arch: list = [50, 50],
        target_update_type: str = "soft",  # "soft" or "hard"
        tau: float = 0.01,  # for soft update
        hard_update_freq: int = 1000,  # for hard update
    ):
        """
        Initialize the DQN Agent
        :param state_dim: dimensionality of the input states
        :param action_dim: dimensionality of the output actions
        :param gamma: discount factor
        :param env: environment to train on
        :param eval_env: environment to evaluate on
        :param extra_eval_env: another environment to evaluate on (optional)
        :param vision: boolean flag to indicate if the input state is an image or not
        """
        self.state_dim = state_dim
        self.device = torch.device("cpu")
        self._q = BranchingQNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            n_actions=n_actions,
            use_dueling=use_dueling,
            net_arch=net_arch,
        )
        self._q_target = BranchingQNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            n_actions=n_actions,
            use_dueling=use_dueling,
            net_arch=net_arch,
        )
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)
        torch.manual_seed(seed)
        self._gamma = gamma
        self._loss_function = loss_function
        self.lr = lr
        self._q_optimizer = optim.Adam(self._q.parameters(), lr=lr)
        self.use_double_dqn = use_double_dqn
        self.n_cpus = n_cpus
        self._replay_buffer = ReplayBuffer(1e6, self.rng)
        self._env = env
        self.eval_env = eval_env
        self._extra_eval_env = extra_eval_env
        self.bench_params = bench_params
        self.eval_env_params = eval_env_params
        date_str, time_str = get_time_str()
        cfg_name = os.path.basename(config).split(".")[0]

        self.out_dir = os.path.join(
            f"{out_dir}/{cfg_name}/gamma:{gamma}_dueling:{use_dueling}",
            f"{time_str}_seed_{seed}",
        )

        os.makedirs(self.out_dir, exist_ok=True)
        shutil.copyfile(
            config,
            os.path.join(self.out_dir, os.path.basename(config)),
        )
        self.config = object_to_dict(load_config(config))
        self.config["experiment"]["seed"] = seed
        self.config["agent"]["gamma"] = gamma
        self.logger = Logger(
            out_dir=self.out_dir,
        )

        self.runtime_logs = {"train": [], "eval": [], "forward": [], "episode": {}}
        self.shift_constant = None
        self.evaluator = DDQNFactEval(
            agent=self,
            obs_space=eval_env.observation_space.shape[0],
            n_eval_episodes_per_instance=self.config["experiment"]["eval_n_episodes"],
            log_path=self.out_dir,
            n_cpus=self.n_cpus,
        )
        self.target_update_type = target_update_type
        self.tau = tau
        self.hard_update_freq = hard_update_freq
        self._last_hard_update_step = 0

        self.base_q1_over_q3 = 16
        self.alpha = 1 / 12

    def update_target_network(self, total_steps):
        """
        Update the target network using either hard or soft update strategy.
        Args:
            total_steps (int): Current training step.
        """
        if self.target_update_type == "hard":
            # Only update every hard_update_freq steps
            if total_steps - self._last_hard_update_step >= self.hard_update_freq:
                hard_update(self._q_target, self._q)
                self._last_hard_update_step = total_steps
                print(f"Hard update at step {total_steps}")
        else:
            soft_update(self._q_target, self._q, self.tau)

    def tt(self, ndarray):
        """
        Convert numpy array to torch tensor and move to device.
        Args:
            ndarray (np.ndarray): Input array.
        Returns:
            torch.Tensor: Tensor on correct device.
        """
        tensor = torch.tensor(ndarray, dtype=torch.float32)
        return tensor.to(self.device)

    def save_rpb(self, path):
        """
        Save the replay buffer to disk.
        Args:
            path (str): Directory to save buffer file.
        """
        self._replay_buffer.save(path)

    def load_rpb(self, path):
        """
        Load the replay buffer from disk.
        Args:
            path (str): Directory to load buffer file from.
        """
        self._replay_buffer.load(path)

    def act(self, x: np.ndarray, epsilon: float = 0.0) -> int:
        """
        Select an action using epsilon-greedy policy based on observation x.
        Args:
            x (np.ndarray): Observation/state.
            epsilon (float): Exploration rate.
        Returns:
            int: Selected action.
        """
        x = self.tt(x)
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        u = torch.argmax(self._q(x), dim=2)
        u = u.squeeze(dim=0).numpy()
        r = self.rng.random()
        if r < epsilon:
            return self.rng.integers(
                low=0,
                high=len(self._env.action_choices[0][0]),
                size=len(self._env.action_choices[0]),
            )
        return u

    def update_epsilon(
        self,
        begin_learning_after,
        max_train_time_steps,
        epsilon_start,
        epsilon_decay_end_point,
        epsilon_decay_end_value,
        cur_total_steps,
    ) -> float:
        """
        Linearly decay epsilon value for exploration.
        Args:
            begin_learning_after (int): Steps before learning starts.
            max_train_time_steps (int): Max training steps.
            epsilon_start (float): Initial epsilon value.
            epsilon_decay_end_point (float): Fraction of training steps for decay.
            epsilon_decay_end_value (float): Final epsilon value.
            cur_total_steps (int): Current step count.
        Returns:
            float: Updated epsilon value.
        """
        end_step = (
            begin_learning_after
            + (max_train_time_steps - begin_learning_after) * epsilon_decay_end_point
        )

        if cur_total_steps >= end_step:
            return epsilon_decay_end_value

        return epsilon_start - (epsilon_start - epsilon_decay_end_value) * (
            cur_total_steps - begin_learning_after
        ) / (end_step - begin_learning_after)

    def train(
        self,
        episodes: int,
        max_env_time_steps: int = 1_000_000,
        epsilon: float = 0.2,
        epsilon_decay: bool = False,
        epsilon_decay_end_point: float = 0.5,  # ignored if epsilon_decay = False
        epsilon_decay_end_value: float = 0.05,  # ignored if epsilon_decay = False
        eval_every_n_steps: int = 1000,
        save_agent_at_every_eval: bool = True,
        max_train_time_steps: int = 1_000_000,
        begin_learning_after: int = 10_000,
        batch_size: int = 2_048,
        log_level=1,
    ):
        """
        Main training loop for DDQN agent.
        Args:
            episodes (int): Number of episodes to train.
            max_env_time_steps (int): Max steps per episode.
            epsilon (float): Initial exploration rate.
            epsilon_decay (bool): Whether to decay epsilon.
            epsilon_decay_end_point (float): Fraction of training steps for decay.
            epsilon_decay_end_value (float): Final epsilon value.
            eval_every_n_steps (int): Evaluation interval (steps).
            save_agent_at_every_eval (bool): Save agent at each eval.
            max_train_time_steps (int): Max training steps.
            begin_learning_after (int): Steps before learning starts.
            batch_size (int): Training batch size.
            log_level (int): Logging verbosity.
        """
        start_training_time = time.time()
        train_args = {
            "episodes": episodes,
            "max_env_time_steps": max_env_time_steps,
            "epsilon": epsilon,
            "epsilon_decay": epsilon_decay,
            "epsilon_decay_end_point": epsilon_decay_end_point,
            "epsilon_decay_end_value": epsilon_decay_end_value,
            "eval_every_n_steps": eval_every_n_steps,
            "save_agent_at_every_eval": save_agent_at_every_eval,
            "max_train_time_steps": max_train_time_steps,
            "begin_learning_after": begin_learning_after,
            "batch_size": batch_size,
            "log_level": log_level,
        }
        self.config["train_args"] = train_args
        with open(os.path.join(self.out_dir, "train_args.json"), "w") as fh:
            json.dump(self.config, fh, indent=4)

        total_steps = 0
        pre_total_steps = 0

        s = self._env.get_state()

        epsilon_start = epsilon
        pbar = tqdm(total=max_train_time_steps, desc="Training Progress")

        total_rewards = []
        for episode in range(episodes):
            ep_losses = []
            reward_sum = 0
            for t in range(max_env_time_steps):
                if epsilon_decay:
                    epsilon = self.update_epsilon(
                        begin_learning_after,
                        max_train_time_steps,
                        epsilon_start,
                        epsilon_decay_end_point,
                        epsilon_decay_end_value,
                        total_steps,
                    )

                a = self.act(s, epsilon if total_steps > begin_learning_after else 1.0)
                ns, r, tr, d, _ = self._env.step(
                    actions=a,
                    shift=self.shift_constant if self.shift_constant else 0,
                )

                total_steps += 1
                pbar.update(1)
                reward_sum += r

                if total_steps == begin_learning_after:
                    rewards = self._replay_buffer.get_reward_stats()
                    q1 = np.percentile(rewards, 25)
                    q3 = np.percentile(rewards, 75)
                    q1_over_q3 = q1 / q3
                    self.shift_constant = (
                        -abs(np.mean(rewards)) * (q1_over_q3 / self.base_q1_over_q3) * self.alpha
                    )

                if total_steps % eval_every_n_steps == 0:
                    eval_step_runtime_start = time.time()
                    self.evaluator.eval(n_steps=total_steps)
                    self.runtime_logs["eval"].append(
                        {
                            "step": total_steps,
                            "episode": episode,
                            "step_runtime": time.time() - eval_step_runtime_start,
                        }
                    )

                # Update replay buffer
                self._replay_buffer.add_transition(s, a, ns, r, d)
                if total_steps > begin_learning_after:
                    data_batch = self._replay_buffer.random_next_batch(batch_size)
                    train_step_runtime_start = time.time()
                    (
                        batch_states,
                        batch_actions,
                        batch_next_states,
                        batch_rewards,
                        batch_terminal_flags,
                    ) = (
                        self.tt(data_batch[0]),
                        self.tt(data_batch[1]),
                        self.tt(data_batch[2]),
                        self.tt(data_batch[3]),
                        self.tt(data_batch[4]),
                    )
                    ## for MP
                    batch_rewards = batch_rewards.unsqueeze(1)
                    batch_terminal_flags = batch_terminal_flags.unsqueeze(1)
                    argmax = torch.argmax(self._q(batch_next_states), dim=2)
                    max_next_q_vals = (
                        self._q_target(batch_next_states)
                        .gather(2, argmax.unsqueeze(2))
                        .squeeze(-1)
                    )
                    target = (
                        batch_rewards + (1 - batch_terminal_flags) * self._gamma * max_next_q_vals
                    )
                    current_prediction = (
                        self._q(batch_states)
                        .gather(2, batch_actions.long().unsqueeze(2))
                        .squeeze(-1)
                    )
                    loss = self._loss_function(current_prediction, target.detach())

                    ep_losses.append(loss.item())

                    self._q_optimizer.zero_grad()
                    loss.backward()
                    self._q_optimizer.step()

                    self.update_target_network(total_steps)

                    self.runtime_logs["train"].append(
                        {
                            "step": total_steps,
                            "episode": episode,
                            "step_runtime": time.time() - train_step_runtime_start,
                        }
                    )
                if d:
                    diff_steps = total_steps - pre_total_steps
                    self.runtime_logs["episode"][episode] = diff_steps
                    pre_total_steps = total_steps
                    break
                s = ns
                if total_steps >= max_train_time_steps:
                    break
                if total_steps % eval_every_n_steps == 0:
                    if ep_losses:
                        avg_loss = np.mean(ep_losses)
                        self.logger.log_scalar(tag="Loss/step", value=avg_loss, step=total_steps)

            s, _ = self._env.reset()

            total_rewards.append(reward_sum)

            if total_steps >= max_train_time_steps:
                break

            # Log metrics at the end of each episode
            if ep_losses:
                avg_loss = np.mean(ep_losses)
                self.logger.log_scalar(tag="Loss/episode", value=avg_loss, step=episode)
        # dump runtime logs
        end_training_time = time.time()
        self.runtime_logs["total"] = end_training_time - start_training_time

        self.test(
            ckpt_dir=self.out_dir,
            verbose=False,
            total_steps=total_steps,
            topk=5,
        )
        # Close the logger
        self.logger.close()

    def __repr__(self):
        """
        String representation of the agent.
        Returns:
            str: Agent name.
        """
        return "factored_ddqn"

    def save_model(self, model_path):
        """
        Save the Q-network model to disk.
        Args:
            model_path (str): Path to save the model.
        """
        torch.save(self._q.state_dict(), os.path.join(model_path + ".pt"))

    def load(self, q_path):
        """
        Load Q-network model from disk.
        Args:
            q_path (str): Path to load the model from.
        """
        self._q.load_state_dict(torch.load(q_path))

    def get_actions_for_all_states(self, n):
        """
        Get actions for all possible states of size n.
        Args:
            n (int): Number of states.
        Returns:
            List: Actions for each state.
        """
        start_time = time.time()
        with torch.no_grad():
            all_states = self.tt(np.array([[n, fx] for fx in range(0, n)]))
            q_values = self._q(all_states)
            acts = q_values.argmax(dim=2).cpu().numpy().tolist()
        self.runtime_logs["forward"].append(
            {
                "step_runtime": time.time() - start_time,
            }
        )
        return acts

    def test(
        self,
        ckpt_dir: str,
        verbose: bool = False,
        total_steps: int = 1_000_000,
        topk: int = 5,
        n_cpus: int = 8,
    ):
        """
        Evaluate the agent using saved checkpoints and plot results.
        Args:
            ckpt_dir (str): Directory containing checkpoints and evaluation data.
            verbose (bool): Whether to print verbose output.
            total_steps (int): Total training steps.
            topk (int): Number of top policies to evaluate.
            n_cpus (int): Number of CPUs for parallel evaluation.
        """
        ## load evaluation data
        evaluations_fpath = os.path.join(ckpt_dir, "evaluations.npz")
        eval_data = np.load(evaluations_fpath, allow_pickle=True)
        eval_policies = np.array(
            eval_data["eval_policies"]
        )  # shape (total_steps//eval_interval, instance_idx, policy)
        instance_set = eval_data["instance_set"].item()
        i = 0
        inst_id = eval_data["inst_ids"][i]
        instance = instance_set[inst_id]
        n = instance["size"]
        eval_runtime_means = np.array([ls[0] for ls in eval_data["eval_runtime_means"]])
        eval_timesteps = np.array(eval_data["eval_timesteps"])
        top_k_min_indices = np.argsort(eval_runtime_means)[:topk]

        runtime_means = []
        runtime_stds = []
        policies = []
        steps = []
        save_fname = os.path.join(ckpt_dir, "evaluations_last.npz")
        eval_runtimes = []
        for step in top_k_min_indices:
            policy = eval_policies[step][0]
            runtimes = Parallel(n_jobs=n_cpus)(
                delayed(ollga_mp_single_run)(self.bench_params, self.eval_env_params, policy, i)
                for i in tqdm(
                    range(1000),
                    desc=f"[Test Stage]: Progress Policy @ {step}-th",
                    disable=False,
                )
            )
            eval_runtimes.append(runtimes)
            runtime_means.append(np.mean(runtimes))
            runtime_stds.append(np.std(runtimes))
            policies.append(policy)
            steps.append(step)

        np.savez(
            file=save_fname,
            n_steps=steps,
            eval_policies=policies,
            eval_runtime_means=runtime_means,
            eval_runtime_stds=runtime_stds,
            eval_runtimes=eval_runtimes,
        )

        eval_charts = plot_results(
            results_fpath=os.path.join(ckpt_dir, "evaluations.npz"),
            off_env_eval=True,
            verbose=verbose,
        )

        self.logger.log_figure(
            tag="Policy_Comparison",
            figure=eval_charts[0],
            step=total_steps,
            out_dir=os.path.join(ckpt_dir),
        )
        self.logger.log_figure(
            tag="Learning_Curve",
            figure=eval_charts[1],
            step=total_steps,
            out_dir=os.path.join(ckpt_dir),
        )


def main():
    """
    Main entry point for training DDQN agent from command line arguments.
    Parses arguments, sets up environments, and starts training.
    """
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", "-o", type=str, default="output", help="output folder")
    parser.add_argument("--config-file", "-c", type=str, help="yml file with all configs")
    parser.add_argument("--seed", "-s", type=int, help="seed for reproducibility", default=123)
    parser.add_argument("--n-cpus", "-n", type=int, help="number of used CPUs", default=1)
    parser.add_argument("--gamma", "-g", type=float, help="discount factor", default=0.99)
    parser.add_argument(
        "--use-dueling",
        "-d",
        action="store_true",
        help="use dueling architecture",
    )
    parser.add_argument(
        "--target-update-type",
        type=str,
        choices=["hard", "soft"],
        default="soft",
        help="type of target network update",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.01,
        help="soft update coefficient for target network",
    )
    parser.add_argument(
        "--hard-update-freq",
        type=int,
        default=1000,
        help="frequency of hard updates for target network",
    )

    args = parser.parse_args()

    config_yml_fn = args.config_file
    (
        exp_params,
        bench_params,
        agent_params,
        train_env_params,
        eval_env_params,
    ) = read_config(config_yml_fn)
    if exp_params["n_cores"] > 1:
        print("WARNING: n_cores>1 is not yet supported")

    # create output folder
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    # create train_env and eval_env
    train_env = make_env(bench_params, train_env_params)
    eval_env = make_env(bench_params, eval_env_params)
    if isinstance(train_env.action_space, gym.spaces.Discrete):
        action_dim = train_env.action_space.n
    else:
        action_dim = train_env.action_space.shape[0]
    state_dim = len(train_env.reset())
    # get loss function
    assert agent_params["loss_function"] in ["mse_loss", "smooth_l1_loss"]
    seed_everything(args.seed)
    if agent_params["name"] == "ddqn":
        agent_class = DDQN
    else:
        raise ValueError(f"Sorry, agent {agent_params['name']} is not yet supported")

    agent_params["gamma"] = args.gamma
    agent = agent_class(
        state_dim=state_dim,
        action_dim=action_dim,
        n_actions=len(bench_params["action_choices"]),
        env=train_env,
        eval_env=eval_env,
        out_dir=out_dir,
        gamma=agent_params["gamma"],
        lr=agent_params["lr"],
        loss_function=getattr(F, agent_params["loss_function"]),
        seed=args.seed,
        config=config_yml_fn,
        n_cpus=args.n_cpus,
        bench_params=bench_params,
        eval_env_params=eval_env_params,
        use_dueling=args.use_dueling,
        net_arch=agent_params["net_arch"],
        target_update_type=args.target_update_type,
        tau=args.tau,
        hard_update_freq=args.hard_update_freq,
    )
    agent.train(
        episodes=exp_params["n_episodes"],
        max_env_time_steps=int(1e9),
        epsilon=agent_params["epsilon"],
        eval_every_n_steps=exp_params["eval_interval"],
        save_agent_at_every_eval=exp_params["save_agent_at_every_eval"],
        max_train_time_steps=exp_params["n_steps"],
        begin_learning_after=agent_params["begin_learning_after"],
        batch_size=agent_params["batch_size"],
        log_level=exp_params["log_level"],
    )

    total_time = time.time() - start_time
    print(f"Total runtime: {total_time}")


if __name__ == "__main__":
    # Set deterministic algorithms
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_default_dtype(torch.float32)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    main()
