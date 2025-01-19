from onemax_dac.dac import ReplayBuffer
from onemax_dac.theory_env import OneMax
from torch import Tensor, nn
import numpy as np
import torch
from typing import Tuple
from onemax_dac.dac.utils import to_tensor


class Agent:
    def __init__(
        self, env: OneMax, replay_buffer: ReplayBuffer, rng=np.random.default_rng()
    ) -> None:
        """Base Agent class handling the interaction with the environment.

        Args:
            env: training environment
            replay_buffer: replay buffer storing experiences
            rng: random number generator
        """
        self.env = env
        self.replay_buffer = replay_buffer
        self.state, _ = self.env.reset()
        self.rng = rng
        self.total_episodes = 0

    def get_action(self, x: np.ndarray, net, epsilon, device=torch.device("cpu")) -> int:
        """
        Simple helper to get action epsilon-greedy based on observation x
        """
        u = torch.argmax(net(to_tensor(x, device))).item()
        r = self.rng.random()
        if r < epsilon:
            return self.rng.integers(low=0, high=self.env.action_dim)
        return u

    @torch.no_grad()
    def play_step(
        self,
        net: nn.Module,
        shift: int = 0,
        epsilon: float = 0.0,
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[float, bool]:
        """Carries out a single interaction step between the agent and the environment.

        Args:
            net: DQN network
            epsilon: value to determine likelihood of taking a random action

        Returns:
            reward, done

        """
        action = self.get_action(self.state, net, epsilon, device)
        # do step in the environment
        # So, in the deprecated version of gym, the env.step() has 4 values unpacked which is
        #     obs, reward, done, info = env.step(action)
        # In the latest version of gym, the step() function returns back an additional variable which is truncated.
        #     obs, reward, terminated, truncated, info = env.step(action)
        new_state, reward, _, done, _ = self.env.step(
            action_index=action,
            shift=shift,
        )

        self.replay_buffer.add_transition(self.state, action, new_state, reward, done)
        self.state = new_state
        if done:
            self.state, _ = self.env.reset()
            self.total_episodes += 1
        return reward, done
