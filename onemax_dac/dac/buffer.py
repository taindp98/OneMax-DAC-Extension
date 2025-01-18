import os
import pickle
from collections import namedtuple
from typing import List
import numpy as np

class ReplayBuffer:
    def __init__(self, max_size, rng=np.random.default_rng()):
        """
        Replay Buffer for storing the transitions
        Args:
            - max_size: maximum size of the buffer
            - rng: random number generator
        """
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
        Add a transition to the replay buffer
        Args:
            - state: current state
            - action: action taken
            - next_state: next state
            - reward: reward received
            - done: terminal flag
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
        Get a random batch of transitions
        Args:
            - batch_size: size of the batch
        """
        batch_indices = self.rng.choice(len(self._data.states), batch_size)
        batch_states = np.array([self._data.states[i] for i in batch_indices])
        batch_actions = np.array([self._data.actions[i] for i in batch_indices])
        batch_next_states = np.array([self._data.next_states[i] for i in batch_indices])
        batch_rewards = np.array([self._data.rewards[i] for i in batch_indices])
        batch_terminal_flags = np.array(
            [self._data.terminal_flags[i] for i in batch_indices]
        )
        return (
            batch_states,
            batch_actions,
            batch_next_states,
            batch_rewards,
            batch_terminal_flags,
        )

    def save(self, path):
        with open(os.path.join(path, "rpb.pkl"), "wb") as fh:
            pickle.dump(list(self._data), fh)

    def load(self, path):
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

    def get_reward_stats(self, mode="mean"):
        """
        Get the reward statistics
        Args:
            - mode: mode of statistics (mean or median)
        """
        if mode == "mean":
            return abs(np.mean(self._data.rewards))
        elif mode == "median":
            return abs(np.median(self._data.rewards))
        else:
            raise ValueError("Mode not recognised")
