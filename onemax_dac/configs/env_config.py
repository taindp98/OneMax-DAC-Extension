from typing import List


class EnvConfig:
    def __init__(
        self,
        problem_size: int,
        state_dim: int,
        discrete_action: bool = True,
        action_choices: list = [],
        reward_choice: str = "original",
        seed: int = 0,
        init_obj_rate=0.5,
        **kwargs,
    ):
        self.problem_size = problem_size
        self.state_dim = state_dim
        self.discrete_action = discrete_action
        self.action_choices = action_choices
        self.reward_choice = reward_choice
        self.seed = seed
        self.init_obj_rate = init_obj_rate
        self.kwargs = kwargs

    def to_dict(self):
        """Convert the class attributes to a dictionary."""
        return self.__dict__
