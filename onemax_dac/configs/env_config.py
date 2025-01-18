from typing import List
class EnvConfig:
    def __init__(
            self,
            problem_size: int,
            state_dim: int,
            discrete_action: bool = True,
            action_choices: list = [],
            reward_choice: str = "imp_minus_evals",
            seed: int = 0,
            init_obj_rate: float = 0.5,
            **kwargs
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
    
    def __repr__(self):
        return f"EnvConfig(problem_size={self.problem_size}, discrete_action={self.discrete_action}, action_choices={self.action_choices}, reward_choice={self.reward_choice}, seed={self.seed}, init_obj_rate={self.init_obj_rate}, kwargs={self.kwargs})"