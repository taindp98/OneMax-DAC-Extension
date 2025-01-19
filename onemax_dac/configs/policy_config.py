from typing import List
class PolicyConfig:
    def __init__(
        self, policy_name: str = "DDQN",
        net_arch: List[int] = [50, 50],
        activation_fn: str = "ReLU",
    ):
        self.policy_name = policy_name
        self.net_arch = net_arch
        self.activation_fn = activation_fn

    def to_dict(self):
        """Convert the class attributes to a dictionary."""
        return self.__dict__