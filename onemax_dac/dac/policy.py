import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import create_mlp


class QNetwork(nn.Module):
    """
    Action-Value (Q-Value) network for DQN
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        activation_fn=nn.ReLU,
        net_arch: list = [50, 50],
    ):
        """
        :param state_dim: Dimension of state space
        :param action_dim: Dimension of action space
        :param activation_fn: Activation function
        :param net_arch: Network architecture
        """
        super(QNetwork, self).__init__()
        q_net = create_mlp(
            input_dim=state_dim,
            output_dim=action_dim,
            net_arch=net_arch,
            activation_fn=activation_fn,
        )
        self.q_net = nn.Sequential(*q_net)

    def forward(self, x):
        """
        Predict the q-values.

        :param obs: Observation
        :return: The estimated Q-Value for each action.
        """
        return self.q_net(x)


if __name__ == "__main__":
    # Test QNetwork
    state_dim = 2
    action_dim = 5
    net = QNetwork(state_dim, action_dim)
    print(net)
    # Test forward
    obs = torch.rand(1, state_dim)
    print(net(obs))
