import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, hidden_sizes, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        # Hyperparameters for our network
        # hidden_sizes = [64, 64]
        self.fcl1 = nn.Linear(int(state_size), hidden_sizes[0])
        self.relu1 = nn.ReLU()
        self.fcl2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.relu2 = nn.ReLU()
        self.actionl = nn.Linear(hidden_sizes[1], int(action_size))


    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = self.relu1(self.fcl1(state))
        x = self.relu2(self.fcl2(x))
        return self.actionl(x.view(x.size(0), -1))
        pass
