import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, hidden_sizes=[64,64]):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_sizes (list[int]): List of hidden layer sizes
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        # First hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(state_size, hidden_sizes[0])])
        
        # Hidden layers in the middle
        self.hidden_layers.extend([nn.Linear(in_size, out_size) for in_size, out_size in zip(hidden_sizes[:-1], hidden_sizes[1:])])
        
        # Output layer
        self.output_layer = nn.Linear(hidden_sizes[-1], action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = state
        for linear in self.hidden_layers:
            x = linear(x)
            x = F.relu(x)

        return self.output_layer(x)
