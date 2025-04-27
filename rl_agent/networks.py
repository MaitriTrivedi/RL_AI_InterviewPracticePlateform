import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, 1)  # Output mean for Gaussian policy
        self.fc_std = nn.Linear(hidden_dim, 1)   # Output std for Gaussian policy
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        # Ensure standard deviation is positive and not too small
        std = F.softplus(self.fc_std(x)) + 1e-3
        return mean, std
    
    def get_action(self, state, deterministic=False):
        """Sample an action from the policy."""
        state = torch.FloatTensor(state)
        mean, std = self.forward(state)
        
        if deterministic:
            # During evaluation, use the mean action
            action = mean
        else:
            # During training, sample from the normal distribution
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
        
        # Clip action to valid range [1, 10] for difficulty
        action = torch.clamp(action, 1.0, 10.0)
        
        log_prob = None
        if not deterministic:
            # Compute log probability of the sampled action
            dist = torch.distributions.Normal(mean, std)
            log_prob = dist.log_prob(action)
        
        return action.detach().numpy(), log_prob

class ValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value 