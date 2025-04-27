import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from collections import deque
from .networks import PolicyNetwork, ValueNetwork

# Remove duplicate network definitions since we're importing from networks.py
# class PolicyNetwork(nn.Module):...
# class ValueNetwork(nn.Module):...

class PPOAgent:
    def __init__(
        self,
        state_dim,
        learning_rate=3e-4,
        gamma=0.99,
        epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        batch_size=64,
        epochs=10,
        buffer_size=2048
    ):
        self.policy = PolicyNetwork(state_dim)
        self.value = ValueNetwork(state_dim)
        
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=learning_rate)
        
        # Hyperparameters
        self.gamma = gamma              # Discount factor
        self.epsilon = epsilon          # PPO clipping parameter
        self.value_coef = value_coef   # Value loss coefficient
        self.entropy_coef = entropy_coef  # Entropy coefficient
        self.max_grad_norm = max_grad_norm  # Gradient clipping
        self.batch_size = batch_size
        self.epochs = epochs
        
        # Experience buffer
        self.buffer_size = buffer_size
        self.reset_buffers()
        
        # Training metrics
        self.metrics = {
            'policy_losses': [],
            'value_losses': [],
            'entropy_losses': [],
            'total_losses': [],
            'mean_rewards': []
        }
    
    def reset_buffers(self):
        """Reset experience buffers."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def select_action(self, state, deterministic=False):
        """Select an action using the policy network."""
        with torch.no_grad():
            action, log_prob = self.policy.get_action(state, deterministic)
            value = self.value(torch.FloatTensor(state))
        
        if not deterministic:
            self.states.append(state)
            self.actions.append(action)
            self.log_probs.append(log_prob)
            self.values.append(value)
        
        return action
    
    def store_transition(self, reward, done):
        """Store reward and done flag for the last transition."""
        self.rewards.append(reward)
        self.dones.append(done)
    
    def compute_advantages(self):
        """Compute advantages using Generalized Advantage Estimation (GAE)."""
        rewards = torch.FloatTensor(self.rewards)
        values = torch.cat(self.values)
        dones = torch.FloatTensor(self.dones)
        
        # Calculate advantages
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = delta + self.gamma * 0.95 * (1 - dones[t]) * last_advantage
            last_advantage = advantages[t]
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages
    
    def update(self):
        """Update policy and value networks using PPO."""
        if len(self.states) < self.batch_size:
            return
        
        # Convert buffers to tensors
        states = torch.FloatTensor(np.array(self.states))
        actions = torch.FloatTensor(np.array(self.actions))
        old_log_probs = torch.cat(self.log_probs)
        advantages = self.compute_advantages()
        returns = advantages + torch.cat(self.values)
        
        # PPO update for specified number of epochs
        for _ in range(self.epochs):
            # Generate random mini-batches
            indices = np.random.permutation(len(self.states))
            
            for start_idx in range(0, len(self.states), self.batch_size):
                # Get mini-batch
                batch_indices = indices[start_idx:start_idx + self.batch_size]
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Get current policy distribution
                mean, std = self.policy(batch_states)
                dist = torch.distributions.Normal(mean, std)
                curr_log_probs = dist.log_prob(batch_actions)
                
                # Calculate ratio and surrogate loss
                ratio = torch.exp(curr_log_probs - batch_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Calculate value loss
                value_pred = self.value(batch_states)
                value_loss = F.mse_loss(value_pred, batch_returns.unsqueeze(1))
                
                # Calculate entropy loss for exploration
                entropy_loss = -dist.entropy().mean()
                
                # Total loss
                total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Update networks
                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.value.parameters(), self.max_grad_norm)
                
                self.policy_optimizer.step()
                self.value_optimizer.step()
                
                # Store metrics
                self.metrics['policy_losses'].append(policy_loss.item())
                self.metrics['value_losses'].append(value_loss.item())
                self.metrics['entropy_losses'].append(entropy_loss.item())
                self.metrics['total_losses'].append(total_loss.item())
        
        # Clear buffers after update
        self.reset_buffers()
        
        return {
            'policy_loss': np.mean(self.metrics['policy_losses'][-10:]),
            'value_loss': np.mean(self.metrics['value_losses'][-10:]),
            'entropy_loss': np.mean(self.metrics['entropy_losses'][-10:]),
            'total_loss': np.mean(self.metrics['total_losses'][-10:])
        }
    
    def get_metrics(self):
        """Get current training metrics."""
        return {
            'policy_loss': np.mean(self.metrics['policy_losses'][-100:]) if self.metrics['policy_losses'] else 0,
            'value_loss': np.mean(self.metrics['value_losses'][-100:]) if self.metrics['value_losses'] else 0,
            'entropy_loss': np.mean(self.metrics['entropy_losses'][-100:]) if self.metrics['entropy_losses'] else 0,
            'total_loss': np.mean(self.metrics['total_losses'][-100:]) if self.metrics['total_losses'] else 0,
            'mean_reward': np.mean(self.metrics['mean_rewards'][-100:]) if self.metrics['mean_rewards'] else 0
        }

    # Unused methods - commenting out
    """
    def get_state_representation(self, interview_state):
        # This method is not used as we have _get_observation in InterviewAgent
        pass

    def get_next_question_params(self, interview_state):
        # This method is not used as we have adjust_difficulty in InterviewAgent
        pass
    
    def select_next_question(self, interview_state, available_questions):
        # This method is not used as we have adjust_difficulty in InterviewAgent
        pass

    def get_state(self):
        # This method is not used as we have _get_observation in InterviewAgent
        pass

    def calculate_time_trend(self):
        # This method is not used
        pass

    def calculate_performance_trend(self):
        # This method is not used
        pass
    """ 