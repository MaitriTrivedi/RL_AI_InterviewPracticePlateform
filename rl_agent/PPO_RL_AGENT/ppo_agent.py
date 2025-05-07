import numpy as np
import torch
from collections import deque
from .neural_network import GaussianPolicy, ValueNetwork, compute_gae
import logging
import os

class PPOAgent:
    def __init__(self, state_dim=9, action_dim=1, hidden_dim=64, 
                 lr_actor=0.0003, lr_critic=0.001, gamma=0.99, 
                 gae_lambda=0.95, clip_ratio=0.2, target_kl=0.01,
                 train_actor_iterations=10, train_critic_iterations=10):
        """Initialize PPO agent with enhanced parameters."""
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Initialize neural networks with enhanced architecture
        self.policy = GaussianPolicy(state_dim, action_dim, hidden_dim)
        self.value_net = ValueNetwork(state_dim, hidden_dim)
        
        # Training parameters with improved defaults
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.train_actor_iterations = train_actor_iterations
        self.train_critic_iterations = train_critic_iterations
        
        # Enhanced metrics tracking
        self.metrics = {
            'actor_loss': deque(maxlen=100),
            'value_loss': deque(maxlen=100),
            'entropy_loss': deque(maxlen=100),
            'total_loss': deque(maxlen=100),
            'mean_reward': deque(maxlen=100),
            'clip_fraction': deque(maxlen=100),
            'approx_kl': deque(maxlen=100),
            'difficulty_accuracy': deque(maxlen=100),
            'topic_diversity': deque(maxlen=100)
        }
        
        # Clear memory buffers
        self.clear_memory()
    
    def clear_memory(self):
        """Clear memory buffers."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def select_action(self, state):
        """Select action with enhanced state processing and stability."""
        try:
            # Ensure state is properly shaped and normalized
            state = np.array(state, dtype=np.float32)
            if len(state.shape) == 1:
                state = state.reshape(1, -1)
            state = np.clip(state, -10.0, 10.0)  # Clip extreme values
            
            # Get action distribution parameters
            mean, std = self.policy.forward(state)
            
            # Ensure mean and std are properly shaped
            if isinstance(mean, np.ndarray) and len(mean.shape) > 1:
                mean = mean.flatten()[0]
            if isinstance(std, np.ndarray) and len(std.shape) > 1:
                std = std.flatten()[0]
                
            # Ensure they are scalar values
            mean = float(mean)
            std = float(np.clip(std, 0.1, 1.0))  # Prevent too small/large standard deviations
            
            # Sample action with controlled randomness
            noise = float(np.random.normal(0, std))
            action = mean + noise
            
            # Clip action to valid range [-1, 1]
            action = float(np.clip(action, -1.0, 1.0))
            
            # Scale action to difficulty range [1, 10]
            scaled_action = float(1.0 + (action + 1.0) * 4.5)  # Maps [-1,1] to [1,10]
            scaled_action = float(np.clip(scaled_action, 1.0, 10.0))
            
            # Compute log probability with numerical stability
            log_prob = float(-0.5 * (((action - mean) / (std + 1e-8)) ** 2 + 2 * np.log(std + 1e-8) + np.log(2 * np.pi)))
            
            # Get value estimate
            value = float(self.value_net.forward(state).flatten()[0])
            value = float(np.clip(value, -10.0, 10.0))
            
            # Store in memory if values are valid
            if not (np.isnan(scaled_action) or np.isnan(value) or np.isnan(log_prob)):
                self.states.append(state)
                self.actions.append(action)  # Store original action, not scaled
                self.log_probs.append(log_prob)
                self.values.append(value)
            
            # Return tuple of (scaled_action, value, log_prob)
            return (float(scaled_action), float(value), float(log_prob))
            
        except Exception as e:
            logging.error(f"Error in select_action: {e}")
            return (5.0, 0.0, 0.0)  # Safe fallback to medium difficulty
    
    def store_transition(self, reward, done):
        """Store transition information."""
        self.rewards.append(reward)
        self.dones.append(done)
    
    def _calculate_advantages(self, rewards, values, dones):
        """Calculate advantages using GAE with enhanced stability."""
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        running_return = 0
        running_advantage = 0
        
        # Reverse iteration for proper bootstrapping
        for t in reversed(range(len(rewards))):
            if dones[t]:
                running_return = 0
                running_advantage = 0
                
            running_return = rewards[t] + self.gamma * running_return * (1 - dones[t])
            if t < len(rewards) - 1:
                delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            else:
                delta = rewards[t] - values[t]
                
            running_advantage = delta + self.gamma * self.gae_lambda * running_advantage * (1 - dones[t])
            
            advantages[t] = running_advantage
            returns[t] = running_return
            
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns

    def train(self):
        """Train policy and value networks with enhanced stability."""
        try:
            if len(self.states) < 32:  # Require minimum batch size
                return None
            
            # Prepare and validate training data
            states = np.vstack(self.states)
            actions = np.vstack(self.actions)
            old_log_probs = np.array(self.log_probs).reshape(-1, 1)
            values = np.array(self.values).reshape(-1, 1)
            rewards = np.array(self.rewards).reshape(-1, 1)
            dones = np.array(self.dones).reshape(-1, 1)
            
            # Compute advantages and returns
            advantages, returns = self._calculate_advantages(rewards, values, dones)
            
            # Training metrics
            policy_losses = []
            value_losses = []
            entropies = []
            
            # Process data in batches
            batch_size = 32
            num_batches = len(states) // batch_size
            indices = np.arange(len(states))
            
            # Calculate training progress for entropy decay
            mean_rewards = [float(r) for r in list(self.metrics['mean_reward'])]
            training_progress = min(1.0, len(mean_rewards) / 1000) if mean_rewards else 0.0
            entropy_coef = 0.01 * (1.0 - training_progress)  # Decay from 0.01 to 0
            
            for _ in range(self.train_actor_iterations):
                np.random.shuffle(indices)  # Shuffle data for each iteration
                epoch_policy_losses = []
                epoch_entropies = []
                
                for batch_idx in range(num_batches):
                    batch_indices = indices[batch_idx * batch_size:(batch_idx + 1) * batch_size]
                    
                    # Get batch data
                    states_batch = states[batch_indices]
                    actions_batch = actions[batch_indices]
                    advantages_batch = advantages[batch_indices]
                    old_log_probs_batch = old_log_probs[batch_indices]
                    
                    # Policy loss
                    mean, std = self.policy.forward(states_batch)
                    new_log_probs = -0.5 * (((actions_batch - mean) / (std + 1e-8)) ** 2 + 2 * np.log(std) + np.log(2 * np.pi))
                    new_log_probs = np.sum(new_log_probs, axis=1).reshape(-1, 1)
                    
                    # Compute ratios and surrogate losses
                    ratios = np.exp(new_log_probs - old_log_probs_batch)
                    ratios = np.clip(ratios, 0.1, 10.0)  # Prevent extreme ratios
                    
                    surr1 = ratios * advantages_batch
                    surr2 = np.clip(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages_batch
                    policy_loss = -np.mean(np.minimum(surr1, surr2))
                    
                    # Entropy bonus for exploration with decay
                    entropy = 0.5 * (np.log(2 * np.pi * std ** 2) + 1)
                    entropy_loss = -entropy_coef * np.mean(entropy)
                    
                    # Update policy with gradient clipping
                    total_policy_loss = policy_loss + entropy_loss
                    gradients = self.policy.get_gradients(states_batch, advantages_batch)
                    
                    # Clip each gradient component
                    for key in gradients:
                        gradients[key] = np.clip(gradients[key], -1.0, 1.0)
                    
                    self.policy.apply_gradients(gradients, self.lr_actor)
                    epoch_policy_losses.append(float(policy_loss))
                    epoch_entropies.append(float(entropy_loss))
                
                policy_losses.append(np.mean(epoch_policy_losses))
                entropies.append(np.mean(epoch_entropies))
            
            # Value network update with gradient clipping
            for _ in range(self.train_critic_iterations):
                np.random.shuffle(indices)
                epoch_value_losses = []
                
                for batch_idx in range(num_batches):
                    batch_indices = indices[batch_idx * batch_size:(batch_idx + 1) * batch_size]
                    
                    # Get batch data
                    states_batch = states[batch_indices]
                    returns_batch = returns[batch_indices]
                    
                    values_pred = self.value_net.forward(states_batch)
                    value_loss = np.mean((returns_batch - values_pred) ** 2)
                    gradients = self.value_net.get_gradients(states_batch, returns_batch - values_pred)
                    
                    # Clip each gradient component
                    for key in gradients:
                        gradients[key] = np.clip(gradients[key], -1.0, 1.0)
                    
                    self.value_net.apply_gradients(gradients, self.lr_critic)
                    epoch_value_losses.append(float(value_loss))
                
                value_losses.append(np.mean(epoch_value_losses))
            
            # Calculate mean metrics
            mean_policy_loss = float(np.mean(policy_losses))
            mean_value_loss = float(np.mean(value_losses))
            mean_entropy_loss = float(np.mean(entropies))
            mean_reward = float(np.mean(rewards))
            
            # Update metrics history
            self.metrics['actor_loss'].append(mean_policy_loss)
            self.metrics['value_loss'].append(mean_value_loss)
            self.metrics['entropy_loss'].append(mean_entropy_loss)
            self.metrics['mean_reward'].append(mean_reward)
            
            # Clear memory after successful training
            self.clear_memory()
            
            # Return metrics dictionary
            return {
                'actor_loss': mean_policy_loss,
                'value_loss': mean_value_loss,
                'entropy_loss': mean_entropy_loss,
                'mean_reward': mean_reward
            }
            
        except Exception as e:
            logging.error(f"Error in train: {e}")
            self.clear_memory()
            return None
    
    def save_checkpoint(self, path):
        """Save neural network weights."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            checkpoint = {
                'policy': {
                    'W1': self.policy.W1,
                    'b1': self.policy.b1,
                    'W2': self.policy.W2,
                    'b2': self.policy.b2,
                    'W_mean': self.policy.W_mean,
                    'b_mean': self.policy.b_mean,
                    'W_std': self.policy.W_std,
                    'b_std': self.policy.b_std
                },
                'value_net': {
                    'W1': self.value_net.W1,
                    'b1': self.value_net.b1,
                    'W2': self.value_net.W2,
                    'b2': self.value_net.b2,
                    'W3': self.value_net.W3,
                    'b3': self.value_net.b3
                }
            }
            np.save(path, checkpoint)
        except Exception as e:
            logging.error(f"Error saving checkpoint: {e}")
            raise
    
    def save_model(self, path):
        """Save the current model state."""
        checkpoint = {
            'policy': {
                'W1': self.policy.W1,
                'b1': self.policy.b1,
                'W2': self.policy.W2,
                'b2': self.policy.b2,
                'W_mean': self.policy.W_mean,
                'b_mean': self.policy.b_mean,
                'W_std': self.policy.W_std,
                'b_std': self.policy.b_std
            },
            'value_net': {
                'W1': self.value_net.W1,
                'b1': self.value_net.b1,
                'W2': self.value_net.W2,
                'b2': self.value_net.b2,
                'W3': self.value_net.W3,
                'b3': self.value_net.b3
            },
            'metrics': self.get_metrics()
        }
        np.save(path, checkpoint)
    
    def load_checkpoint(self, path):
        """Load neural network weights."""
        checkpoint = np.load(path, allow_pickle=True).item()
        
        # Load policy weights
        self.policy.W1 = checkpoint['policy']['W1']
        self.policy.b1 = checkpoint['policy']['b1']
        self.policy.W2 = checkpoint['policy']['W2']
        self.policy.b2 = checkpoint['policy']['b2']
        self.policy.W_mean = checkpoint['policy']['W_mean']
        self.policy.b_mean = checkpoint['policy']['b_mean']
        self.policy.W_std = checkpoint['policy']['W_std']
        self.policy.b_std = checkpoint['policy']['b_std']
        
        # Load value network weights
        self.value_net.W1 = checkpoint['value_net']['W1']
        self.value_net.b1 = checkpoint['value_net']['b1']
        self.value_net.W2 = checkpoint['value_net']['W2']
        self.value_net.b2 = checkpoint['value_net']['b2']
        self.value_net.W3 = checkpoint['value_net']['W3']
        self.value_net.b3 = checkpoint['value_net']['b3']
    
    def load_model(self, path):
        """Load model state."""
        checkpoint = np.load(path, allow_pickle=True).item()
        
        # Load policy weights
        self.policy.W1 = checkpoint['policy']['W1']
        self.policy.b1 = checkpoint['policy']['b1']
        self.policy.W2 = checkpoint['policy']['W2']
        self.policy.b2 = checkpoint['policy']['b2']
        self.policy.W_mean = checkpoint['policy']['W_mean']
        self.policy.b_mean = checkpoint['policy']['b_mean']
        self.policy.W_std = checkpoint['policy']['W_std']
        self.policy.b_std = checkpoint['policy']['b_std']
        
        # Load value network weights
        self.value_net.W1 = checkpoint['value_net']['W1']
        self.value_net.b1 = checkpoint['value_net']['b1']
        self.value_net.W2 = checkpoint['value_net']['W2']
        self.value_net.b2 = checkpoint['value_net']['b2']
        self.value_net.W3 = checkpoint['value_net']['W3']
        self.value_net.b3 = checkpoint['value_net']['b3']
        
        # Update metrics if available
        if 'metrics' in checkpoint:
            for key, value in checkpoint['metrics'].items():
                if key in self.metrics:
                    self.metrics[key] = deque(maxlen=100)
                    self.metrics[key].append(value)
    
    def get_metrics(self):
        """Get current training metrics."""
        try:
            metrics = {}
            for key in self.metrics:
                values = list(self.metrics[key])
                metrics[key] = float(np.mean(values)) if values else 0.0
            return metrics
        except Exception as e:
            logging.error(f"Error in get_metrics: {e}")
            return {
                'policy_loss': 0.0,
                'value_loss': 0.0,
                'entropy_loss': 0.0,
                'total_loss': 0.0,
                'mean_reward': 0.0,
                'clip_fraction': 0.0,
                'approx_kl': 0.0,
                'difficulty_accuracy': 0.0,
                'topic_diversity': 0.0
            } 