import numpy as np
from collections import deque
from .neural_network import GaussianPolicy, ValueNetwork, compute_gae

class PPOAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=64,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        batch_size=64,
        epochs=10,
        buffer_size=2048
    ):
        """Initialize PPO agent with hyperparameters."""
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.epochs = epochs
        self.buffer_size = buffer_size
        
        # Initialize neural networks
        self.policy = GaussianPolicy(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            learning_rate=learning_rate
        )
        self.value = ValueNetwork(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            learning_rate=learning_rate
        )
        
        # Initialize experience buffers
        self.reset_buffers()
        
        # Initialize metrics with deque for efficient tracking
        self.metrics = {
            'policy_loss': deque(maxlen=100),
            'value_loss': deque(maxlen=100),
            'entropy_loss': deque(maxlen=100),
            'total_loss': deque(maxlen=100),
            'mean_reward': deque(maxlen=100),
            'clip_fraction': deque(maxlen=100),
            'approx_kl': deque(maxlen=100)
        }
    
    def reset_buffers(self):
        """Reset experience replay buffers."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.advantages = None
        self.returns = None
    
    def select_action(self, state):
        """Select action based on current policy."""
        try:
            # Ensure state is correctly shaped
            if len(state.shape) == 1:
                state = state.reshape(1, -1)
            
            # Get action distribution parameters
            mean, std = self.policy.forward(state)
            
            # Sample action and clip to valid range [1, 10]
            action = self.policy.sample(mean, std)
            action = np.clip(action, 1.0, 10.0)
            
            # Get value estimate and log probability
            value = self.value.forward(state)
            log_prob = self.policy.log_prob(action, mean, std)
            
            return action, value, log_prob
        except Exception as e:
            print(f"Error in select_action: {e}")
            # Return safe default values
            return np.array([5.0]), 0.0, 0.0
    
    def store_transition(self, state, action, reward, value, log_prob, done):
        """Store transition in experience replay buffer."""
        try:
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.values.append(value)
            self.log_probs.append(log_prob)
            self.dones.append(done)
            
            # Clear oldest transitions if buffer is full
            if len(self.states) > self.buffer_size:
                self.states = self.states[-self.buffer_size:]
                self.actions = self.actions[-self.buffer_size:]
                self.rewards = self.rewards[-self.buffer_size:]
                self.values = self.values[-self.buffer_size:]
                self.log_probs = self.log_probs[-self.buffer_size:]
                self.dones = self.dones[-self.buffer_size:]
        except Exception as e:
            print(f"Error in store_transition: {e}")
    
    def _compute_advantages_and_returns(self):
        """Compute advantages and returns using GAE."""
        try:
            states = np.array(self.states)
            values = np.array(self.values)
            rewards = np.array(self.rewards)
            dones = np.array(self.dones)
            
            # Get final value estimate for bootstrapping
            next_value = self.value.forward(states[-1:])
            
            # Compute advantages and returns
            self.advantages, self.returns = compute_gae(
                rewards, values, dones,
                gamma=self.gamma, lam=self.gae_lambda
            )
            
            # Normalize advantages for training stability
            self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)
        except Exception as e:
            print(f"Error in _compute_advantages_and_returns: {e}")
            self.advantages = np.zeros_like(self.rewards)
            self.returns = np.zeros_like(self.rewards)
    
    def _get_minibatch(self, indices):
        """Get minibatch of experiences."""
        try:
            states = np.array(self.states)[indices]
            actions = np.array(self.actions)[indices]
            old_log_probs = np.array(self.log_probs)[indices]
            advantages = self.advantages[indices]
            returns = self.returns[indices]
            return states, actions, old_log_probs, advantages, returns
        except Exception as e:
            print(f"Error in _get_minibatch: {e}")
            return None
    
    def update(self):
        """Update policy and value networks using PPO."""
        if len(self.states) < self.batch_size:
            return None
            
        try:
            # Compute advantages and returns
            self._compute_advantages_and_returns()
            
            n_samples = len(self.states)
            batch_indices = np.arange(n_samples)
            n_batches = max(n_samples // self.batch_size, 1)
            
            metrics = {
                'policy_loss': 0,
                'value_loss': 0,
                'entropy_loss': 0,
                'total_loss': 0,
                'clip_fraction': 0,
                'approx_kl': 0
            }
            
            # Perform multiple epochs of updates
            for _ in range(self.epochs):
                np.random.shuffle(batch_indices)
                
                # Process minibatches
                for i in range(n_batches):
                    start_idx = i * self.batch_size
                    end_idx = min(start_idx + self.batch_size, n_samples)
                    mb_indices = batch_indices[start_idx:end_idx]
                    
                    # Get minibatch data
                    batch_data = self._get_minibatch(mb_indices)
                    if batch_data is None:
                        continue
                    states, actions, old_log_probs, advantages, returns = batch_data
                    
                    # Get current action probabilities and values
                    mean, std = self.policy.forward(states)
                    new_log_probs = self.policy.log_prob(actions, mean, std)
                    values = self.value.forward(states)
                    
                    # Compute probability ratio and clipped ratio
                    ratio = np.exp(new_log_probs - old_log_probs)
                    clipped_ratio = np.clip(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                    
                    # Compute policy loss
                    policy_loss = -np.minimum(
                        ratio * advantages,
                        clipped_ratio * advantages
                    ).mean()
                    
                    # Compute value loss
                    value_loss = 0.5 * ((values - returns) ** 2).mean()
                    
                    # Compute entropy loss (encourage exploration)
                    entropy_loss = -np.mean(np.log(std))
                    
                    # Compute total loss
                    total_loss = (
                        policy_loss +
                        self.value_coef * value_loss +
                        self.entropy_coef * entropy_loss
                    )
                    
                    # Compute metrics
                    clip_fraction = np.mean(np.abs(ratio - 1.0) > self.clip_ratio)
                    approx_kl = 0.5 * np.mean((new_log_probs - old_log_probs) ** 2)
                    
                    # Update networks with gradient clipping
                    self.policy.backward(policy_loss, entropy_loss)
                    self.value.backward(value_loss)
                    
                    # Update metrics
                    metrics['policy_loss'] += policy_loss
                    metrics['value_loss'] += value_loss
                    metrics['entropy_loss'] += entropy_loss
                    metrics['total_loss'] += total_loss
                    metrics['clip_fraction'] += clip_fraction
                    metrics['approx_kl'] += approx_kl
            
            # Average metrics over all updates
            total_updates = self.epochs * n_batches
            if total_updates > 0:
                for key in metrics:
                    metrics[key] /= total_updates
                    self.metrics[key].append(metrics[key])
            
            # Compute mean reward
            mean_reward = np.mean(self.rewards)
            metrics['mean_reward'] = mean_reward
            self.metrics['mean_reward'].append(mean_reward)
            
            # Reset buffers
            self.reset_buffers()
            
            return metrics
        except Exception as e:
            print(f"Error in update: {e}")
            return None
    
    def get_metrics(self):
        """Get current training metrics."""
        try:
            return {
                'policy_loss': np.mean(list(self.metrics['policy_loss'])),
                'value_loss': np.mean(list(self.metrics['value_loss'])),
                'entropy_loss': np.mean(list(self.metrics['entropy_loss'])),
                'total_loss': np.mean(list(self.metrics['total_loss'])),
                'mean_reward': np.mean(list(self.metrics['mean_reward'])),
                'clip_fraction': np.mean(list(self.metrics['clip_fraction'])),
                'approx_kl': np.mean(list(self.metrics['approx_kl']))
            }
        except Exception as e:
            print(f"Error in get_metrics: {e}")
            return {} 