import numpy as np
import torch
from collections import deque
from .neural_network import GaussianPolicy, ValueNetwork, compute_gae
import logging
import os

class PPOAgent:
    def __init__(self, state_dim=9, action_dim=1, hidden_dim=128,  # Even smaller network for faster learning
                 lr_actor=0.00005, lr_critic=0.0001, gamma=0.99, 
                 gae_lambda=0.95, clip_ratio=0.1, target_kl=0.008,
                 train_actor_iterations=5, train_critic_iterations=8):
        """Initialize PPO agent with highly optimized parameters."""
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Initialize neural networks with enhanced architecture
        self.policy = GaussianPolicy(state_dim, action_dim, hidden_dim)
        self.value_net = ValueNetwork(state_dim, hidden_dim)
        
        # Initialize metrics tracking
        self.metrics = {
            'policy_loss': deque(maxlen=100),
            'value_loss': deque(maxlen=100),
            'entropy_loss': deque(maxlen=100),
            'total_loss': deque(maxlen=100),
            'mean_reward': deque(maxlen=100),
            'clip_fraction': deque(maxlen=100),
            'approx_kl': deque(maxlen=100),
            'difficulty_accuracy': deque(maxlen=100),
            'topic_diversity': deque(maxlen=100),
            'topic_performance': {}  # Will be populated with topics dynamically
        }
        
        # Training parameters with improved defaults
        self.initial_lr_actor = lr_actor
        self.initial_lr_critic = lr_critic
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.train_actor_iterations = train_actor_iterations
        self.train_critic_iterations = train_critic_iterations
        
        # Enhanced exploration parameters
        self.min_std = 0.1  # Even more focused exploration
        self.max_std = 0.5  # More stable learning
        self.std_decay_rate = 0.995  # Faster decay
        
        # Learning rate decay parameters
        self.lr_decay = 0.997  # Slower decay for more stable learning
        self.min_lr = 1e-6
        
        # Add KL divergence tracking with adaptive threshold
        self.last_kl = 0
        self.kl_threshold = 0.3  # Even tighter policy updates
        self.kl_target_factor = 1.1
        
        # Enhanced batch processing
        self.batch_size = 32  # Smaller batches for more frequent updates
        self.mini_batch_size = 8
        
        # Add value clipping ratio
        self.value_clip_ratio = 0.1
        
        # Add adaptive entropy coefficient
        self.initial_entropy_coef = 0.05  # Much more focused on exploitation
        self.min_entropy_coef = 0.005
        self.entropy_decay = 0.995
        
        # Add momentum for policy updates
        self.policy_momentum = 0.98  # Higher momentum for more stable updates
        self.policy_velocity = {}
        
        # Add state normalization with history
        self.state_history = deque(maxlen=2000)  # Shorter history for more recent focus
        self.state_mean = np.zeros(state_dim)
        self.state_std = np.ones(state_dim)
        self.state_count = 0
        self.running_state_mean = np.zeros(state_dim)
        self.running_state_std = np.ones(state_dim)
        
        # Add gradient normalization parameters
        self.max_grad_norm = 0.2  # Even tighter gradient clipping
        
        # Add noise injection for policy with adaptive scaling
        self.noise_scale = 0.01  # Very small initial noise
        self.noise_decay = 0.997  # Moderate decay
        self.min_noise = 0.001
        
        # Add experience replay buffer with priorities
        self.replay_buffer = {
            'states': deque(maxlen=2000),
            'actions': deque(maxlen=2000),
            'rewards': deque(maxlen=2000),
            'values': deque(maxlen=2000),
            'log_probs': deque(maxlen=2000),
            'dones': deque(maxlen=2000),
            'priorities': deque(maxlen=2000),
            'topics': deque(maxlen=2000)
        }
        
        # Add curriculum learning parameters with smoother progression
        self.curriculum = {
            'current_stage': 0,
            'stage_thresholds': [0.45, 0.6, 0.7],  # Lower initial thresholds
            'difficulty_ranges': [(1, 2), (1.5, 4), (3, 6)],  # Much more gradual progression
            'min_questions_per_stage': 20,  # Faster progression
            'performance_window': 15  # Window for performance evaluation
        }
        
        # Add performance tracking
        self.performance_history = deque(maxlen=100)
        self.difficulty_history = deque(maxlen=100)
        
        # Clear memory buffers
        self.clear_memory()
        
        # Training step counter
        self.train_steps = 0
        
        # Add adaptive parameters
        self.adaptive_params = {
            'entropy_coef': self.initial_entropy_coef,
            'current_std': 0.3,
            'exploration_factor': 1.0
        }
    
    def clear_memory(self):
        """Clear memory buffers."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def normalize_states(self, states):
        """Normalize states using running statistics."""
        # Update running statistics
        batch_mean = np.mean(states, axis=0)
        batch_std = np.std(states, axis=0)
        batch_count = states.shape[0]
        
        # Update running mean and std using Welford's online algorithm
        delta = batch_mean - self.running_state_mean
        self.running_state_mean += delta * batch_count / (self.state_count + batch_count)
        self.running_state_std = np.sqrt(
            (self.running_state_std ** 2 * self.state_count + batch_std ** 2 * batch_count +
             delta ** 2 * self.state_count * batch_count / (self.state_count + batch_count)) /
            (self.state_count + batch_count)
        )
        self.state_count += batch_count
        
        # Use running statistics for normalization
        return (states - self.running_state_mean) / (self.running_state_std + 1e-8)

    def normalize_gradients(self, gradients):
        """Normalize gradients to prevent extreme updates."""
        total_norm = np.sqrt(sum(np.sum(g**2) for g in gradients.values()))
        if total_norm > self.max_grad_norm:
            scale = self.max_grad_norm / (total_norm + 1e-8)
            return {k: v * scale for k, v in gradients.items()}
        return gradients

    def compute_priorities(self, advantages):
        """Compute priorities for experience replay."""
        # Ensure advantages are 1-dimensional
        advantages = np.array(advantages).flatten()
        
        # Compute absolute priorities with small offset to avoid zero probabilities
        priorities = np.abs(advantages) + 1e-5
        
        # Normalize priorities
        priorities = priorities / np.sum(priorities)
        
        return priorities.flatten()  # Ensure 1-dimensional output

    def layer_norm(self, x, epsilon=1e-8):
        """Apply layer normalization."""
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True)
        return (x - mean) / (std + epsilon)
    
    def select_action(self, state):
        """Select action with enhanced stability and exploration control."""
        try:
            # Ensure state is properly shaped and normalized
            state = np.array(state, dtype=np.float32)
            if len(state.shape) == 1:
                state = state.reshape(1, -1)
            
            # Update state history and normalize
            self.state_history.append(state)
            normalized_state = self.normalize_states(state)
            
            # Convert to tensor for policy
            state_tensor = torch.FloatTensor(normalized_state)
            
            # Get action distribution
            with torch.no_grad():
                mu, std = self.policy(state_tensor)
            
            # Convert tensors to numpy if needed
            mu = mu.numpy() if isinstance(mu, torch.Tensor) else mu
            std = std.numpy() if isinstance(std, torch.Tensor) else std
            
            # Adjust exploration based on performance
            if len(self.performance_history) >= 5:
                recent_performance = np.mean(list(self.performance_history)[-5:])
                self.adaptive_params['exploration_factor'] = max(0.5, 1.0 - recent_performance)
            
            # Add adaptive noise based on performance and training progress
            noise_scale = self.noise_scale * self.adaptive_params['exploration_factor']
            noise = np.random.normal(0, noise_scale)
            
            # Apply progressive noise reduction
            if self.train_steps > 0:
                noise *= (1.0 / np.sqrt(1 + self.train_steps * 0.01))
            
            mu = mu + noise
            
            # Adjust standard deviation based on performance
            std = std * self.adaptive_params['exploration_factor']
            std = np.clip(std, self.min_std, self.max_std)
            
            # Sample action with controlled randomness
            action = np.random.normal(mu, std)
            
            # Clip action to valid range with smooth boundaries
            action = np.tanh(action) * 4.5 + 5.5  # Maps to [1, 10] with smooth boundaries
            
            # Get value estimate using forward method
            value = self.value_net.forward(normalized_state)
            
            # Ensure action, value, and log_prob are properly shaped
            action = np.array(action).reshape(-1)  # Flatten to 1D
            value = np.array(value).reshape(-1)    # Flatten to 1D
            
            # Get log probability
            action_tensor = torch.FloatTensor(action.reshape(-1, self.action_dim))
            log_prob = self.policy.get_log_prob(state_tensor, action_tensor)
            log_prob = log_prob.numpy() if isinstance(log_prob, torch.Tensor) else log_prob
            log_prob = log_prob.reshape(-1)  # Flatten to 1D
            
            # Update noise scale with minimum bound
            self.noise_scale = max(self.min_noise, self.noise_scale * self.noise_decay)
            
            # Update entropy coefficient
            self.adaptive_params['entropy_coef'] = max(
                self.min_entropy_coef,
                self.adaptive_params['entropy_coef'] * self.entropy_decay
            )
            
            return action, value, log_prob
            
        except Exception as e:
            logging.error(f"Error in select_action: {e}")
            return np.array([5.0]), np.array([0.0]), np.array([0.0])  # Safe fallback
    
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
        """Train policy and value networks with curriculum learning and enhanced stability."""
        try:
            if len(self.states) < self.batch_size:
                logging.info(f"Skipping training: insufficient samples ({len(self.states)} < {self.batch_size})")
                return None
            
            self.train_steps += 1
            logging.info(f"Starting training step {self.train_steps}")
            
            # Update curriculum stage based on performance
            self._update_curriculum_stage()
            
            # Prepare current batch data with proper normalization
            states = np.vstack(self.states)
            states = self.normalize_states(states)  # Normalize states
            
            actions = np.vstack(self.actions)
            old_log_probs = np.array(self.log_probs).reshape(-1, 1)
            values = np.array(self.values).reshape(-1, 1)
            rewards = np.array(self.rewards).reshape(-1, 1)
            dones = np.array(self.dones).reshape(-1, 1)
            
            logging.info(f"Batch statistics - States: {states.shape}, Actions: {actions.shape}, Rewards: {len(rewards)}")
            
            # Normalize rewards for stability
            rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-8)
            
            # Compute advantages and returns with GAE
            advantages, returns = self._calculate_advantages(rewards, values, dones)
            
            # Normalize advantages
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
            
            # Compute priorities for experience replay
            priorities = self.compute_priorities(advantages)
            
            # Training metrics
            epoch_metrics = {
                'policy_loss': [],
                'value_loss': [],
                'entropy_loss': [],
                'approx_kl': [],
                'clip_fraction': [],
                'mean_reward': float(np.mean(rewards))
            }
            
            # Process data in mini-batches
            num_batches = len(states) // self.mini_batch_size
            
            # Value network update first for better value estimates
            for _ in range(self.train_critic_iterations):
                np.random.shuffle(priorities)  # Shuffle priorities for each iteration
                
                for batch_idx in range(num_batches):
                    batch_start = batch_idx * self.mini_batch_size
                    batch_end = (batch_idx + 1) * self.mini_batch_size
                    
                    # Get batch data
                    states_batch = states[batch_start:batch_end]
                    returns_batch = returns[batch_start:batch_end]
                    
                    # Compute value loss
                    values_pred = self.value_net.forward(states_batch)
                    value_loss, value_clip = self._compute_value_loss(
                        values_pred, returns_batch, self.curriculum['current_stage']
                    )
                    
                    logging.info(f"Value network update - Loss: {value_loss:.4f}, Clip: {value_clip:.4f}")
                    
                    if not np.isnan(value_loss):  # Only update if loss is valid
                        # Get value gradients
                        gradients = self.value_net.get_gradients(states_batch, returns_batch)
                        
                        # Normalize gradients
                        gradients = self.normalize_gradients(gradients)
                        
                        # Update value network
                        self.value_net.apply_gradients(gradients, self.lr_critic)
                        epoch_metrics['value_loss'].append(value_loss)
                        epoch_metrics['clip_fraction'].append(value_clip)
            
            # Policy network update
            for _ in range(self.train_actor_iterations):
                np.random.shuffle(priorities)  # Shuffle priorities for each iteration
                
                for batch_idx in range(num_batches):
                    batch_start = batch_idx * self.mini_batch_size
                    batch_end = (batch_idx + 1) * self.mini_batch_size
                    
                    # Get batch data
                    states_batch = states[batch_start:batch_end]
                    actions_batch = actions[batch_start:batch_end]
                    old_log_probs_batch = old_log_probs[batch_start:batch_end]
                    advantages_batch = advantages[batch_start:batch_end]
                    
                    # Get policy distribution
                    mean, std = self.policy.forward(states_batch)
                    new_log_probs = -0.5 * (((actions_batch - mean) / (std + 1e-8)) ** 2 
                                          + 2 * np.log(std + 1e-8) + np.log(2 * np.pi))
                    
                    # Compute ratios
                    ratios = np.exp(new_log_probs - old_log_probs_batch)
                    
                    # Compute policy loss
                    policy_loss, entropy, kl = self._compute_policy_loss(
                        ratios, advantages_batch, old_log_probs_batch, new_log_probs,
                        self.curriculum['current_stage']
                    )
                    
                    logging.info(f"Policy network update - Loss: {policy_loss:.4f}, Entropy: {entropy:.4f}, KL: {kl:.4f}")
                    
                    if not np.isnan(policy_loss):  # Only update if loss is valid
                        # Get policy gradients
                        gradients = self.policy.get_gradients(states_batch, advantages_batch)
                        
                        # Normalize gradients
                        gradients = self.normalize_gradients(gradients)
                        
                        # Update policy
                        self.policy.apply_gradients(gradients, self.lr_actor)
                        
                        epoch_metrics['policy_loss'].append(policy_loss)
                        epoch_metrics['entropy_loss'].append(entropy)
                        epoch_metrics['approx_kl'].append(kl)
            
            # Calculate mean metrics
            metrics = {}
            for key, values in epoch_metrics.items():
                if values:  # Only compute mean if we have values
                    metrics[key] = float(np.mean([v for v in values if not np.isnan(v)]))
                else:
                    metrics[key] = 0.0
            
            logging.info(f"Training step {self.train_steps} completed - Metrics: {metrics}")
            
            # Update metrics history
            self.update_metrics(metrics)
            
            # Clear memory after successful training
            self.clear_memory()
            
            return metrics
            
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
            metrics_dict = {}
            
            # Process standard metrics
            for key in ['policy_loss', 'value_loss', 'entropy_loss', 'total_loss', 
                       'mean_reward', 'clip_fraction', 'approx_kl', 
                       'difficulty_accuracy', 'topic_diversity']:
                values = [float(v) for v in self.metrics[key] if not np.isnan(v)]
                metrics_dict[key] = float(np.mean(values)) if values else 0.0
            
            # Process topic performance separately if it exists
            if 'topic_performance' in self.metrics:
                topic_metrics = {}
                for topic, scores in self.metrics['topic_performance'].items():
                    if isinstance(scores, (list, deque)):
                        values = [float(s) for s in scores if not np.isnan(s)]
                        topic_metrics[topic] = float(np.mean(values)) if values else 0.0
                metrics_dict['topic_performance'] = topic_metrics
            
            return metrics_dict
            
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
                'topic_diversity': 0.0,
                'topic_performance': {}
            }

    def update_metrics(self, metrics_dict):
        """Update training metrics."""
        try:
            for key, value in metrics_dict.items():
                if key == 'topic_performance':
                    # Handle topic performance separately
                    for topic, score in value.items():
                        if topic not in self.metrics['topic_performance']:
                            self.metrics['topic_performance'][topic] = deque(maxlen=100)
                        self.metrics['topic_performance'][topic].append(float(score))
                else:
                    # Handle standard metrics
                    if key in self.metrics:
                        self.metrics[key].append(float(value))
        except Exception as e:
            logging.error(f"Error updating metrics: {e}")

    def _compute_value_loss(self, values_pred, returns, curr_stage):
        """Compute value loss with clipping and numerical stability."""
        try:
            # Ensure inputs are numpy arrays and properly shaped
            values_pred = np.array(values_pred, dtype=np.float32).reshape(-1, 1)
            returns = np.array(returns, dtype=np.float32).reshape(-1, 1)
            
            # Normalize predictions and returns to prevent extreme values
            values_pred = np.clip(values_pred, -10.0, 10.0)
            returns = np.clip(returns, -10.0, 10.0)
            
            # Compute clipped value predictions
            values_clipped = np.clip(
                values_pred,
                returns - self.value_clip_ratio,
                returns + self.value_clip_ratio
            )
            
            # Compute losses with numerical stability
            loss1 = np.square(returns - values_pred)
            loss2 = np.square(returns - values_clipped)
            
            # Take maximum of losses and apply safety bounds
            value_loss = np.mean(np.maximum(loss1, loss2))
            value_loss = np.clip(value_loss, 0.0, 100.0)  # Prevent extreme loss values
            
            # Calculate clip fraction for monitoring
            clip_fraction = np.mean(np.abs(values_pred - returns) > self.value_clip_ratio)
            
            # Apply curriculum-based scaling with safety bounds
            stage_scale = 1.0 + (0.05 * curr_stage)  # 5% increase per stage
            stage_scale = np.clip(stage_scale, 0.8, 1.5)  # Limit scaling range
            
            value_loss *= stage_scale
            value_loss = np.clip(value_loss, 0.0, 100.0)  # Final safety bound
            
            return float(value_loss), float(clip_fraction)
            
        except Exception as e:
            logging.error(f"Error in _compute_value_loss: {e}")
            return 0.0, 0.0

    def _compute_policy_loss(self, ratios, advantages, old_log_probs, new_log_probs, curr_stage):
        """Compute policy loss with enhanced stability and adaptive clipping."""
        try:
            # Ensure inputs are numpy arrays and properly shaped
            ratios = np.array(ratios, dtype=np.float32).reshape(-1, 1)
            advantages = np.array(advantages, dtype=np.float32).reshape(-1, 1)
            
            # Adaptive clipping based on advantage magnitude
            clip_ratio = self.clip_ratio * (1 + 0.5 * np.mean(np.abs(advantages)))
            clip_ratio = np.clip(clip_ratio, 0.1, 0.3)  # Keep within reasonable bounds
            
            # Compute importance-sampled policy loss with numerical stability
            loss1 = ratios * advantages
            loss2 = np.clip(ratios, 1 - clip_ratio, 1 + clip_ratio) * advantages
            
            # Adaptive advantage weighting
            advantage_weights = 1.0 + 0.5 * np.abs(advantages)  # More weight to significant advantages
            advantage_weights = np.clip(advantage_weights, 0.5, 2.0)
            
            # Compute weighted policy loss
            policy_loss = -np.mean(np.minimum(loss1, loss2) * advantage_weights)
            
            # Add entropy bonus with adaptive coefficient
            entropy = -np.mean(new_log_probs) * self.adaptive_params['entropy_coef']
            
            # Compute KL divergence for monitoring
            log_ratio = new_log_probs - old_log_probs
            approx_kl = np.mean(np.exp(log_ratio) - 1 - log_ratio)
            
            # Apply curriculum-based scaling
            stage_scale = 1.0 + (0.1 * curr_stage)  # 10% increase per stage
            stage_scale = np.clip(stage_scale, 0.8, 1.2)
            
            # Combine losses with safety bounds
            total_loss = (policy_loss - entropy) * stage_scale
            total_loss = np.clip(total_loss, -10.0, 10.0)
            
            return float(total_loss), float(entropy), float(approx_kl)
            
        except Exception as e:
            logging.error(f"Error in _compute_policy_loss: {e}")
            return 0.0, 0.0, 0.0

    def _update_policy_momentum(self, gradients):
        """Update policy gradients using momentum."""
        if not self.policy_velocity:
            self.policy_velocity = {k: np.zeros_like(v) for k, v in gradients.items()}
        
        updated_gradients = {}
        for key in gradients:
            # Update velocity
            self.policy_velocity[key] = (
                self.policy_momentum * self.policy_velocity[key] +
                (1 - self.policy_momentum) * gradients[key]
            )
            # Use velocity for gradient update
            updated_gradients[key] = self.policy_velocity[key]
        
        return updated_gradients

    def _adapt_learning_rates(self):
        """Adapt learning rates based on progress."""
        # Decay learning rates
        self.lr_actor = max(self.lr_actor * self.lr_decay, self.min_lr)
        self.lr_critic = max(self.lr_critic * self.lr_decay, self.min_lr)
        
        # Adapt based on KL divergence
        if self.last_kl > self.target_kl * self.kl_target_factor:
            self.lr_actor *= 0.5
        elif self.last_kl < self.target_kl / self.kl_target_factor:
            self.lr_actor *= 1.5
        
        # Clip learning rates
        self.lr_actor = np.clip(self.lr_actor, self.min_lr, self.initial_lr_actor)
        self.lr_critic = np.clip(self.lr_critic, self.min_lr, self.initial_lr_critic)

    def _update_curriculum_stage(self):
        """Update curriculum stage based on performance."""
        if len(self.metrics['mean_reward']) < self.curriculum['min_questions_per_stage']:
            return  # Not enough data to evaluate stage progression
            
        # Calculate recent performance metrics
        recent_rewards = list(self.metrics['mean_reward'])[-self.curriculum['min_questions_per_stage']:]
        avg_performance = np.mean(recent_rewards)
        
        # Get current stage info
        curr_stage = self.curriculum['current_stage']
        max_stage = len(self.curriculum['stage_thresholds']) - 1
        
        # Check if performance meets threshold for advancement
        if curr_stage < max_stage:
            threshold = self.curriculum['stage_thresholds'][curr_stage]
            if avg_performance >= threshold:
                self.curriculum['current_stage'] = min(curr_stage + 1, max_stage)
                # Reset exploration parameters for new stage
                self.noise_scale *= 1.2  # Temporarily increase exploration
                logging.info(f"Advanced to curriculum stage {self.curriculum['current_stage']}")
        
        # Check if performance is too low for current stage
        elif curr_stage > 0:
            prev_threshold = self.curriculum['stage_thresholds'][curr_stage - 1]
            if avg_performance < prev_threshold * 0.8:  # 20% below previous threshold
                self.curriculum['current_stage'] = max(curr_stage - 1, 0)
                logging.info(f"Regressed to curriculum stage {self.curriculum['current_stage']}")

    def _scale_gradients_by_curriculum(self, gradients, curr_stage):
        """Scale gradients based on curriculum stage and performance."""
        # Base scaling factor increases with stage
        base_scale = 1.0 + (curr_stage * 0.1)  # 10% increase per stage
        
        # Get recent performance metrics
        if len(self.metrics['mean_reward']) > 0:
            recent_performance = np.mean(list(self.metrics['mean_reward'])[-10:])
            # Adjust scaling based on recent performance
            performance_scale = np.clip(recent_performance, 0.5, 1.5)
        else:
            performance_scale = 1.0
        
        # Calculate final scale factor
        scale_factor = base_scale * performance_scale
        
        # Apply scaling to gradients
        scaled_gradients = {
            k: v * scale_factor for k, v in gradients.items()
        }
        
        # Add gradient noise scaled by curriculum stage
        noise_scale = 0.01 * (1.0 + curr_stage * 0.1)
        for k in scaled_gradients:
            noise = np.random.normal(0, noise_scale, scaled_gradients[k].shape)
            scaled_gradients[k] += noise
        
        return scaled_gradients

    def get_curriculum_info(self):
        """Get current curriculum learning status."""
        return {
            'current_stage': self.curriculum['current_stage'],
            'stage_thresholds': self.curriculum['stage_thresholds'],
            'difficulty_ranges': self.curriculum['difficulty_ranges'],
            'min_questions_per_stage': self.curriculum['min_questions_per_stage'],
            'recent_performance': np.mean(list(self.metrics['mean_reward'])[-10:]) if len(self.metrics['mean_reward']) > 0 else 0.0
        } 