import numpy as np
from collections import deque
from neural_network import GaussianPolicy, ValueNetwork, compute_gae

class PPOAgent:
    def __init__(self, state_dim=9, action_dim=1, hidden_dim=64, 
                 lr_actor=0.0003, lr_critic=0.001, gamma=0.99, 
                 gae_lambda=0.95, clip_ratio=0.2, target_kl=0.01,
                 train_actor_iterations=10, train_critic_iterations=10):
        """Initialize PPO agent with policy and value networks."""
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Initialize neural networks
        self.policy = GaussianPolicy(state_dim, action_dim, hidden_dim)
        self.value_net = ValueNetwork(state_dim, hidden_dim)
        
        # Training parameters
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.train_actor_iterations = train_actor_iterations
        self.train_critic_iterations = train_critic_iterations
        
        # Initialize memory buffers
        self.clear_memory()
        
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
    
    def clear_memory(self):
        """Clear memory buffers."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def select_action(self, state):
        """Select action using the policy network."""
        try:
            # Ensure state is properly shaped
            state = np.array(state).reshape(1, -1)
            
            # Get action distribution parameters
            mean, std = self.policy.forward(state)
            
            # Sample action and compute log probability
            action = self.policy.sample(mean, std)
            log_prob = self.policy.log_prob(action, mean, std)
            
            # Get value estimate
            value = self.value_net.forward(state)
            
            # Store in memory
            self.states.append(state)
            self.actions.append(action)
            self.log_probs.append(log_prob)
            self.values.append(value)
            
            # Return tuple of (action, value, log_prob)
            return np.array([action.flatten()[0]]), float(value), float(log_prob)
        except Exception as e:
            print(f"Error in select_action: {e}")
            return np.array([0.5]), 0.0, 0.0  # Return safe default values
    
    def store_transition(self, reward, done):
        """Store transition information."""
        self.rewards.append(reward)
        self.dones.append(done)
    
    def train(self):
        """Train policy and value networks using PPO."""
        # Convert lists to numpy arrays
        states = np.vstack(self.states)
        actions = np.vstack(self.actions)
        old_log_probs = np.vstack(self.log_probs)
        values = np.vstack(self.values)
        rewards = np.array(self.rewards).reshape(-1, 1)
        dones = np.array(self.dones).reshape(-1, 1)
        
        # Compute advantages using GAE
        advantages, returns = compute_gae(
            rewards, values, dones, 
            self.gamma, self.gae_lambda
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Compute returns for value function training
        returns = advantages + values
        
        # PPO training loop
        for _ in range(self.train_actor_iterations):
            # Forward pass through policy
            mean, std = self.policy.forward(states)
            new_log_probs = self.policy.log_prob(actions, mean, std)
            
            # Compute ratio and clipped ratio
            ratio = np.exp(new_log_probs - old_log_probs)
            clipped_ratio = np.clip(ratio, 1-self.clip_ratio, 1+self.clip_ratio)
            
            # Compute losses
            surrogate1 = ratio * advantages
            surrogate2 = clipped_ratio * advantages
            actor_loss = -np.minimum(surrogate1, surrogate2).mean()
            
            # Compute KL divergence
            approx_kl = ((old_log_probs - new_log_probs) ** 2).mean()
            if approx_kl > self.target_kl:
                break
            
            # Update policy
            policy_grads = self.policy.get_gradients(states, -advantages)
            self.policy.apply_gradients(policy_grads, self.lr_actor)
        
        # Value function training
        for _ in range(self.train_critic_iterations):
            # Compute value loss
            values = self.value_net.forward(states)
            value_loss = ((values - returns) ** 2).mean()
            
            # Update value network
            value_grads = self.value_net.get_gradients(states, 2*(values - returns))
            self.value_net.apply_gradients(value_grads, self.lr_critic)
        
        # Clear memory after training
        self.clear_memory()
        
        # Calculate mean reward
        mean_reward = float(np.mean(self.rewards)) if self.rewards else 0.0
        
        return {
            'actor_loss': float(actor_loss),
            'value_loss': float(value_loss),
            'approx_kl': float(approx_kl),
            'mean_episode_reward': mean_reward
        }
    
    def save_checkpoint(self, path):
        """Save neural network weights."""
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