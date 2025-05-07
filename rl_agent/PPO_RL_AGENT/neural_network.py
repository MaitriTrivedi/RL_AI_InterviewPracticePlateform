import numpy as np

def relu(x):
    """ReLU activation function."""
    return np.maximum(0, x)

def relu_derivative(x):
    """Derivative of ReLU."""
    return np.where(x > 0, 1, 0)

def tanh(x):
    """Tanh activation function."""
    return np.tanh(x)

def tanh_derivative(x):
    """Derivative of tanh."""
    return 1 - np.tanh(x) ** 2

class Layer:
    def __init__(self, input_dim, output_dim, activation='relu'):
        """Initialize a neural network layer."""
        # Xavier initialization
        scale = np.sqrt(2.0 / (input_dim + output_dim))
        self.weights = np.random.randn(input_dim, output_dim) * scale
        self.biases = np.zeros(output_dim)
        
        # Select activation function
        if activation == 'relu':
            self.activation = relu
            self.activation_derivative = relu_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_derivative = tanh_derivative
        else:
            self.activation = lambda x: x
            self.activation_derivative = lambda x: 1
        
        # For backpropagation
        self.input = None
        self.output = None
        self.gradients = {'weights': np.zeros_like(self.weights),
                         'biases': np.zeros_like(self.biases)}
    
    def forward(self, x):
        """Forward pass through the layer."""
        self.input = x
        self.output_pre_activation = np.dot(x, self.weights) + self.biases
        self.output = self.activation(self.output_pre_activation)
        return self.output
    
    def backward(self, grad_output):
        """Backward pass through the layer."""
        try:
            # Ensure grad_output has correct shape
            if len(grad_output.shape) == 1:
                grad_output = grad_output.reshape(1, -1)
            
            # Gradient of activation
            grad_activation = grad_output * self.activation_derivative(self.output_pre_activation)
            
            # Ensure input has correct shape for batch processing
            if len(self.input.shape) == 1:
                self.input = self.input.reshape(1, -1)
            
            # Gradient of weights and biases
            self.gradients['weights'] = np.dot(self.input.T, grad_activation)
            self.gradients['biases'] = np.sum(grad_activation, axis=0)
            
            # Gradient for next layer
            grad_input = np.dot(grad_activation, self.weights.T)
            
            return grad_input
            
        except Exception as e:
            print(f"Error in Layer backward: {e}")
            print(f"grad_output shape: {grad_output.shape}")
            print(f"input shape: {self.input.shape}")
            print(f"weights shape: {self.weights.shape}")
            return np.zeros_like(self.input)

class GaussianPolicy:
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        """Initialize policy network."""
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Initialize weights and biases
        self.W1 = np.random.randn(state_dim, hidden_dim) / np.sqrt(state_dim)
        self.b1 = np.zeros(hidden_dim)
        
        self.W2 = np.random.randn(hidden_dim, hidden_dim) / np.sqrt(hidden_dim)
        self.b2 = np.zeros(hidden_dim)
        
        self.W_mean = np.random.randn(hidden_dim, action_dim) / np.sqrt(hidden_dim)
        self.b_mean = np.zeros(action_dim)
        
        self.W_std = np.random.randn(hidden_dim, action_dim) / np.sqrt(hidden_dim)
        self.b_std = np.zeros(action_dim)
    
    def forward(self, state):
        """Forward pass through network."""
        # First hidden layer
        x = np.dot(state, self.W1) + self.b1
        x = np.tanh(x)
        
        # Second hidden layer
        x = np.dot(x, self.W2) + self.b2
        x = np.tanh(x)
        
        # Output layers
        mean = np.dot(x, self.W_mean) + self.b_mean
        mean = np.tanh(mean) * 0.5 + 0.5  # Scale to [0, 1]
        
        std = np.dot(x, self.W_std) + self.b_std
        std = np.exp(np.clip(std, -20, 2))  # Ensure positive and reasonable
        
        return mean, std
    
    def sample(self, mean, std):
        """Sample action from Gaussian distribution."""
        return mean + std * np.random.randn(*mean.shape)
    
    def log_prob(self, action, mean, std):
        """Compute log probability of action under Gaussian distribution."""
        return -0.5 * (np.log(2 * np.pi) + 2 * np.log(std) + ((action - mean) / std) ** 2)
    
    def get_gradients(self, state, delta):
        """Compute gradients for policy update."""
        # Forward pass storing intermediates
        h1 = np.dot(state, self.W1) + self.b1
        h1_act = np.tanh(h1)
        
        h2 = np.dot(h1_act, self.W2) + self.b2
        h2_act = np.tanh(h2)
        
        mean = np.dot(h2_act, self.W_mean) + self.b_mean
        mean = np.tanh(mean) * 0.5 + 0.5
        
        std = np.dot(h2_act, self.W_std) + self.b_std
        std = np.exp(np.clip(std, -20, 2))
        
        # Backward pass
        d_mean = delta
        d_std = delta
        
        # Mean path
        d_h2_mean = np.dot(d_mean, self.W_mean.T)
        d_W_mean = np.dot(h2_act.T, d_mean)
        d_b_mean = np.sum(d_mean, axis=0)
        
        # Std path
        d_h2_std = np.dot(d_std, self.W_std.T)
        d_W_std = np.dot(h2_act.T, d_std)
        d_b_std = np.sum(d_std, axis=0)
        
        # Combined gradients for h2
        d_h2 = d_h2_mean + d_h2_std
        d_h2_act = d_h2 * (1 - h2_act**2)
        
        # Second layer
        d_h1 = np.dot(d_h2_act, self.W2.T)
        d_W2 = np.dot(h1_act.T, d_h2_act)
        d_b2 = np.sum(d_h2_act, axis=0)
        
        # First layer
        d_h1_act = d_h1 * (1 - h1_act**2)
        d_W1 = np.dot(state.T, d_h1_act)
        d_b1 = np.sum(d_h1_act, axis=0)
        
        return {
            'W1': d_W1, 'b1': d_b1,
            'W2': d_W2, 'b2': d_b2,
            'W_mean': d_W_mean, 'b_mean': d_b_mean,
            'W_std': d_W_std, 'b_std': d_b_std
        }
    
    def apply_gradients(self, grads, learning_rate):
        """Apply computed gradients."""
        self.W1 += learning_rate * grads['W1']
        self.b1 += learning_rate * grads['b1']
        self.W2 += learning_rate * grads['W2']
        self.b2 += learning_rate * grads['b2']
        self.W_mean += learning_rate * grads['W_mean']
        self.b_mean += learning_rate * grads['b_mean']
        self.W_std += learning_rate * grads['W_std']
        self.b_std += learning_rate * grads['b_std']

class ValueNetwork:
    def __init__(self, state_dim, hidden_dim=64):
        """Initialize value network."""
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        
        # Initialize weights and biases
        self.W1 = np.random.randn(state_dim, hidden_dim) / np.sqrt(state_dim)
        self.b1 = np.zeros(hidden_dim)
        
        self.W2 = np.random.randn(hidden_dim, hidden_dim) / np.sqrt(hidden_dim)
        self.b2 = np.zeros(hidden_dim)
        
        self.W3 = np.random.randn(hidden_dim, 1) / np.sqrt(hidden_dim)
        self.b3 = np.zeros(1)
    
    def forward(self, state):
        """Forward pass through network."""
        # First hidden layer
        x = np.dot(state, self.W1) + self.b1
        x = np.tanh(x)
        
        # Second hidden layer
        x = np.dot(x, self.W2) + self.b2
        x = np.tanh(x)
        
        # Output layer (with value clipping)
        value = np.dot(x, self.W3) + self.b3
        value = np.clip(value, -10.0, 10.0)  # Clip value predictions
        return value
    
    def get_gradients(self, state, delta):
        """Compute gradients for value update."""
        # Forward pass storing intermediates
        h1 = np.dot(state, self.W1) + self.b1
        h1_act = np.tanh(h1)
        
        h2 = np.dot(h1_act, self.W2) + self.b2
        h2_act = np.tanh(h2)
        
        value = np.dot(h2_act, self.W3) + self.b3
        
        # Backward pass
        d_value = delta
        
        # Output layer
        d_h2 = np.dot(d_value, self.W3.T)
        d_W3 = np.dot(h2_act.T, d_value)
        d_b3 = np.sum(d_value, axis=0)
        
        # Second layer
        d_h2_act = d_h2 * (1 - h2_act**2)
        d_h1 = np.dot(d_h2_act, self.W2.T)
        d_W2 = np.dot(h1_act.T, d_h2_act)
        d_b2 = np.sum(d_h2_act, axis=0)
        
        # First layer
        d_h1_act = d_h1 * (1 - h1_act**2)
        d_W1 = np.dot(state.T, d_h1_act)
        d_b1 = np.sum(d_h1_act, axis=0)
        
        return {
            'W1': d_W1, 'b1': d_b1,
            'W2': d_W2, 'b2': d_b2,
            'W3': d_W3, 'b3': d_b3
        }
    
    def apply_gradients(self, grads, learning_rate):
        """Apply computed gradients."""
        self.W1 += learning_rate * grads['W1']
        self.b1 += learning_rate * grads['b1']
        self.W2 += learning_rate * grads['W2']
        self.b2 += learning_rate * grads['b2']
        self.W3 += learning_rate * grads['W3']
        self.b3 += learning_rate * grads['b3']

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """Compute Generalized Advantage Estimation."""
    # Ensure inputs are numpy arrays and clip values
    rewards = np.clip(rewards, -10.0, 10.0)
    values = np.clip(values, -10.0, 10.0)
    
    advantages = np.zeros_like(rewards)
    last_advantage = 0
    last_value = values[-1] if len(values) > 0 else 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = last_value
        else:
            next_value = values[t + 1]
        
        # Compute TD error with clipping
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        delta = np.clip(delta, -1.0, 1.0)  # Clip TD error
        
        # Compute advantage with clipping
        advantages[t] = delta + gamma * lam * (1 - dones[t]) * last_advantage
        advantages[t] = np.clip(advantages[t], -10.0, 10.0)  # Clip advantages
        last_advantage = advantages[t]
    
    returns = advantages + values
    returns = np.clip(returns, -10.0, 10.0)  # Clip returns
    
    # Normalize advantages
    if len(advantages) > 0:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = np.clip(advantages, -3.0, 3.0)  # Clip normalized advantages
    
    return advantages, returns 