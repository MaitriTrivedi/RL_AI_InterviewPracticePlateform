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
        # Gradient of activation
        grad_activation = grad_output * self.activation_derivative(self.output_pre_activation)
        
        # Gradient of weights and biases
        self.gradients['weights'] = np.dot(self.input.T, grad_activation)
        self.gradients['biases'] = np.sum(grad_activation, axis=0)
        
        # Gradient for next layer
        grad_input = np.dot(grad_activation, self.weights.T)
        return grad_input

class GaussianPolicy:
    def __init__(self, state_dim, hidden_dim=64, learning_rate=0.001):
        """Initialize Gaussian policy network."""
        self.layers = [
            Layer(state_dim, hidden_dim, 'relu'),
            Layer(hidden_dim, hidden_dim, 'relu'),
            Layer(hidden_dim, 2)  # Output mean and log_std
        ]
        self.learning_rate = learning_rate
    
    def forward(self, state):
        """Forward pass to get mean and standard deviation."""
        x = np.array(state).reshape(-1, state.shape[-1])
        for layer in self.layers[:-1]:
            x = layer.forward(x)
        
        output = self.layers[-1].forward(x)
        mean = output[:, 0]
        log_std = np.clip(output[:, 1], -20, 2)  # Clip log_std for numerical stability
        std = np.exp(log_std)
        
        return mean, std
    
    def sample(self, mean, std):
        """Sample action from Gaussian distribution."""
        noise = np.random.randn(*mean.shape)
        action = mean + std * noise
        return np.clip(action, -3, 3)  # Clip actions for stability
    
    def log_prob(self, action, mean, std):
        """Compute log probability of action under Gaussian distribution."""
        var = np.maximum(std ** 2, 1e-6)  # Add small constant for numerical stability
        log_density = -0.5 * (
            ((action - mean) ** 2) / var + 
            2 * np.log(std) + 
            np.log(2 * np.pi)
        )
        return np.clip(log_density, -20, 20)  # Clip for numerical stability
    
    def backward(self, policy_grad, entropy_grad):
        """Backward pass through the network."""
        grad = np.stack([policy_grad, entropy_grad], axis=-1)
        grad = np.clip(grad, -1, 1)  # Clip gradients for stability
        
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
            
            # Update weights and biases with gradient clipping
            for param in ['weights', 'biases']:
                grad_norm = np.linalg.norm(layer.gradients[param])
                if grad_norm > 1:
                    layer.gradients[param] *= 1 / grad_norm
                layer.weights -= self.learning_rate * layer.gradients[param]
                layer.biases -= self.learning_rate * layer.gradients[param]

class ValueNetwork:
    def __init__(self, state_dim, hidden_dim=64, learning_rate=0.001):
        """Initialize value network."""
        self.layers = [
            Layer(state_dim, hidden_dim, 'relu'),
            Layer(hidden_dim, hidden_dim, 'relu'),
            Layer(hidden_dim, 1)
        ]
        self.learning_rate = learning_rate
    
    def forward(self, state):
        """Forward pass to get value estimate."""
        x = np.array(state).reshape(-1, state.shape[-1])
        for layer in self.layers:
            x = layer.forward(x)
        return x.squeeze()
    
    def backward(self, value_grad):
        """Backward pass through the network."""
        grad = value_grad.reshape(-1, 1)
        grad = np.clip(grad, -1, 1)  # Clip gradients for stability
        
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
            
            # Update weights and biases with gradient clipping
            for param in ['weights', 'biases']:
                grad_norm = np.linalg.norm(layer.gradients[param])
                if grad_norm > 1:
                    layer.gradients[param] *= 1 / grad_norm
                layer.weights -= self.learning_rate * layer.gradients[param]
                layer.biases -= self.learning_rate * layer.gradients[param]

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """Compute Generalized Advantage Estimation."""
    advantages = np.zeros_like(rewards)
    last_advantage = 0
    last_value = values[-1] if len(values) > 0 else 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = last_value
        else:
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        advantages[t] = delta + gamma * lam * (1 - dones[t]) * last_advantage
        last_advantage = advantages[t]
    
    returns = advantages + values
    
    # Normalize advantages
    if len(advantages) > 0:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    return advantages, returns 