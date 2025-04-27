import os
import json
import numpy as np
from datetime import datetime

class ModelHandler:
    def __init__(self, model_dir='models'):
        """Initialize model handler with directory structure."""
        self.model_dir = model_dir
        self.checkpoints_dir = os.path.join(model_dir, 'checkpoints')
        self.versions_dir = os.path.join(model_dir, 'versions')
        
        # Create directories if they don't exist
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.versions_dir, exist_ok=True)
    
    def _save_network_weights(self, network, prefix, save_path):
        """Save weights for a single network."""
        weights_dict = {}
        for i, layer in enumerate(network.layers):
            weights_dict[f'{prefix}_layer_{i}_weights'] = layer.weights
            weights_dict[f'{prefix}_layer_{i}_biases'] = layer.biases
        np.savez(save_path, **weights_dict)
    
    def _load_network_weights(self, network, prefix, load_path):
        """Load weights for a single network."""
        weights_dict = np.load(load_path)
        for i, layer in enumerate(network.layers):
            layer.weights = weights_dict[f'{prefix}_layer_{i}_weights']
            layer.biases = weights_dict[f'{prefix}_layer_{i}_biases']
    
    def save_model(self, policy_net, value_net, metrics=None, version=None):
        """Save model with optional metrics and version."""
        if version is None:
            version = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        save_dir = os.path.join(self.versions_dir, str(version))
        os.makedirs(save_dir, exist_ok=True)
        
        # Save network weights
        weights_path = os.path.join(save_dir, 'weights.npz')
        weights_dict = {}
        
        # Save policy network weights
        for i, layer in enumerate(policy_net.layers):
            weights_dict[f'policy_layer_{i}_weights'] = layer.weights
            weights_dict[f'policy_layer_{i}_biases'] = layer.biases
        
        # Save value network weights
        for i, layer in enumerate(value_net.layers):
            weights_dict[f'value_layer_{i}_weights'] = layer.weights
            weights_dict[f'value_layer_{i}_biases'] = layer.biases
        
        np.savez(weights_path, **weights_dict)
        
        # Save metrics if provided
        if metrics is not None:
            metrics_path = os.path.join(save_dir, 'metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f)
        
        return version
    
    def save_checkpoint(self, policy_net, value_net, metrics=None):
        """Save training checkpoint."""
        checkpoint_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = os.path.join(self.checkpoints_dir, checkpoint_id)
        os.makedirs(save_dir, exist_ok=True)
        
        # Save network weights
        weights_path = os.path.join(save_dir, 'weights.npz')
        weights_dict = {}
        
        # Save policy network weights
        for i, layer in enumerate(policy_net.layers):
            weights_dict[f'policy_layer_{i}_weights'] = layer.weights
            weights_dict[f'policy_layer_{i}_biases'] = layer.biases
        
        # Save value network weights
        for i, layer in enumerate(value_net.layers):
            weights_dict[f'value_layer_{i}_weights'] = layer.weights
            weights_dict[f'value_layer_{i}_biases'] = layer.biases
        
        np.savez(weights_path, **weights_dict)
        
        # Save metrics if provided
        if metrics is not None:
            metrics_path = os.path.join(save_dir, 'metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f)
        
        return checkpoint_id
    
    def load_model(self, policy_net, value_net, version):
        """Load model from version."""
        load_dir = os.path.join(self.versions_dir, str(version))
        weights_path = os.path.join(load_dir, 'weights.npz')
        
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"No model found for version {version}")
        
        # Load weights
        weights_dict = np.load(weights_path)
        
        # Load policy network weights
        for i, layer in enumerate(policy_net.layers):
            layer.weights = weights_dict[f'policy_layer_{i}_weights']
            layer.biases = weights_dict[f'policy_layer_{i}_biases']
        
        # Load value network weights
        for i, layer in enumerate(value_net.layers):
            layer.weights = weights_dict[f'value_layer_{i}_weights']
            layer.biases = weights_dict[f'value_layer_{i}_biases']
        
        # Load metrics if available
        metrics_path = os.path.join(load_dir, 'metrics.json')
        metrics = None
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
        
        return metrics
    
    def load_checkpoint(self, policy_net, value_net, checkpoint_id):
        """Load checkpoint."""
        load_dir = os.path.join(self.checkpoints_dir, checkpoint_id)
        weights_path = os.path.join(load_dir, 'weights.npz')
        
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"No checkpoint found with ID {checkpoint_id}")
        
        # Load weights
        weights_dict = np.load(weights_path)
        
        # Load policy network weights
        for i, layer in enumerate(policy_net.layers):
            layer.weights = weights_dict[f'policy_layer_{i}_weights']
            layer.biases = weights_dict[f'policy_layer_{i}_biases']
        
        # Load value network weights
        for i, layer in enumerate(value_net.layers):
            layer.weights = weights_dict[f'value_layer_{i}_weights']
            layer.biases = weights_dict[f'value_layer_{i}_biases']
        
        # Load metrics if available
        metrics_path = os.path.join(load_dir, 'metrics.json')
        metrics = None
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
        
        return metrics
    
    def list_versions(self):
        """List available model versions."""
        if not os.path.exists(self.versions_dir):
            return []
        return sorted(os.listdir(self.versions_dir))
    
    def list_checkpoints(self):
        """List available checkpoints."""
        if not os.path.exists(self.checkpoints_dir):
            return []
        return sorted(os.listdir(self.checkpoints_dir)) 