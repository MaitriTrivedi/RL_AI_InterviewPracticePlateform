import os
import numpy as np
import json
from datetime import datetime

class ModelHandler:
    def __init__(self, base_dir="models"):
        """Initialize model handler."""
        self.base_dir = base_dir
        self.checkpoints_dir = os.path.join(base_dir, "checkpoints")
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)
    
    def _save_network_weights(self, network, network_type, filepath):
        """Save network weights to a file."""
        if network_type == 'policy':
            weights = {
                'W1': network.W1.tolist(),
                'b1': network.b1.tolist(),
                'W2': network.W2.tolist(),
                'b2': network.b2.tolist(),
                'W_mean': network.W_mean.tolist(),
                'b_mean': network.b_mean.tolist(),
                'W_std': network.W_std.tolist(),
                'b_std': network.b_std.tolist()
            }
        else:  # value network
            weights = {
                'W1': network.W1.tolist(),
                'b1': network.b1.tolist(),
                'W2': network.W2.tolist(),
                'b2': network.b2.tolist(),
                'W3': network.W3.tolist(),
                'b3': network.b3.tolist()
            }
        
        # Save weights to file
        with open(filepath, 'w') as f:
            json.dump(weights, f, indent=2)
    
    def _load_network_weights(self, network, network_type, filepath):
        """Load network weights from file."""
        with open(filepath, 'r') as f:
            weights = json.load(f)
        
        if network_type == 'policy':
            network.W1 = np.array(weights['W1'])
            network.b1 = np.array(weights['b1'])
            network.W2 = np.array(weights['W2'])
            network.b2 = np.array(weights['b2'])
            network.W_mean = np.array(weights['W_mean'])
            network.b_mean = np.array(weights['b_mean'])
            network.W_std = np.array(weights['W_std'])
            network.b_std = np.array(weights['b_std'])
        else:  # value network
            network.W1 = np.array(weights['W1'])
            network.b1 = np.array(weights['b1'])
            network.W2 = np.array(weights['W2'])
            network.b2 = np.array(weights['b2'])
            network.W3 = np.array(weights['W3'])
            network.b3 = np.array(weights['b3'])
    
    def save_checkpoint(self, agent, metadata):
        """Save training checkpoint."""
        # Create checkpoint filename
        timestamp = metadata.get('timestamp', datetime.now().strftime('%Y%m%d_%H%M%S'))
        interview_num = metadata.get('interview_num', 0)
        checkpoint_dir = os.path.join(self.checkpoints_dir, f"checkpoint_{interview_num}_{timestamp}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save network weights
        self._save_network_weights(agent.policy, 'policy', os.path.join(checkpoint_dir, 'policy_weights.json'))
        self._save_network_weights(agent.value_net, 'value', os.path.join(checkpoint_dir, 'value_weights.json'))
        
        # Save metadata
        with open(os.path.join(checkpoint_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return checkpoint_dir
    
    def save_model(self, policy_net, value_net, metrics):
        """Save final model."""
        # Create version string
        timestamp = metrics.get('timestamp', datetime.now().strftime('%Y%m%d_%H%M%S'))
        mean_reward = metrics.get('mean_reward', 0.0)
        version = f"model_v1_{timestamp}_reward_{mean_reward:.3f}"
        
        # Save model
        model_dir = os.path.join(self.base_dir, version)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save networks and metrics
        self._save_network_weights(policy_net, 'policy', os.path.join(model_dir, 'policy_weights.json'))
        self._save_network_weights(value_net, 'value', os.path.join(model_dir, 'value_weights.json'))
        
        # Save metrics
        with open(os.path.join(model_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return version
    
    def load_model(self, policy_net, value_net, version):
        """Load model from version."""
        try:
            model_dir = os.path.join(self.base_dir, version)
            
            # Load network weights
            self._load_network_weights(policy_net, 'policy', os.path.join(model_dir, 'policy_weights.json'))
            self._load_network_weights(value_net, 'value', os.path.join(model_dir, 'value_weights.json'))
            
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def get_training_progress(self):
        """Get training progress statistics."""
        if not hasattr(self, 'history') or not self.history.get('versions'):
            return None
        
        latest = self.history['versions'][-1]
        first = self.history['versions'][0]
        
        return {
            'current_version': self.history['current_version'],
            'total_episodes': self.history['total_episodes'],
            'last_update': self.history['last_update'],
            'total_versions': len(self.history['versions']),
            'latest_metrics': latest.get('metrics'),
            'overall_improvement': {
                'mean_reward': latest['metrics']['mean_reward'] - first['metrics']['mean_reward']
                if latest.get('metrics') and first.get('metrics') else None
            }
        }
    
    def list_versions(self):
        """List available model versions."""
        if not hasattr(self, 'versions_dir'):
            return []
        return sorted(os.listdir(self.versions_dir))
    
    def list_checkpoints(self):
        """List available checkpoints."""
        if not os.path.exists(self.checkpoints_dir):
            return []
        return sorted(os.listdir(self.checkpoints_dir)) 