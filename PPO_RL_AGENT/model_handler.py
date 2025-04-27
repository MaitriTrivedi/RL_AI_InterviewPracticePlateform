import os
import numpy as np
from datetime import datetime

class ModelHandler:
    def __init__(self, base_dir="models"):
        """Initialize model handler."""
        self.base_dir = base_dir
        self.checkpoints_dir = os.path.join(base_dir, "checkpoints")
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)
    
    def save_checkpoint(self, agent, metadata):
        """Save training checkpoint."""
        # Create checkpoint filename
        timestamp = metadata.get('timestamp', datetime.now().strftime('%Y%m%d_%H%M%S'))
        interview_num = metadata.get('interview_num', 0)
        filename = f"checkpoint_interview_{interview_num}_{timestamp}.npy"
        path = os.path.join(self.checkpoints_dir, filename)
        
        # Save checkpoint
        agent.save_checkpoint(path)
        return path
    
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
        np.save(os.path.join(model_dir, "policy_net.npy"), {
            'W1': policy_net.W1,
            'b1': policy_net.b1,
            'W2': policy_net.W2,
            'b2': policy_net.b2,
            'W_mean': policy_net.W_mean,
            'b_mean': policy_net.b_mean,
            'W_std': policy_net.W_std,
            'b_std': policy_net.b_std
        })
        
        np.save(os.path.join(model_dir, "value_net.npy"), {
            'W1': value_net.W1,
            'b1': value_net.b1,
            'W2': value_net.W2,
            'b2': value_net.b2,
            'W3': value_net.W3,
            'b3': value_net.b3
        })
        
        np.save(os.path.join(model_dir, "metrics.npy"), metrics)
        return version
    
    def load_model(self, policy_net, value_net, version):
        """Load model from version."""
        try:
            model_dir = os.path.join(self.base_dir, version)
            
            # Load policy network weights
            policy_weights = np.load(os.path.join(model_dir, "policy_net.npy"), allow_pickle=True).item()
            policy_net.W1 = policy_weights['W1']
            policy_net.b1 = policy_weights['b1']
            policy_net.W2 = policy_weights['W2']
            policy_net.b2 = policy_weights['b2']
            policy_net.W_mean = policy_weights['W_mean']
            policy_net.b_mean = policy_weights['b_mean']
            policy_net.W_std = policy_weights['W_std']
            policy_net.b_std = policy_weights['b_std']
            
            # Load value network weights
            value_weights = np.load(os.path.join(model_dir, "value_net.npy"), allow_pickle=True).item()
            value_net.W1 = value_weights['W1']
            value_net.b1 = value_weights['b1']
            value_net.W2 = value_weights['W2']
            value_net.b2 = value_weights['b2']
            value_net.W3 = value_weights['W3']
            value_net.b3 = value_weights['b3']
            
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def get_training_progress(self):
        """Get training progress statistics."""
        if not self.history['versions']:
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
    
    def save_checkpoint(self, policy_net, value_net, metrics=None):
        """Save training checkpoint."""
        checkpoint_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_dir = os.path.join(self.checkpoints_dir, checkpoint_id)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save network weights
        self._save_network_weights(policy_net, 'policy', os.path.join(checkpoint_dir, 'policy_weights.json'))
        self._save_network_weights(value_net, 'value', os.path.join(checkpoint_dir, 'value_weights.json'))
        
        # Save metrics if provided
        if metrics:
            with open(os.path.join(checkpoint_dir, 'metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=2)
        
        return checkpoint_id
    
    def load_checkpoint(self, policy_net, value_net, checkpoint_id):
        """Load checkpoint."""
        checkpoint_dir = os.path.join(self.checkpoints_dir, checkpoint_id)
        if not os.path.exists(checkpoint_dir):
            raise FileNotFoundError(f"No checkpoint found with ID {checkpoint_id}")
        
        # Load network weights
        self._load_network_weights(policy_net, 'policy', os.path.join(checkpoint_dir, 'policy_weights.json'))
        self._load_network_weights(value_net, 'value', os.path.join(checkpoint_dir, 'value_weights.json'))
        
        # Load metrics if available
        metrics_path = os.path.join(checkpoint_dir, 'metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                return json.load(f)
        return None
    
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