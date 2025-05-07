import os
import torch
import json
from datetime import datetime
import numpy as np
import logging

class ModelHandler:
    def __init__(self, models_dir=None):
        """Initialize model handler with models directory."""
        # Default models directory is now in main data/models
        self.models_dir = models_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'data',
            'models'
        )
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
            
        # Create subdirectories
        self.checkpoints_dir = os.path.join(self.models_dir, 'checkpoints')
        self.history_dir = os.path.join(self.models_dir, 'history')
        self.versions_dir = os.path.join(self.models_dir, 'versions')
        
        for directory in [self.checkpoints_dir, self.history_dir, self.versions_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                
    def save_checkpoint(self, model, metadata):
        """Save a training checkpoint."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_path = os.path.join(
            self.checkpoints_dir,
            f'checkpoint_{metadata["interview_num"]}_{timestamp}.npy'
        )
        
        # Save model state
        np.save(checkpoint_path, {
            'model_state': {
                'policy': {
                    'W1': model.policy.W1,
                    'b1': model.policy.b1,
                    'W2': model.policy.W2,
                    'b2': model.policy.b2,
                    'W_mean': model.policy.W_mean,
                    'b_mean': model.policy.b_mean,
                    'W_std': model.policy.W_std,
                    'b_std': model.policy.b_std
                },
                'value_net': {
                    'W1': model.value_net.W1,
                    'b1': model.value_net.b1,
                    'W2': model.value_net.W2,
                    'b2': model.value_net.b2,
                    'W3': model.value_net.W3,
                    'b3': model.value_net.b3
                }
            },
            'metadata': metadata
        })
        
        # Save metadata separately for easy access
        metadata_path = os.path.join(
            self.history_dir,
            f'checkpoint_{metadata["interview_num"]}_{timestamp}.json'
        )
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        return checkpoint_path
        
    def save_model(self, policy_net, value_net, metrics, path=None):
        """Save model with weights and metrics."""
        try:
            # Create models directory if it doesn't exist
            if path is None:
                # Generate default path if none provided
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                mean_reward = metrics.get('mean_reward', 0.0)
                version = f"model_v1_{timestamp}_reward_{mean_reward:.3f}"
                path = os.path.join(self.versions_dir, f"{version}.npy")  # Save in versions subdirectory
            
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Prepare model state
            model_state = {
                'policy': {
                    'W1': policy_net.W1,
                    'b1': policy_net.b1,
                    'W2': policy_net.W2,
                    'b2': policy_net.b2,
                    'W_mean': policy_net.W_mean,
                    'b_mean': policy_net.b_mean,
                    'W_std': policy_net.W_std,
                    'b_std': policy_net.b_std
                },
                'value_net': {
                    'W1': value_net.W1,
                    'b1': value_net.b1,
                    'W2': value_net.W2,
                    'b2': value_net.b2,
                    'W3': value_net.W3,
                    'b3': value_net.b3
                },
                'metrics': metrics,
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
            }
            
            # Save model
            np.save(path, model_state)
            logging.info(f"Model saved successfully at: {path}")
            
            # Save metadata separately for easy access
            metadata_path = os.path.join(self.history_dir, f"{os.path.splitext(os.path.basename(path))[0]}_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump({
                    'version': os.path.splitext(os.path.basename(path))[0],
                    'metrics': metrics,
                    'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
                }, f, indent=2)
            
            # Extract version from path
            version = os.path.splitext(os.path.basename(path))[0]
            return version
            
        except Exception as e:
            logging.error(f"Error in save_model: {e}")
            raise
        
    def load_model(self, policy_net, value_net, version):
        """Load a specific model version."""
        try:
            # Try different path variations
            possible_paths = [
                version,  # Direct path
                os.path.join(self.models_dir, version),  # Version as filename
                os.path.join(self.models_dir, f"{version}.npy"),  # Version with .npy
                os.path.join(self.models_dir, "versions", version),  # In versions subdirectory
                os.path.join(self.models_dir, "versions", f"{version}.npy")  # In versions with .npy
            ]
            
            model_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            
            if model_path is None:
                raise ValueError(f"Model version {version} not found. Tried paths: {possible_paths}")
            
            logging.info(f"Loading model from: {model_path}")
            checkpoint = np.load(model_path, allow_pickle=True).item()
            
            # Load policy weights
            policy_net.W1 = checkpoint['policy']['W1']
            policy_net.b1 = checkpoint['policy']['b1']
            policy_net.W2 = checkpoint['policy']['W2']
            policy_net.b2 = checkpoint['policy']['b2']
            policy_net.W_mean = checkpoint['policy']['W_mean']
            policy_net.b_mean = checkpoint['policy']['b_mean']
            policy_net.W_std = checkpoint['policy']['W_std']
            policy_net.b_std = checkpoint['policy']['b_std']
            
            # Load value network weights
            value_net.W1 = checkpoint['value_net']['W1']
            value_net.b1 = checkpoint['value_net']['b1']
            value_net.W2 = checkpoint['value_net']['W2']
            value_net.b2 = checkpoint['value_net']['b2']
            value_net.W3 = checkpoint['value_net']['W3']
            value_net.b3 = checkpoint['value_net']['b3']
            
            logging.info(f"Model loaded successfully from: {model_path}")
            return checkpoint.get('metrics', {})
            
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise
    
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