import os
import json
import torch
import numpy as np
from datetime import datetime

class ModelHandler:
    def __init__(self, base_dir="models"):
        """Initialize the model handler with directory structure."""
        self.base_dir = base_dir
        self.models_dir = os.path.join(base_dir, "saved_models")
        self.metrics_dir = os.path.join(base_dir, "training_metrics")
        self.checkpoints_dir = os.path.join(base_dir, "checkpoints")
        
        # Create directories if they don't exist
        for directory in [self.base_dir, self.models_dir, self.metrics_dir, self.checkpoints_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
    
    def get_latest_model_version(self):
        """Get the version number of the latest model."""
        try:
            model_files = os.listdir(self.models_dir)
            if not model_files:
                return None
            versions = [f.split('_model.pt')[0] for f in model_files if f.endswith('_model.pt')]
            return max(versions) if versions else None
        except Exception as e:
            print(f"Error getting latest model version: {e}")
            return None
    
    def load_model(self, model, version=None):
        """Load model weights and metrics."""
        try:
            if version is None:
                version = self.get_latest_model_version()
            
            if version is None:
                raise ValueError("No models found")
            
            model_path = os.path.join(self.models_dir, f"{version}_model.pt")
            if not os.path.exists(model_path):
                raise ValueError(f"Model version {version} not found")
            
            # Load model weights
            model.load_state_dict(torch.load(model_path))
            
            # Load metrics
            metrics = {}
            metrics_path = os.path.join(self.metrics_dir, f"{version}_metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
            
            return metrics
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return {}
    
    def save_model(self, model, metrics=None, version=None):
        """Save model weights and metrics."""
        try:
            if version is None:
                version = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save model weights
            model_path = os.path.join(self.models_dir, f"{version}_model.pt")
            torch.save(model.state_dict(), model_path)
            
            # Save metrics
            if metrics:
                metrics_path = os.path.join(self.metrics_dir, f"{version}_metrics.json")
                with open(metrics_path, 'w') as f:
                    json.dump(metrics, f)
            
            return version
            
        except Exception as e:
            print(f"Error saving model: {e}")
            return None
    
    def save_checkpoint(self, model, metrics=None, step=0):
        """Save a checkpoint during training."""
        try:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_path = os.path.join(self.checkpoints_dir, f"checkpoint_{step}_{version}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            
            if metrics:
                metrics_path = os.path.join(self.checkpoints_dir, f"checkpoint_{step}_{version}_metrics.json")
                with open(metrics_path, 'w') as f:
                    json.dump(metrics, f)
            
            return version
            
        except Exception as e:
            print(f"Error saving checkpoint: {e}")
            return None 