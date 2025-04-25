import os
import json
from datetime import datetime
from stable_baselines3 import PPO
import numpy as np
from typing import Dict, Optional
import shutil

class ModelHandler:
    def __init__(self, base_dir: str = "models"):
        """Initialize model handler with directory structure."""
        self.base_dir = base_dir
        self.models_dir = os.path.join(base_dir, "saved_models")
        self.metrics_dir = os.path.join(base_dir, "training_metrics")
        self.checkpoint_dir = os.path.join(base_dir, "checkpoints")
        
        # Create directories if they don't exist
        for directory in [self.models_dir, self.metrics_dir, self.checkpoint_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def save_model(self, model: PPO, metrics: Dict, version: Optional[str] = None) -> str:
        """Save a trained model with its metrics."""
        # Generate version if not provided
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model
        model_path = os.path.join(self.models_dir, f"ppo_model_{version}")
        model.save(model_path)
        
        # Save metrics
        metrics_path = os.path.join(self.metrics_dir, f"metrics_{version}.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Saved model and metrics with version: {version}")
        return version
    
    def load_latest_model(self) -> tuple[PPO, Dict]:
        """Load the most recent model and its metrics."""
        # Get all model files
        model_files = [f for f in os.listdir(self.models_dir) if f.startswith("ppo_model_")]
        if not model_files:
            raise FileNotFoundError("No saved models found")
        
        # Get latest version
        latest_version = sorted(model_files)[-1].replace("ppo_model_", "")
        return self.load_model(latest_version)
    
    def load_model(self, version: str) -> tuple[PPO, Dict]:
        """Load a specific model version and its metrics."""
        model_path = os.path.join(self.models_dir, f"ppo_model_{version}")
        metrics_path = os.path.join(self.metrics_dir, f"metrics_{version}.json")
        
        # Load model
        model = PPO.load(model_path)
        
        # Load metrics
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        print(f"Loaded model version: {version}")
        return model, metrics
    
    def save_checkpoint(self, model: PPO, metrics: Dict, step: int):
        """Save a training checkpoint."""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_{step}")
        model.save(checkpoint_path)
        
        metrics_path = os.path.join(self.checkpoint_dir, f"metrics_{step}.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def load_checkpoint(self, step: int) -> tuple[PPO, Dict]:
        """Load a specific checkpoint."""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_{step}")
        metrics_path = os.path.join(self.checkpoint_dir, f"metrics_{step}.json")
        
        model = PPO.load(checkpoint_path)
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        return model, metrics
    
    def list_available_models(self) -> Dict[str, Dict]:
        """List all available models and their metrics summaries."""
        models = {}
        for model_file in os.listdir(self.models_dir):
            if model_file.startswith("ppo_model_"):
                version = model_file.replace("ppo_model_", "")
                metrics_path = os.path.join(self.metrics_dir, f"metrics_{version}.json")
                
                if os.path.exists(metrics_path):
                    with open(metrics_path, 'r') as f:
                        metrics = json.load(f)
                    models[version] = {
                        'created_at': version,
                        'metrics_summary': {
                            'avg_reward': metrics.get('avg_reward', None),
                            'success_rate': metrics.get('success_rate', None)
                        }
                    }
        return models
    
    def cleanup_old_checkpoints(self, keep_last_n: int = 5):
        """Remove old checkpoints, keeping only the n most recent ones."""
        checkpoints = [f for f in os.listdir(self.checkpoint_dir) if f.startswith("checkpoint_")]
        if len(checkpoints) <= keep_last_n:
            return
        
        # Sort checkpoints by step number
        checkpoints.sort(key=lambda x: int(x.split('_')[1]))
        
        # Remove old checkpoints
        for checkpoint in checkpoints[:-keep_last_n]:
            os.remove(os.path.join(self.checkpoint_dir, checkpoint))
            metrics_file = f"metrics_{checkpoint.split('_')[1]}.json"
            metrics_path = os.path.join(self.checkpoint_dir, metrics_file)
            if os.path.exists(metrics_path):
                os.remove(metrics_path) 