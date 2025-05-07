import numpy as np
from collections import defaultdict
from datetime import datetime
import os

class TrainingHistory:
    def __init__(self):
        """Initialize training history tracker."""
        # Episode-level metrics
        self.episode_rewards = []
        self.episode_lengths = []
        
        # Detailed performance metrics
        self.topic_performances = defaultdict(list)  # Performance by topic
        self.difficulty_history = []  # Track difficulty progression
        self.time_efficiency = []  # Track time management
        
        # Training metrics
        self.policy_losses = []
        self.value_losses = []
        self.entropy_losses = []
        
        # Periodic snapshots for analysis
        self.snapshots = []
    
    def add_episode(self, total_reward, length, topic_scores, difficulties, time_stats):
        """Add episode results to history."""
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(length)
        
        # Update topic performances
        for topic, score in topic_scores.items():
            self.topic_performances[topic].append(score)
        
        # Track difficulty adjustments
        self.difficulty_history.extend(difficulties)
        
        # Track time efficiency
        self.time_efficiency.extend(time_stats)
    
    def add_training_metrics(self, policy_loss, value_loss, entropy_loss):
        """Add training metrics from an update step."""
        self.policy_losses.append(policy_loss)
        self.value_losses.append(value_loss)
        self.entropy_losses.append(entropy_loss)
    
    def create_snapshot(self):
        """Create a snapshot of current performance metrics."""
        snapshot = {
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'avg_reward': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0,
            'avg_length': np.mean(self.episode_lengths[-100:]) if self.episode_lengths else 0,
            'topic_performances': {
                topic: np.mean(scores[-100:]) if scores else 0
                for topic, scores in self.topic_performances.items()
            },
            'avg_difficulty': np.mean(self.difficulty_history[-100:]) if self.difficulty_history else 0,
            'avg_time_efficiency': np.mean(self.time_efficiency[-100:]) if self.time_efficiency else 0,
            'training_metrics': {
                'policy_loss': np.mean(self.policy_losses[-100:]) if self.policy_losses else 0,
                'value_loss': np.mean(self.value_losses[-100:]) if self.value_losses else 0,
                'entropy_loss': np.mean(self.entropy_losses[-100:]) if self.entropy_losses else 0
            }
        }
        self.snapshots.append(snapshot)
        return snapshot
    
    def get_performance_summary(self):
        """Get a summary of recent performance."""
        if not self.episode_rewards:
            return {
                'avg_reward': 0,
                'avg_length': 0,
                'topic_performances': {},
                'avg_difficulty': 0,
                'avg_time_efficiency': 0
            }
        
        return {
            'avg_reward': float(np.mean(self.episode_rewards[-100:])),
            'avg_length': float(np.mean(self.episode_lengths[-100:])),
            'topic_performances': {
                topic: float(np.mean(scores[-100:]))
                for topic, scores in self.topic_performances.items()
            },
            'avg_difficulty': float(np.mean(self.difficulty_history[-100:])),
            'avg_time_efficiency': float(np.mean(self.time_efficiency[-100:]))
        }
    
    def get_training_metrics(self):
        """Get recent training metrics."""
        return {
            'policy_loss': float(np.mean(self.policy_losses[-100:])) if self.policy_losses else 0,
            'value_loss': float(np.mean(self.value_losses[-100:])) if self.value_losses else 0,
            'entropy_loss': float(np.mean(self.entropy_losses[-100:])) if self.entropy_losses else 0
        }
    
    def save_to_file(self, filepath=None):
        """Save history to a file."""
        # Use main data directory if filepath not specified
        if filepath is None:
            filepath = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                'data',
                'training_history',
                f'history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.npz'
            )
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        history_data = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'topic_performances': dict(self.topic_performances),
            'difficulty_history': self.difficulty_history,
            'time_efficiency': self.time_efficiency,
            'policy_losses': self.policy_losses,
            'value_losses': self.value_losses,
            'entropy_losses': self.entropy_losses,
            'snapshots': self.snapshots
        }
        np.savez_compressed(filepath, **history_data)
    
    def load_from_file(self, filepath=None):
        """Load history from a file."""
        # Use main data directory if filepath not specified
        if filepath is None:
            # Find most recent history file
            history_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                'data',
                'training_history'
            )
            if not os.path.exists(history_dir):
                raise ValueError("No training history directory found")
            
            history_files = sorted(os.listdir(history_dir))
            if not history_files:
                raise ValueError("No training history files found")
            
            filepath = os.path.join(history_dir, history_files[-1])
        
        data = np.load(filepath, allow_pickle=True)
        self.episode_rewards = data['episode_rewards'].tolist()
        self.episode_lengths = data['episode_lengths'].tolist()
        self.topic_performances = defaultdict(list, data['topic_performances'].item())
        self.difficulty_history = data['difficulty_history'].tolist()
        self.time_efficiency = data['time_efficiency'].tolist()
        self.policy_losses = data['policy_losses'].tolist()
        self.value_losses = data['value_losses'].tolist()
        self.entropy_losses = data['entropy_losses'].tolist()
        self.snapshots = data['snapshots'].tolist() 