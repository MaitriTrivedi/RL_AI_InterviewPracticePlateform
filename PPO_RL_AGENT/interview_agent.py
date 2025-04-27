import numpy as np
from collections import deque
from datetime import datetime
from .ppo_agent import PPOAgent
from .model_handler import ModelHandler

class InterviewAgent:
    def __init__(self, state_dim=6, model_version=None):
        """Initialize interview agent with PPO."""
        self.state_dim = state_dim
        self.action_dim = 1  # Difficulty adjustment
        
        # Initialize PPO agent
        self.agent = PPOAgent(
            state_dim=state_dim,
            action_dim=self.action_dim,
            hidden_dim=64,  # Changed from hidden_dims
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_ratio=0.2,
            value_coef=0.5,
            entropy_coef=0.01,
            batch_size=64,
            epochs=10,
            buffer_size=100  # 10 questions * 10 interviews
        )
        
        # Initialize model handler
        self.model_handler = ModelHandler()
        
        # Load model if version provided
        if model_version:
            self.load_model(model_version)
        
        # Initialize interview state
        self.topics = ['ds', 'algo', 'oops', 'dbms', 'os', 'cn', 'system_design']  # Updated topics list
        self.reset_interview_state()
    
    def reset_interview_state(self):
        """Reset interview state for new interview."""
        self.current_step = 0
        self.performances = []
        self.topic_performances = {topic: [] for topic in self.topics}
        self.time_efficiency = []
        self.difficulties = []
        self.current_state = None
        self.current_action = None
        self.current_value = None
        self.current_log_prob = None
    
    def _get_state(self, topic):
        """Get current state for decision making."""
        # Calculate features
        progress = self.current_step / 10  # 10 questions per interview
        
        # Calculate average performance
        avg_score = np.mean([p['score'] for p in self.performances]) if self.performances else 0.5
        
        # Calculate time efficiency
        time_eff = np.mean(self.time_efficiency) if self.time_efficiency else 1.0
        
        # Get topic index and performance
        topic_idx = self.topics.index(topic) / len(self.topics)
        topic_perf = np.mean(self.topic_performances[topic]) if self.topic_performances[topic] else 0.5
        
        # Get current difficulty
        current_diff = self.difficulties[-1] if self.difficulties else 5.0
        current_diff = current_diff / 10.0  # Normalize to [0, 1]
        
        return np.array([
            progress,
            avg_score,
            time_eff,
            topic_idx,
            topic_perf,
            current_diff
        ], dtype=np.float32)
    
    def get_next_question(self, topic):
        """Get difficulty for next question."""
        # Get current state
        self.current_state = self._get_state(topic)
        
        # Get action from policy
        action, value, log_prob = self.agent.select_action(self.current_state)
        
        # Store current step info
        self.current_action = action
        self.current_value = value
        self.current_log_prob = log_prob
        
        # Store initial difficulty
        difficulty = float(action[0])  # Convert to float
        self.difficulties.append(difficulty)
        
        return {
            'difficulty': difficulty,
            'value_estimate': value,
            'log_prob': log_prob
        }
    
    def update_performance(self, topic, performance_score, time_taken):
        """Update agent with question performance."""
        # Store performance
        self.performances.append({
            'score': performance_score,
            'time_taken': time_taken
        })
        
        # Update topic-specific performance
        self.topic_performances[topic].append(performance_score)
        
        # Update time efficiency
        self.time_efficiency.append(time_taken)
        
        # Calculate reward components
        performance_reward = performance_score
        difficulty_penalty = -0.1 * abs(self.difficulties[-1] - 5.0)  # Penalize deviation from medium difficulty
        time_reward = 0.1 * (1.0 - abs(1.0 - time_taken / 15.0))  # Assuming 15 minutes is standard time
        
        # Total reward
        reward = performance_reward + difficulty_penalty + time_reward
        
        # Store transition in agent's buffer
        done = self.current_step >= 9  # 10 questions per interview (0-9)
        self.agent.store_transition(
            state=self.current_state,
            action=self.current_action,
            reward=reward,
            value=self.current_value,
            log_prob=self.current_log_prob,
            done=done
        )
        
        # Update step counter
        self.current_step += 1
    
    def train(self):
        """Train the agent on collected experience."""
        if self.current_step < 10:  # Need complete interview
            return None
        
        # Train PPO agent
        metrics = self.agent.update()
        return metrics
    
    def save_checkpoint(self, interview_num):
        """Save training checkpoint."""
        stats = self.get_interview_stats()
        self.model_handler.save_checkpoint(
            self.agent,
            {
                'interview_num': interview_num,
                'stats': stats,
                'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
            }
        )
    
    def save_model(self):
        """Save final model."""
        stats = self.get_interview_stats()
        version = self.model_handler.save_model(
            self.agent,
            {
                'final_stats': stats,
                'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
            }
        )
        return version
    
    def load_model(self, version):
        """Load model from version."""
        self.agent = self.model_handler.load_model(version)
    
    def get_interview_stats(self):
        """Get current interview statistics."""
        if not self.performances:
            return {
                'average_score': 0.0,
                'time_efficiency': 1.0,
                'topic_performances': {t: 0.0 for t in self.topics}
            }
        
        return {
            'average_score': np.mean([p['score'] for p in self.performances]),
            'time_efficiency': np.mean(self.time_efficiency),
            'topic_performances': {
                t: np.mean(perfs) if perfs else 0.0
                for t, perfs in self.topic_performances.items()
            }
        } 