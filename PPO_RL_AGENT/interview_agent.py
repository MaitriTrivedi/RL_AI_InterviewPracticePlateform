import numpy as np
from collections import deque
from datetime import datetime
from ppo_agent import PPOAgent
from model_handler import ModelHandler

class InterviewAgent:
    def __init__(self, state_dim=9, model_version=None):
        """Initialize interview agent with PPO."""
        self.state_dim = state_dim
        self.action_dim = 1  # Difficulty adjustment (-2 to +2)
        
        # Initialize PPO agent
        self.agent = PPOAgent(
            state_dim=state_dim,
            action_dim=self.action_dim,
            hidden_dim=64,
            lr_actor=3e-4,
            lr_critic=1e-3,
            gamma=0.99,
            gae_lambda=0.95,
            clip_ratio=0.2,
            target_kl=0.01,
            train_actor_iterations=10,
            train_critic_iterations=10
        )
        
        # Initialize model handler
        self.model_handler = ModelHandler()
        
        # Initialize interview state
        self.topics = ['ds', 'algo', 'oops', 'dbms', 'os', 'cn', 'system_design']
        self.reset_interview_state()
        
        # Load model if version provided
        if model_version:
            self.load_model(model_version)
    
    def reset_interview_state(self):
        """Reset interview state for new interview."""
        self.current_step = 0
        self.current_score = 0.0
        self.current_streak = 0
        self.time_efficiency = 1.0
        self.current_difficulty = 5  # Start with medium difficulty
        self.question_history = {topic: 0.0 for topic in self.topics}
        self.performances = []
        self.topic_performances = {topic: [] for topic in self.topics}
        self.time_efficiency_history = []
        self.current_state = None
        self.current_action = None
        self.current_value = None
        self.current_log_prob = None
    
    def _get_state(self, topic):
        """Get current state for decision making."""
        topic_idx = self.topics.index(topic) / (len(self.topics) - 1)
        
        return np.array([
            topic_idx,                                    # Current topic (normalized)
            self.current_difficulty / 10.0,               # Current difficulty (normalized)
            self.current_score,                           # Average performance
            self.time_efficiency,                         # Time efficiency
            self.current_streak / 10.0,                   # Streak (normalized)
            *list(self.question_history.values())[-4:]    # Last 4 topics' performance
        ], dtype=np.float32)
    
    def get_next_question(self, topic):
        """Get difficulty adjustment for next question."""
        # Get current state
        self.current_state = self._get_state(topic)
        
        # Get action from policy (difficulty adjustment)
        action, value, log_prob = self.agent.select_action(self.current_state)
        
        # Store current step info
        self.current_action = action
        self.current_value = value
        self.current_log_prob = log_prob
        
        # Convert action to difficulty adjustment (-2 to +2)
        diff_adjustment = (action[0] * 4) - 2  # Scale from [0,1] to [-2,2]
        
        # Update difficulty with bounds
        new_difficulty = np.clip(self.current_difficulty + diff_adjustment, 1, 10)
        self.current_difficulty = float(new_difficulty)
        
        return {
            'difficulty': self.current_difficulty,
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
        self.question_history[topic] = np.mean(self.topic_performances[topic])
        
        # Update time efficiency
        expected_time = 5 + self.current_difficulty  # Higher difficulty = more time
        time_efficiency = max(0, 1 - abs(time_taken - expected_time) / expected_time)
        self.time_efficiency_history.append(time_efficiency)
        self.time_efficiency = np.mean(self.time_efficiency_history[-3:])  # Moving average
        
        # Update current score and streak
        self.current_score = performance_score
        if performance_score > 0.6:
            self.current_streak = min(10, self.current_streak + 1)
        else:
            self.current_streak = 0
        
        # Calculate reward components
        base_reward = performance_score
        streak_bonus = 0.1 * self.current_streak
        difficulty_bonus = 0.2 * (self.current_difficulty - 5) / 5 if performance_score > 0.6 and self.current_difficulty > 5 else 0
        time_bonus = 0.2 if time_efficiency > 0.8 else 0
        topic_coverage = 0.1 * np.mean(list(self.question_history.values())) + 0.1 * min(self.question_history.values())
        
        # Total reward
        reward = base_reward + streak_bonus + difficulty_bonus + time_bonus + topic_coverage
        
        # Store transition in agent's buffer
        done = self.current_step >= 9  # 10 questions per interview (0-9)
        self.agent.store_transition(reward, done)
        
        # Update step counter
        self.current_step += 1
    
    def train(self):
        """Train the agent on collected experience."""
        if self.current_step < 10:  # Need complete interview
            return None
        
        # Train PPO agent
        metrics = self.agent.train()
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
            policy_net=self.agent.policy,
            value_net=self.agent.value_net,
            metrics={
                'final_stats': stats,
                'mean_reward': np.mean([p['score'] for p in self.performances]) if self.performances else 0.0,
                'policy_loss': self.agent.metrics['policy_loss'][-1] if self.agent.metrics['policy_loss'] else 0.0,
                'value_loss': self.agent.metrics['value_loss'][-1] if self.agent.metrics['value_loss'] else 0.0,
                'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
            }
        )
        return version
    
    def load_model(self, version):
        """Load model from version."""
        if version:
            self.model_handler.load_model(
                policy_net=self.agent.policy,
                value_net=self.agent.value_net,
                version=version
            )
    
    def get_interview_stats(self):
        """Get current interview statistics."""
        if not self.performances:
            return {
                'average_score': 0.0,
                'time_efficiency': 1.0,
                'topic_performances': {t: 0.0 for t in self.topics},
                'current_streak': 0,
                'difficulty_level': 5.0
            }
        
        return {
            'average_score': np.mean([p['score'] for p in self.performances]),
            'time_efficiency': np.mean(self.time_efficiency_history),
            'topic_performances': self.question_history.copy(),
            'current_streak': self.current_streak,
            'difficulty_level': self.current_difficulty
        } 