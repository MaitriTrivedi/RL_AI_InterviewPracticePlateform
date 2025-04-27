import numpy as np
import torch
from .model_handler import ModelHandler
from .ppo_agent import PPOAgent

class InterviewAgent:
    def __init__(self):
        """Initialize the interview agent."""
        # Available topics
        self.topics = ['algo', 'ds', 'os', 'cn', 'dbms', 'system_design']
        
        # Initialize PPO agent
        # State space: [difficulty, topic_idx, time_allocated, questions_completed]
        self.state_dim = 4
        self.agent = PPOAgent(state_dim=self.state_dim)
        
        # Load model if available
        self.model_handler = ModelHandler()
        try:
            metrics = self.model_handler.load_model(self.agent.policy)
            print("Loaded model with metrics:", metrics)
        except Exception as e:
            print(f"Starting with fresh model: {e}")
        
        # Initialize interview state
        self.reset_state()
    
    def reset_state(self):
        """Reset the interview state."""
        self.current_state = {
            'questions_completed': 0,
            'questions_remaining': 10,  # Default to 10 questions per interview
            'total_score': 0.0,
            'total_time_efficiency': 0.0,
            'topic_performances': {topic: [] for topic in self.topics}
        }
    
    def start_new_interview(self):
        """Start a new interview session."""
        self.reset_state()
    
    def _get_observation(self, question_info):
        """Convert question info to observation vector."""
        # Normalize values
        difficulty = question_info['difficulty'] / 10.0
        topic_idx = self.topics.index(question_info['topic']) / len(self.topics)
        time_allocated = min(question_info['time_allocated'] / 60.0, 1.0)  # Normalize to 1 hour max
        progress = self.current_state['questions_completed'] / 10.0  # Progress through interview
        
        return np.array([difficulty, topic_idx, time_allocated, progress], dtype=np.float32)
    
    def adjust_difficulty(self, question_info):
        """Get the adjusted difficulty for the next question."""
        # Get observation
        obs = self._get_observation(question_info)
        
        # Get action from policy
        action = self.agent.select_action(obs, deterministic=True)
        
        # Action is already clipped to [1, 10] in the policy
        return float(action)
    
    def update_performance(self, performance):
        """Update the agent's state with the latest performance."""
        # Update interview state
        self.current_state['questions_completed'] += 1
        self.current_state['questions_remaining'] -= 1
        self.current_state['total_score'] += performance['score']
        
        time_efficiency = performance['time_allocated'] / max(performance['time_taken'], 1)
        self.current_state['total_time_efficiency'] += time_efficiency
        
        # Calculate reward
        base_reward = performance['score']  # Base reward is the score (0 to 1)
        time_bonus = max(0, time_efficiency - 0.8) * 0.2  # Bonus for good time management
        reward = base_reward + time_bonus
        
        # Store transition in agent's buffer
        self.agent.store_transition(reward, self.current_state['questions_remaining'] == 0)
        
        # Update policy if we have enough samples
        if len(self.agent.states) >= self.agent.batch_size:
            metrics = self.agent.update()
            print("Training metrics:", metrics)
    
    def get_interview_stats(self):
        """Get current interview statistics."""
        questions_completed = self.current_state['questions_completed']
        if questions_completed == 0:
            return {
                'average_score': 0.0,
                'time_efficiency': 0.0,
                'questions_completed': 0,
                'questions_remaining': self.current_state['questions_remaining']
            }
        
        return {
            'average_score': self.current_state['total_score'] / questions_completed,
            'time_efficiency': self.current_state['total_time_efficiency'] / questions_completed,
            'questions_completed': questions_completed,
            'questions_remaining': self.current_state['questions_remaining']
        }
    
    def save_model(self):
        """Save the current model state."""
        metrics = self.agent.get_metrics()
        version = self.model_handler.save_model(self.agent.policy, metrics)
        return version 