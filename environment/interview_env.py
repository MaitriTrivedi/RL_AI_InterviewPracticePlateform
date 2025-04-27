import numpy as np
import json
import random

class InterviewEnvironment:
    def __init__(self, dataset_path='rl_agent/training_episodes.json'):
        """Initialize interview environment."""
        # Load dataset
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
            self.episodes = dataset['episodes']
        
        # Initialize state
        self.topics = ['ds', 'algo', 'oops', 'dbms', 'os', 'cn', 'system_design']
        self.current_episode = None
        self.current_step = 0
        self.max_steps = 10
    
    def reset(self):
        """Reset environment for new episode."""
        # Reset step counter
        self.current_step = 0
        
        # Select random episode
        self.current_episode = random.choice(self.episodes)
        
        # Get initial observation
        obs = self._get_observation()
        
        return obs, {}
    
    def _get_observation(self):
        """Get current state observation."""
        if self.current_step >= len(self.current_episode['questions']):
            return np.zeros(6, dtype=np.float32)
        
        question = self.current_episode['questions'][self.current_step]
        performances = self.current_episode['performances'][:self.current_step]
        
        # Calculate features
        progress = self.current_step / self.max_steps
        avg_score = np.mean([p['score'] for p in performances]) if performances else 0.5
        time_efficiency = np.mean([p['time_taken'] / q['time_allocated'] 
                                 for p, q in zip(performances, self.current_episode['questions'])]) if performances else 1.0
        topic_idx = self.topics.index(question['topic']) / len(self.topics)
        topic_performance = np.mean([p['score'] for p, q in zip(performances, self.current_episode['questions'])
                                   if q['topic'] == question['topic']]) if performances else 0.5
        current_difficulty = question['difficulty'] / 10.0
        
        return np.array([
            progress,
            avg_score,
            time_efficiency,
            topic_idx,
            topic_performance,
            current_difficulty
        ], dtype=np.float32)
    
    def step(self, difficulty):
        """Take a step in the environment with given difficulty."""
        if self.current_step >= len(self.current_episode['questions']):
            return self._get_observation(), 0.0, True, False, {}
        
        # Get current question and performance
        question = self.current_episode['questions'][self.current_step]
        performance = self.current_episode['performances'][self.current_step]
        
        # Adjust difficulty (clip to valid range)
        adjusted_difficulty = np.clip(difficulty, 1.0, 10.0)
        
        # Calculate performance based on difficulty adjustment
        base_performance = performance['score']
        difficulty_factor = 1.0 - (abs(adjusted_difficulty - question['difficulty']) / 10.0)
        actual_performance = base_performance * difficulty_factor
        
        # Calculate reward components
        performance_reward = actual_performance
        difficulty_penalty = -0.1 * abs(adjusted_difficulty - question['difficulty'])
        time_reward = 0.1 * (1.0 - abs(1.0 - performance['time_taken'] / question['time_allocated']))
        
        # Total reward
        reward = performance_reward + difficulty_penalty + time_reward
        
        # Update state
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        # Get next observation
        next_obs = self._get_observation()
        
        # Create info dict
        info = {
            'topic': question['topic'],
            'difficulty': adjusted_difficulty,
            'performance': actual_performance,
            'time_taken': performance['time_taken'],
            'time_allocated': question['time_allocated']
        }
        
        return next_obs, float(reward), terminated, truncated, info
    
    def get_current_topic(self):
        """Get the topic of the current question."""
        if self.current_step >= len(self.current_episode['questions']):
            return self.topics[0]
        return self.current_episode['questions'][self.current_step]['topic'] 