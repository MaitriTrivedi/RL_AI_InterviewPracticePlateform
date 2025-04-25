"""
agent.py - Reinforcement Learning Agent for Interview Question Selection

This module implements the RL agent that learns to select appropriate
interview questions based on the candidate's resume and performance.
"""

import numpy as np
import random
from typing import Dict, List, Tuple, Any, Optional
import os
import pickle
import json

class QLearningAgent:
    """
    Q-Learning agent for selecting interview questions.
    
    This agent learns to:
    1. Select appropriate question difficulty based on candidate profile and performance
    2. Choose relevant question topics based on resume
    3. Adapt to candidate responses over time
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        exploration_rate: float = 1.0,
        exploration_decay: float = 0.995,
        min_exploration_rate: float = 0.1,
        bins_per_dimension: int = 10
    ):
        """
        Initialize the Q-Learning agent.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            learning_rate: Learning rate (alpha) for Q-value updates
            discount_factor: Discount factor (gamma) for future rewards
            exploration_rate: Initial exploration rate (epsilon)
            exploration_decay: Rate of decay for exploration
            min_exploration_rate: Minimum exploration rate
            bins_per_dimension: Number of bins for discretizing continuous state space
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        self.bins_per_dimension = bins_per_dimension
        
        # Define discrete action space
        self.num_difficulty_levels = 10  # 1-10 difficulty levels
        self.num_topic_preferences = 10  # 10 possible topic preferences
        
        # Initialize Q-table with small random values
        self._init_q_table()
        
    def _init_q_table(self):
        """Initialize the Q-table with small random values."""
        # Create bins for each state dimension
        self.state_bins = [self.bins_per_dimension] * self.state_dim
        
        # Initialize Q-table
        # For the action space, we use:
        # - num_difficulty_levels for difficulty selection
        # - num_topic_preferences for topic selection
        q_table_shape = tuple(self.state_bins + [self.num_difficulty_levels, self.num_topic_preferences])
        
        # Initialize with small random values for exploration
        self.q_table = np.random.uniform(low=0, high=0.1, size=q_table_shape)
        
        # Track visits for each state-action pair
        self.visit_counts = np.zeros(q_table_shape)
    
    def _discretize_state(self, state: np.ndarray) -> Tuple:
        """
        Convert continuous state to discrete state index for Q-table lookup.
        
        Args:
            state: Continuous state vector
            
        Returns:
            Tuple of indices for Q-table
        """
        # Clip state values to [0, 1] range
        clipped_state = np.clip(state, 0, 1)
        
        # Discretize each dimension
        discrete_state = []
        for i, val in enumerate(clipped_state):
            # Map [0, 1] to [0, bins_per_dimension-1]
            bin_idx = min(int(val * self.bins_per_dimension), self.bins_per_dimension - 1)
            discrete_state.append(bin_idx)
        
        return tuple(discrete_state)
    
    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state vector
            training: Whether we're in training mode (applies exploration)
            
        Returns:
            Action vector
        """
        # Discretize state for Q-table lookup
        discrete_state = self._discretize_state(state)
        
        # Epsilon-greedy action selection
        if training and random.random() < self.exploration_rate:
            # Explore: select random action
            difficulty_idx = random.randint(0, self.num_difficulty_levels - 1)
            topic_idx = random.randint(0, self.num_topic_preferences - 1)
        else:
            # Exploit: select best action
            # Find indices of highest Q-value
            q_values = self.q_table[discrete_state]
            max_indices = np.unravel_index(np.argmax(q_values), q_values.shape)
            difficulty_idx, topic_idx = max_indices
        
        # Convert discrete actions to continuous action vector
        # Map difficulty_idx [0, num_difficulty_levels-1] to [0, 1]
        difficulty = (difficulty_idx + 0.5) / self.num_difficulty_levels
        
        # Map topic_idx [0, num_topic_preferences-1] to [0, 1]
        topic = (topic_idx + 0.5) / self.num_topic_preferences
        
        return np.array([difficulty, topic])
    
    def update(
        self, 
        state: np.ndarray, 
        action: np.ndarray, 
        reward: float, 
        next_state: np.ndarray, 
        done: bool
    ):
        """
        Update Q-values based on observed transition.
        
        Args:
            state: Current state vector
            action: Action vector taken
            reward: Reward received
            next_state: Next state vector
            done: Whether episode is done
        """
        # Discretize states for Q-table lookup
        discrete_state = self._discretize_state(state)
        discrete_next_state = self._discretize_state(next_state)
        
        # Convert continuous action to discrete indices
        difficulty_idx = min(int(action[0] * self.num_difficulty_levels), self.num_difficulty_levels - 1)
        topic_idx = min(int(action[1] * self.num_topic_preferences), self.num_topic_preferences - 1)
        
        # Current Q-value
        current_q = self.q_table[discrete_state + (difficulty_idx, topic_idx)]
        
        # Calculate max Q-value for next state
        max_next_q = np.max(self.q_table[discrete_next_state]) if not done else 0
        
        # Q-learning update rule
        target_q = reward + self.discount_factor * max_next_q
        
        # Update visit count
        self.visit_counts[discrete_state + (difficulty_idx, topic_idx)] += 1
        
        # Use visit count to adjust learning rate (optional)
        adjusted_lr = self.learning_rate / (1 + 0.1 * self.visit_counts[discrete_state + (difficulty_idx, topic_idx)])
        
        # Update Q-value
        self.q_table[discrete_state + (difficulty_idx, topic_idx)] = (
            (1 - adjusted_lr) * current_q + adjusted_lr * target_q
        )
        
        # Decay exploration rate
        if self.exploration_rate > self.min_exploration_rate:
            self.exploration_rate *= self.exploration_decay
    
    def save(self, file_path: str):
        """
        Save the Q-table and agent parameters to a file.
        
        Args:
            file_path: Path to save the agent
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save agent state
        with open(file_path, 'wb') as f:
            pickle.dump({
                'q_table': self.q_table,
                'visit_counts': self.visit_counts,
                'exploration_rate': self.exploration_rate,
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'learning_rate': self.learning_rate,
                'discount_factor': self.discount_factor,
                'exploration_decay': self.exploration_decay,
                'min_exploration_rate': self.min_exploration_rate,
                'bins_per_dimension': self.bins_per_dimension
            }, f)
    
    def load(self, file_path: str):
        """
        Load the Q-table and agent parameters from a file.
        
        Args:
            file_path: Path to load the agent from
        """
        if not os.path.exists(file_path):
            print(f"Warning: No agent file found at {file_path}")
            return False
        
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                
                # Load agent parameters
                self.q_table = data['q_table']
                self.visit_counts = data['visit_counts']
                self.exploration_rate = data['exploration_rate']
                self.state_dim = data['state_dim']
                self.action_dim = data['action_dim']
                self.learning_rate = data['learning_rate']
                self.discount_factor = data['discount_factor']
                self.exploration_decay = data['exploration_decay']
                self.min_exploration_rate = data['min_exploration_rate']
                self.bins_per_dimension = data['bins_per_dimension']
                
                return True
        except Exception as e:
            print(f"Error loading agent: {e}")
            return False


class DQNAgent:
    """
    Placeholder for a Deep Q-Network agent.
    
    This would be a more sophisticated agent using neural networks
    for Q-function approximation, useful for larger state/action spaces.
    """
    
    def __init__(self):
        """
        Initialize DQN agent.
        
        For future implementation - using deep learning libraries
        like TensorFlow or PyTorch to create a neural network that
        approximates the Q-function.
        """
        print("DQN Agent placeholder - not implemented yet")
        
        # Would define neural network architecture, replay buffer, etc. 