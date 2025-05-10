import numpy as np
import torch
import json
import os
from collections import deque
from datetime import datetime
from .ppo_agent import PPOAgent
from .model_handler import ModelHandler
from .enhanced_components import DifficultyManager, TopicManager, FatigueManager
import sys
sys.path.append('/home/maitri/Study/RLproject/RL_AI_InterviewPracticePlateform')
from config import INTERVIEW_CONFIG
import random
import logging

class InterviewAgent:
    def __init__(self, state_dim=None, model_version=None, models_dir=None):
        """Initialize interview agent with enhanced components."""
        # Base configuration
        self.state_dim = state_dim or 9
        self.models_dir = models_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'data',
            'models'
        )
        
        # Initialize PPO agent and model handler
        self.agent = PPOAgent(state_dim=self.state_dim, action_dim=1)
        self.model_handler = ModelHandler(models_dir=self.models_dir)
        
        # Load specified model version if provided
        if model_version:
            self.load_model(model_version)
            
        # Topic difficulty levels and progression thresholds
        self.topic_difficulty = {
            'ds': {'base': 1, 'max': 8, 'progression_rate': 0.5},
            'algo': {'base': 2, 'max': 9, 'progression_rate': 0.6},
            'oops': {'base': 1, 'max': 8, 'progression_rate': 0.5},
            'dbms': {'base': 2, 'max': 9, 'progression_rate': 0.6},
            'os': {'base': 2, 'max': 9, 'progression_rate': 0.6},
            'cn': {'base': 2, 'max': 9, 'progression_rate': 0.6},
            'system_design': {'base': 3, 'max': 10, 'progression_rate': 0.7}
        }
        
        # Performance thresholds
        self.performance_thresholds = {
            'excellent': 0.8,
            'good': 0.7,
            'average': 0.5,
            'poor': 0.3
        }
        
        # Difficulty adjustment rates
        self.difficulty_adjustment = {
            'increase_rate': 0.5,    # Slower increase
            'decrease_rate': 1.0,    # Faster decrease
            'max_increase': 1.5,     # Maximum single increase
            'max_decrease': 2.0      # Maximum single decrease
        }
        
        # Topic cooldown and recovery
        self.topic_cooldown = {}     # Track topics that need cooldown
        self.cooldown_threshold = 2  # Number of poor performances before cooldown
        self.recovery_threshold = 3   # Number of good performances needed for recovery
        
        # Initialize other components as before...
        self.topics = ['ds', 'algo', 'oops', 'dbms', 'os', 'cn', 'system_design']
        self.subtopics = {
            'ds': ["Arrays", "Strings", "Linked Lists", "Stacks and Queues", 
                  "Hashing", "Trees", "Heaps", "Tries", "Graphs", "Segment Trees"],
            'algo': ["Sorting", "Searching", "Two Pointer", "Recursion", 
                    "Backtracking", "Greedy", "Divide and Conquer", 
                    "Sliding Window", "Dynamic Programming", "Graph Algorithms"],
            'oops': ["Classes and Objects", "Encapsulation", "Inheritance", 
                    "Polymorphism", "Abstraction", "Interfaces", "SOLID Principles", 
                    "Design Patterns", "UML", "System Modeling"],
            'dbms': ["Basic SQL", "Joins", "Normalization", "Indexes", 
                    "Transactions", "Stored Procedures", "Concurrency", 
                    "Query Optimization", "NoSQL", "CAP Theorem"],
            'os': ["Process Management", "Memory Management", "CPU Scheduling", 
                  "Deadlocks", "IPC", "Virtual Memory", "File Systems", 
                  "Multithreading", "Synchronization", "Context Switching"],
            'cn': ["OSI Model", "IP Addressing", "TCP/UDP", "DNS/DHCP", 
                  "HTTP/HTTPS", "Routing", "Firewalls", "Congestion Control", 
                  "Socket Programming", "Application Protocols"],
            'system_design': ["Scalability", "Load Balancing", "Caching", 
                            "Database Design", "CAP Theorem", "Message Queues", 
                            "API Design", "Rate Limiting", "High Availability", 
                            "System Architecture"]
        }
        
        # Enhanced tracking
        self.topic_difficulty_ceilings = {topic: self.topic_difficulty[topic]['max'] for topic in self.topics}
        self.topic_current_difficulty = {topic: self.topic_difficulty[topic]['base'] for topic in self.topics}
        self.consecutive_performances = {topic: [] for topic in self.topics}
        
        # Reset interview state
        self.reset_interview_state()
    
    def reset_interview_state(self):
        """Reset interview state with enhanced tracking."""
        self.current_step = 0
        self.current_score = 0.0
        self.current_streak = 0
        self.time_efficiency = 1.0
        self.current_difficulty = 2.0  # Start easier to assess user level
        self.current_topic = None
        self.last_topic = None
        self.topic_transition_count = 0
        self.consecutive_topic_changes = 0
        
        # Enhanced performance tracking
        self.question_history = {topic: 0.0 for topic in self.topics}
        self.performances = []
        self.topic_performances = {topic: [] for topic in self.topics}
        self.topic_difficulty_history = {topic: [] for topic in self.topics}
        self.topic_time_history = {topic: [] for topic in self.topics}
        
        # Reset topic-specific difficulties
        self.topic_current_difficulty = {topic: self.topic_difficulty[topic]['base'] for topic in self.topics}
        
        # Reset cooldowns
        self.topic_cooldown = {}
        
        # Initialize other tracking as before...
        self.time_efficiency_history = []
        self.topic_time_spent = {topic: 0 for topic in self.topics}
        self.expected_time_per_topic = 300  # 5 minutes per topic
        self.warmup_phase = True
        self.warmup_questions = 0
        
        # Reset PPO state
        self.current_state = None
        self.current_action = None
        self.current_value = None
        self.current_log_prob = None
    
    def _adjust_difficulty_for_topic(self, topic, performance_score, time_taken, expected_time):
        """Adjust difficulty based on performance and topic-specific factors."""
        current_diff = self.topic_current_difficulty[topic]
        base_diff = self.topic_difficulty[topic]['base']
        max_diff = self.topic_difficulty_ceilings[topic]
        
        # Time efficiency factor (0 to 1)
        time_factor = min(1.0, expected_time / max(time_taken, 1))
        
        # Calculate performance-based adjustment
        if performance_score >= self.performance_thresholds['excellent']:
            if time_factor >= 0.8:  # Only increase if time efficiency is good
                # Excellent performance - steady increase
                adjustment = self.difficulty_adjustment['increase_rate']
            else:
                # Good performance but slow - maintain current level
                adjustment = 0
        elif performance_score <= self.performance_thresholds['poor']:
            # Poor performance - guaranteed decrease
            adjustment = -self.difficulty_adjustment['decrease_rate']
        elif performance_score >= self.performance_thresholds['good']:
            # Good performance - small increase
            if time_factor >= 0.7:
                adjustment = self.difficulty_adjustment['increase_rate'] * 0.5
            else:
                adjustment = 0
        else:
            # Below average performance - small decrease
            adjustment = -self.difficulty_adjustment['decrease_rate'] * 0.3
        
        # Apply time penalty for very slow responses
        if time_factor < 0.5:  # Taking more than twice the expected time
            time_penalty = (0.5 - time_factor) * self.difficulty_adjustment['decrease_rate']
            adjustment -= time_penalty
        
        # Ensure adjustment stays within bounds
        if adjustment > 0:
            adjustment = min(adjustment, self.difficulty_adjustment['max_increase'])
        else:
            adjustment = max(adjustment, -self.difficulty_adjustment['max_decrease'])
        
        # Calculate new difficulty
        new_difficulty = current_diff + adjustment
        
        # Ensure difficulty stays within topic bounds
        new_difficulty = max(base_diff, min(max_diff, new_difficulty))
        
        # Ensure smooth transition (prevent large jumps)
        max_change = self.difficulty_adjustment['max_increase']
        new_difficulty = max(
            current_diff - max_change,
            min(current_diff + max_change, new_difficulty)
        )
        
        # Log significant changes for debugging
        if abs(new_difficulty - current_diff) > 0.5:
            logging.info(f"Significant difficulty change for {topic}: {current_diff:.2f} -> {new_difficulty:.2f}")
            logging.info(f"Performance: {performance_score:.2f}, Time factor: {time_factor:.2f}")
        
        # Update topic difficulty
        self.topic_current_difficulty[topic] = new_difficulty
        self.topic_difficulty_history[topic].append(new_difficulty)
        
        return new_difficulty

    def _update_topic_cooldown(self, topic, performance_score):
        """Update topic cooldown status based on performance."""
        # Keep track of consecutive performances (up to last 5)
        self.consecutive_performances[topic].append(performance_score)
        if len(self.consecutive_performances[topic]) > 5:
            self.consecutive_performances[topic].pop(0)
        
        recent_performances = self.consecutive_performances[topic]
        
        # Check for cooldown condition - stricter conditions
        if len(recent_performances) >= 2:
            recent_poor = all(score <= self.performance_thresholds['poor'] for score in recent_performances[-2:])
            avg_score = sum(recent_performances[-2:]) / 2
            
            if recent_poor or avg_score < self.performance_thresholds['poor']:
                self.topic_cooldown[topic] = {
                    'count': 0,
                    'required_recovery': self.recovery_threshold,
                    'original_difficulty': self.topic_current_difficulty[topic]
                }
                # Significantly reduce difficulty
                self.topic_current_difficulty[topic] = max(
                    self.topic_difficulty[topic]['base'],
                    self.topic_current_difficulty[topic] * 0.7  # 30% reduction
                )
                logging.info(f"Topic {topic} entered cooldown. Difficulty reduced to {self.topic_current_difficulty[topic]:.2f}")
        
        # Check for recovery condition
        if topic in self.topic_cooldown:
            if performance_score >= self.performance_thresholds['good']:
                self.topic_cooldown[topic]['count'] += 1
                if self.topic_cooldown[topic]['count'] >= self.topic_cooldown[topic]['required_recovery']:
                    # Gradually restore difficulty
                    original_diff = self.topic_cooldown[topic]['original_difficulty']
                    current_diff = self.topic_current_difficulty[topic]
                    target_diff = min(original_diff, self.topic_difficulty[topic]['max'])
                    self.topic_current_difficulty[topic] = current_diff + (target_diff - current_diff) * 0.3
                    
                    del self.topic_cooldown[topic]
                    logging.info(f"Topic {topic} recovered from cooldown. New difficulty: {self.topic_current_difficulty[topic]:.2f}")

    def _get_state(self, topic):
        """Get enhanced state representation for better policy learning."""
        try:
            # Normalize topic index
            topic_idx = float(self.topics.index(topic)) / float(len(self.topics) - 1)
            
            # Get recent performance metrics
            recent_scores = self.performances[-3:] if self.performances else [0.5]
            avg_recent_score = np.mean(recent_scores)
            
            # Get topic-specific metrics
            topic_scores = self.topic_performances.get(topic, [0.5])
            topic_avg_score = np.mean(topic_scores) if topic_scores else 0.5
            topic_progress = 0.0
            if len(topic_scores) >= 2:
                early_scores = topic_scores[:len(topic_scores)//2]
                recent_scores = topic_scores[len(topic_scores)//2:]
                topic_progress = np.mean(recent_scores) - np.mean(early_scores)
            
            # Get difficulty metrics
            current_diff = self.topic_current_difficulty[topic]
            normalized_diff = (current_diff - self.topic_difficulty[topic]['base']) / (self.topic_difficulty[topic]['max'] - self.topic_difficulty[topic]['base'])
            
            # Get time efficiency metrics
            topic_times = self.topic_time_history.get(topic, [300])
            avg_time = np.mean(topic_times) if topic_times else 300
            time_efficiency = min(1.0, self.expected_time_per_topic / max(avg_time, 1))
            
            # Construct state vector
            state = np.array([
                topic_idx,                    # Topic index (normalized)
                normalized_diff,              # Current difficulty (normalized)
                avg_recent_score,             # Recent overall performance
                topic_avg_score,              # Topic-specific performance
                topic_progress,               # Topic learning progress
                time_efficiency,              # Time efficiency
                float(self.current_streak) / 5.0,  # Performance streak (normalized)
                len(topic_scores) / 10.0,     # Topic exposure (normalized)
                float(topic in self.topic_cooldown)  # Cooldown status
            ], dtype=np.float32)
            
            return state
            
        except Exception as e:
            logging.error(f"Error in _get_state: {e}")
            return np.zeros(self.agent.state_dim, dtype=np.float32)  # Safe fallback

    def get_next_question(self, topic):
        """Get next question with enhanced policy-based decision making."""
        try:
            # Check if topic is in cooldown
            if topic in self.topic_cooldown:
                available_topics = [t for t in self.topics if t not in self.topic_cooldown]
                if available_topics:
                    topic = self._select_alternative_topic(available_topics)
            
            # Get enhanced state representation
            state = self._get_state(topic)
            
            # Get action from policy
            action, value, log_prob = self.agent.select_action(state)
            
            # Store PPO state
            self.current_state = state
            self.current_action = action
            self.current_value = value
            self.current_log_prob = log_prob
            
            # Get topic-specific difficulty bounds
            base_diff = self.topic_difficulty[topic]['base']
            max_diff = self.topic_difficulty_ceilings[topic]
            
            # Scale action to difficulty range
            difficulty = float(action)  # Action is already in [1, 10] range
            difficulty = max(base_diff, min(max_diff, difficulty))
            
            # Apply warmup phase adjustments
            if self.warmup_phase:
                if self.warmup_questions < 3:
                    difficulty = min(3.5, self.topic_difficulty[topic]['base'] + 1.0)
                    self.warmup_questions += 1
                else:
                    self.warmup_phase = False
            
            # Select appropriate subtopic
            subtopic = self.select_subtopic(topic)
            
            # Update tracking
            self.current_topic = topic
            self.last_topic = topic
            self.topic_current_difficulty[topic] = difficulty
            
            return difficulty, subtopic
            
        except Exception as e:
            logging.error(f"Error in get_next_question: {e}")
            return 3.0, self.subtopics[topic][0]  # Safe fallback

    def _select_alternative_topic(self, available_topics):
        """Select alternative topic based on performance history."""
        topic_scores = {}
        for topic in available_topics:
            scores = self.topic_performances.get(topic, [])
            if scores:
                avg_score = sum(scores) / len(scores)
                topic_scores[topic] = avg_score
            else:
                topic_scores[topic] = 0.5  # Default score for unexplored topics
        
        # Select topic with best potential for learning
        return max(topic_scores.items(), key=lambda x: x[1])[0]

    def update_performance(self, topic, subtopic, performance_score, time_taken):
        """Update agent's performance metrics and train if possible."""
        try:
            # Calculate reward
            reward = self._calculate_reward(
                topic, subtopic, performance_score, time_taken
            )
            
            logging.info(f"Performance update - Topic: {topic}, Score: {performance_score:.2f}, Reward: {reward:.2f}")
            
            training_metrics = None
            
            # Update PPO agent
            if hasattr(self, 'current_state'):
                # Store transition in PPO memory
                self.agent.store_transition(reward, False)  # done=False as interview continues
                
                # Log buffer status
                buffer_size = len(self.agent.replay_buffer['states']) if hasattr(self.agent, 'replay_buffer') else 0
                batch_size = self.agent.batch_size if hasattr(self.agent, 'batch_size') else 0
                logging.info(f"Replay buffer status - Current size: {buffer_size}, Batch size required: {batch_size}")
                
                # Train if enough steps
                if buffer_size >= batch_size:
                    logging.info("Starting PPO training update")
                    metrics = self.agent.train()
                    if metrics:
                        training_metrics = {
                            'policy_loss': float(metrics.get('policy_loss', 0.0)),
                            'value_loss': float(metrics.get('value_loss', 0.0)),
                            'entropy_loss': float(metrics.get('entropy_loss', 0.0))
                        }
                        logging.info(f"Training metrics updated: {training_metrics}")
                    else:
                        logging.warning("Training completed but no metrics returned")
                else:
                    logging.info("Skipping training - insufficient samples")
            
            # Update difficulty and cooldown
            self._adjust_difficulty_for_topic(topic, performance_score, time_taken, self.expected_time_per_topic)
            self._update_topic_cooldown(topic, performance_score)
            
            # Update streak
            if performance_score >= self.performance_thresholds['good']:
                self.current_streak += 1
            else:
                self.current_streak = 0
            
            # Update time efficiency
            time_efficiency = self.expected_time_per_topic / max(time_taken, 1)
            self.time_efficiency = time_efficiency
            self.time_efficiency_history.append(time_efficiency)
            
            # Update topic time tracking
            self.topic_time_spent[topic] += time_taken
            
            # Update question history
            self.question_history[topic] += 1
            
            # Return training metrics if available
            return training_metrics
            
        except Exception as e:
            logging.error(f"Error in update_performance: {e}")
            return None

    def get_interview_stats(self):
        """Get enhanced interview statistics."""
        try:
            # Calculate topic coverage
            covered_topics = len([t for t in self.topics if self.topic_performances[t]])
            topic_coverage = covered_topics / len(self.topics)
            
            # Calculate average scores per topic
            topic_averages = {}
            topic_difficulties = {}
            for topic in self.topics:
                scores = self.topic_performances[topic]
                difficulties = self.topic_difficulty_history[topic]
                topic_averages[topic] = float(np.mean(scores)) if scores else 0.0
                topic_difficulties[topic] = float(np.mean(difficulties)) if difficulties else self.topic_difficulty[topic]['base']
            
            # Calculate learning progress
            learning_progress = {}
            for topic in self.topics:
                scores = self.topic_performances[topic]
                if len(scores) >= 2:
                    early_avg = np.mean(scores[:len(scores)//2])
                    recent_avg = np.mean(scores[len(scores)//2:])
                    learning_progress[topic] = float(recent_avg - early_avg)
                else:
                    learning_progress[topic] = 0.0
            
            return {
                'average_score': float(np.mean(self.performances)) if self.performances else 0.0,
                'time_efficiency': float(np.mean(self.time_efficiency_history)) if self.time_efficiency_history else 1.0,
                'topic_performances': topic_averages,
                'topic_difficulties': topic_difficulties,
                'learning_progress': learning_progress,
                'current_streak': int(self.current_streak),
                'difficulty_level': float(self.current_difficulty),
                'topic_coverage': float(topic_coverage),
                'total_questions': len(self.performances),
                'cooldown_topics': list(self.topic_cooldown.keys())
            }
            
        except Exception as e:
            logging.error(f"Error in get_interview_stats: {str(e)}")
            return {
                'average_score': 0.0,
                'time_efficiency': 1.0,
                'topic_performances': {},
                'current_streak': 0,
                'difficulty_level': self.current_difficulty,
                'topic_coverage': 0.0,
                'total_questions': 0
            }
    
    def train(self):
        """Train the agent on collected experience."""
        if self.current_step < 10:  # Need complete interview
            return None
        
        # Train PPO agent
        metrics = self.agent.train()
        return metrics
    
    def save_checkpoint(self, checkpoint_path):
        """Save checkpoint of the model."""
        try:
            # Save agent checkpoint
            self.agent.save_checkpoint(checkpoint_path)
            
            # Save additional interview agent state if needed
            agent_state = {
                'current_step': self.current_step,
                'current_score': self.current_score,
                'current_streak': self.current_streak,
                'time_efficiency': self.time_efficiency,
                'current_difficulty': self.current_difficulty,
                'question_history': self.question_history,
                'topic_performances': self.topic_performances,
                'topic_time_spent': self.topic_time_spent
            }
            
            # Save agent state in same directory
            state_path = os.path.join(os.path.dirname(checkpoint_path), 'agent_state.npy')
            np.save(state_path, agent_state)
            
        except Exception as e:
            logging.error(f"Error saving checkpoint: {e}")
            raise  # Re-raise to handle in caller
    
    def save_model(self):
        """Save the current model."""
        try:
            # Create models directory if it doesn't exist
            os.makedirs(self.models_dir, exist_ok=True)
            
            # Get current metrics
            metrics = self.agent.get_metrics()
            
            # Generate version name with timestamp and reward
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            mean_reward = metrics.get('mean_reward', 0.0)
            version = f"model_v1_{timestamp}_reward_{mean_reward:.3f}"
            
            # Save model file
            model_path = os.path.join(self.models_dir, f"{version}.npy")
            
            # Save through model handler
            self.model_handler.save_model(
                policy_net=self.agent.policy,
                value_net=self.agent.value_net,
                metrics=metrics,
                path=model_path
            )
            
            logging.info(f"Model saved at: {model_path}")
            return version
            
        except Exception as e:
            logging.error(f"Error saving model: {e}")
            raise
    
    def load_model(self, version):
        """Load a specific model version."""
        try:
            self.model_handler.load_model(self.agent.policy, self.agent.value_net, version)
            logging.info(f"Successfully loaded model version: {version}")
        except Exception as e:
            logging.error(f"Error loading model version {version}: {e}")
            raise
        
    def select_subtopic(self, topic):
        """Select a subtopic based on current difficulty and performance history."""
        available_subtopics = self.subtopics[topic]
        current_difficulty = self.topic_current_difficulty[topic]
        
        # Map difficulty levels to subtopic indices
        # Lower difficulty -> earlier subtopics, higher difficulty -> later subtopics
        difficulty_index = int((current_difficulty / self.topic_difficulty[topic]['max']) * len(available_subtopics))
        difficulty_index = max(0, min(difficulty_index, len(available_subtopics) - 1))
        
        # Create a window of subtopics around the difficulty index
        window_size = 3  # Number of subtopics before and after the current difficulty
        start_idx = max(0, difficulty_index - window_size)
        end_idx = min(len(available_subtopics), difficulty_index + window_size + 1)
        candidate_subtopics = available_subtopics[start_idx:end_idx]
        
        # Add some randomness to prevent predictability
        if random.random() < 0.2:  # 20% chance to pick a random subtopic
            return random.choice(available_subtopics)
        
        return random.choice(candidate_subtopics)  # Pick from the difficulty-appropriate window 