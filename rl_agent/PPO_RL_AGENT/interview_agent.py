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
            
        # Topic difficulty levels
        self.topic_difficulty = {
            'ds': 1,      # Base level
            'algo': 2,    # More complex
            'oops': 1,    # Conceptual, base level
            'dbms': 2,    # More complex
            'os': 2,      # More complex
            'cn': 2,      # More complex
            'system_design': 3  # Most complex
        }
        
        # Available topics and subtopics
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
        
        # Subtopic name mapping for handling variations
        self.subtopic_mapping = {
            # Data Structures
            'Linked Lists (Singly, Doubly)': 'Linked Lists',
            'Stacks and Queues': 'Stacks and Queues',
            'Hashing (HashMaps, HashSets)': 'Hashing',
            'Trees (Binary Tree, BST, Traversals)': 'Trees',
            'Heaps (Min/Max Heap, Priority Queue)': 'Heaps',
            'Tries (Prefix Trees)': 'Tries',
            'Graphs (Adjacency List/Matrix, BFS, DFS)': 'Graphs',
            'Segment Trees / Binary Indexed Trees': 'Segment Trees',
            
            # Algorithms
            'Sorting (Bubble, Selection, Insertion)': 'Sorting',
            'Searching (Linear, Binary Search)': 'Searching',
            'Two Pointer Techniques': 'Two Pointer',
            'Two Pointers': 'Two Pointer',
            'Greedy Algorithms': 'Greedy',
            'Graph Algorithms (Dijkstra, Floyd-Warshall, Topological Sort, Union-Find)': 'Graph Algorithms',
            
            # OOPS
            'Classes': 'Classes and Objects',
            'Interfaces and Abstract Classes': 'Interfaces',
            'Design Patterns (Singleton, Factory, Observer)': 'Design Patterns',
            'UML & Class Diagrams': 'UML',
            'Real-world System Modeling': 'System Modeling',
            
            # DBMS
            'Basic SQL (SELECT, INSERT, UPDATE, DELETE)': 'Basic SQL',
            'SQL': 'Basic SQL',
            'Joins (INNER, LEFT, RIGHT, FULL)': 'Joins',
            'Constraints & Normalization (1NF, 2NF, 3NF)': 'Normalization',
            'Indexes & Views': 'Indexes',
            'Indexing': 'Indexes',
            'Transactions (ACID Properties)': 'Transactions',
            'Stored Procedures & Triggers': 'Stored Procedures',
            'Concurrency & Locking': 'Concurrency',
            'NoSQL vs RDBMS': 'NoSQL',
            'CAP Theorem & Distributed DB Concepts': 'CAP Theorem',
            
            # Operating Systems
            'Process vs Thread': 'Process Management',
            'Memory Management (Paging, Segmentation)': 'Memory Management',
            'CPU Scheduling Algorithms (FCFS, SJF, RR)': 'CPU Scheduling',
            'Deadlocks (Conditions, Prevention)': 'Deadlocks',
            'Virtual Memory & Thrashing': 'Virtual Memory',
            'File Systems & Inodes': 'File Systems',
            'Mutex vs Semaphore': 'Synchronization',
            'Context Switching & Scheduling': 'Context Switching',
            
            # Computer Networks
            'OSI vs TCP/IP Models': 'OSI Model',
            'TCP/IP': 'TCP/UDP',
            'DNS, DHCP, ARP': 'DNS/DHCP',
            'HTTP/HTTPS & REST APIs': 'HTTP/HTTPS',
            'Routing & Switching Basics': 'Routing',
            'Networking': 'Socket Programming',
            'Security': 'Firewalls',
            'Application Layer Protocols': 'Application Protocols',
            'Protocols': 'Application Protocols',
            'Network Protocols': 'Application Protocols',
            'Communication Protocols': 'Application Protocols',
            
            # System Design
            'Basics of Scalability (Vertical vs Horizontal)': 'Scalability',
            'Caching (Redis, CDN)': 'Caching',
            'Database Sharding & Replication': 'Database Design',
            'Databases': 'Database Design',
            'Designing RESTful APIs': 'API Design',
            'Rate Limiting & Throttling': 'Rate Limiting',
            'High Availability & Fault Tolerance': 'High Availability',
            'End-to-End Design of Systems': 'System Architecture',
            'Microservices': 'System Architecture',
            'System Architecture Design': 'System Architecture'
        }
        
        # Initialize enhanced components
        self.difficulty_manager = DifficultyManager()
        self.topic_manager = TopicManager(self.topics, self.subtopics)
        self.fatigue_manager = FatigueManager()
        
        # Reset interview state
        self.reset_interview_state()
    
    def reset_interview_state(self):
        """Reset interview state with enhanced tracking."""
        self.current_step = 0                                                                                                               
        self.current_score = 0.0
        self.current_streak = 0
        self.time_efficiency = 1.0
        self.current_difficulty = 5.0  # Start with medium difficulty
        self.current_topic = None
        self.last_topic = None
        self.topic_transition_count = 0
        self.consecutive_topic_changes = 0
        
        # Initialize performance tracking
        self.question_history = {topic: 0.0 for topic in self.topics}
        self.performances = []
        self.topic_performances = {topic: [] for topic in self.topics}
        self.subtopic_performances = {
            topic: {subtopic: [] for subtopic in subtopics}
            for topic, subtopics in self.subtopics.items()
        }
        
        # Enhanced time tracking
        self.time_efficiency_history = []
        self.topic_time_spent = {topic: 0 for topic in self.topics}
        self.expected_time_per_topic = 300  # 5 minutes per topic
        self.warmup_phase = True
        self.warmup_questions = 0
        
        # Reset enhanced components
        self.difficulty_manager = DifficultyManager()
        self.topic_manager = TopicManager(self.topics, self.subtopics)
        self.fatigue_manager = FatigueManager()
        
        # Reset PPO state
        self.current_state = None
        self.current_action = None
        self.current_value = None
        self.current_log_prob = None
    
    def _get_state(self, topic):
        """Get enhanced state representation."""
        try:
            topic_idx = float(self.topics.index(topic)) / float(len(self.topics) - 1)
        except (ValueError, IndexError):
            topic_idx = 0.0
        
        # Enhanced state features
        state = np.array([
            topic_idx,                                    # Current topic (normalized)
            self.current_difficulty / 10.0,               # Current difficulty (normalized)
            self.current_score,                           # Current performance
            self.time_efficiency,                         # Time efficiency
            self.current_streak / 10.0,                   # Performance streak
            self.fatigue_manager.fatigue_factor / 2.0,    # Fatigue factor (normalized)
            *[float(val) for val in list(self.question_history.values())[-3:]]  # Last 3 topics
        ], dtype=np.float32)
        
        # Ensure correct dimension
        if len(state) < self.state_dim:
            state = np.pad(state, (0, self.state_dim - len(state)), 'constant')
        elif len(state) > self.state_dim:
            state = state[:self.state_dim]
            
        return state
    
    def get_next_question(self, topic):
        """Get next question with enhanced topic management and safety constraints."""
        try:
            # Track topic transitions
            if self.current_topic != topic:
                self.consecutive_topic_changes += 1
                if self.consecutive_topic_changes > 2:
                    # Force staying on current topic if too many changes
                    topic = self.current_topic or topic
                    self.consecutive_topic_changes = 0
            else:
                self.consecutive_topic_changes = 0
            
            # Warmup phase handling with topic consideration
            if self.warmup_phase:
                if self.warmup_questions < 3:
                    # Start with easier questions based on topic difficulty
                    base_difficulty = 2.0 + self.topic_difficulty[topic]
                    self.current_difficulty = min(4.0, base_difficulty)
                    self.warmup_questions += 1
                else:
                    self.warmup_phase = False
            
            # Get state and select action
            state = self._get_state(topic)
            action_result = self.agent.select_action(state)
            
            # Handle different return types from select_action
            if isinstance(action_result, (tuple, list)):
                action = float(action_result[0])  # First element is always the action
                value = float(action_result[1]) if len(action_result) > 1 else 0.0
                log_prob = float(action_result[2]) if len(action_result) > 2 else 0.0
            elif isinstance(action_result, (np.ndarray, np.floating, float, int)):
                action = float(action_result)
                value = 0.0
                log_prob = 0.0
            else:
                raise ValueError(f"Unexpected action type: {type(action_result)}")
            
            # Apply safety constraints based on performance history
            if hasattr(self, 'current_score'):
                # Get recent performance metrics
                recent_scores = self.performances[-3:] if self.performances else []
                avg_recent_score = np.mean(recent_scores) if recent_scores else 0.5
                
                # Count consecutive low scores
                consecutive_low_scores = 0
                for score in reversed(self.performances):
                    if score < 0.3:
                        consecutive_low_scores += 1
                    else:
                        break
                
                # Calculate maximum allowed difficulty change
                if consecutive_low_scores >= 2:
                    # Force difficulty reduction on multiple low scores
                    max_difficulty = max(1.0, self.current_difficulty * 0.6)
                    action = min(action, max_difficulty)
                else:
                    # Normal difficulty progression
                    max_increase = min(2.0, avg_recent_score * 3)
                    if action > self.current_difficulty:
                        action = min(action, self.current_difficulty + max_increase)
            
            # Consider topic difficulty in final action
            topic_factor = self.topic_difficulty[topic] / 3.0  # Normalize to 0-1
            max_topic_difficulty = 7.0 + (topic_factor * 3.0)  # Max difficulty based on topic
            action = min(action, max_topic_difficulty)
            
            # Ensure action is within valid range
            action = np.clip(action, 1.0, 10.0)
            
            # Adjust difficulty based on time spent on topic
            time_factor = self.topic_time_spent[topic] / self.expected_time_per_topic
            if time_factor > 1.2:  # Spent too much time on topic
                action = max(1.0, action * 0.9)  # Reduce difficulty
            elif time_factor < 0.5:  # Not enough time spent
                action = min(10.0, action * 1.1)  # Increase difficulty
            
            # Store current state
            self.current_state = state
            self.current_action = action
            self.current_value = value
            self.current_log_prob = log_prob
            self.last_topic = self.current_topic
            self.current_topic = topic
            
            # Select appropriate subtopic
            subtopic = self.select_subtopic(topic)
            
            return action, subtopic
            
        except Exception as e:
            logging.error(f"Error in get_next_question: {e}")
            return 5.0, self.subtopics[topic][0]  # Safe fallback to medium difficulty
    
    def _normalize_subtopic(self, subtopic):
        """Normalize subtopic name to match the standard names in self.subtopics."""
        try:
            # First try direct mapping
            if subtopic in self.subtopic_mapping:
                return self.subtopic_mapping[subtopic]
            
            # If not in mapping, check if it exists in any topic's subtopics
            for topic_subtopics in self.subtopics.values():
                if subtopic in topic_subtopics:
                    return subtopic
            
            # If still not found, try to find a close match
            for standard_subtopic in self.subtopic_mapping.values():
                if subtopic.lower() in standard_subtopic.lower():
                    return standard_subtopic
            
            # If no match found, return a default subtopic for the current topic
            if self.current_topic and self.current_topic in self.subtopics:
                return self.subtopics[self.current_topic][0]
            
            # Final fallback
            return list(self.subtopics['ds'])[0]
            
        except Exception as e:
            logging.error(f"Error normalizing subtopic {subtopic}: {e}")
            return list(self.subtopics['ds'])[0]  # Safe fallback

    def select_subtopic(self, topic):
        """Select a subtopic for the given topic based on performance history and current state."""
        if topic not in self.subtopics:
            return self.subtopics['ds'][0]  # Fallback to first DS subtopic
            
        available_subtopics = self.subtopics[topic]
        
        # Get performance history for each subtopic
        subtopic_scores = {}
        for subtopic in available_subtopics:
            performances = self.subtopic_performances[topic][subtopic]
            if not performances:
                # No history - moderate score to balance exploration
                subtopic_scores[subtopic] = 0.5
            else:
                # Use recent performance with some randomness
                avg_perf = np.mean(performances[-3:]) if len(performances) >= 3 else np.mean(performances)
                subtopic_scores[subtopic] = avg_perf
        
        # Check recent performance
        recent_scores = self.performances[-3:] if self.performances else []
        avg_recent_score = np.mean(recent_scores) if recent_scores else 0.5
        
        # Adjust exploration rate based on performance
        if avg_recent_score < 0.3:  # Poor performance
            # Stay with familiar subtopics
            explored_subtopics = [s for s, scores in self.subtopic_performances[topic].items() if scores]
            if explored_subtopics:
                # Select from previously explored subtopics with decent performance
                good_subtopics = [s for s in explored_subtopics 
                                if np.mean(self.subtopic_performances[topic][s]) > 0.4]
                if good_subtopics:
                    return np.random.choice(good_subtopics)
                return np.random.choice(explored_subtopics)
        elif avg_recent_score < 0.6:  # Moderate performance
            # Reduce exploration probability
            if np.random.random() < 0.1:  # 10% chance of exploration
                unexplored = [s for s, scores in subtopic_scores.items() if scores == 0.5]
                if unexplored:
                    return np.random.choice(unexplored)
        else:  # Good performance
            # Normal exploration rate
            if np.random.random() < 0.2:  # 20% chance of exploration
                unexplored = [s for s, scores in subtopic_scores.items() if scores == 0.5]
                if unexplored:
                    return np.random.choice(unexplored)
        
        # Convert scores to selection probabilities with performance weighting
        total_score = sum(subtopic_scores.values())
        if total_score == 0:
            # If no scores, use uniform distribution
            return np.random.choice(available_subtopics)
            
        probs = [score/total_score for score in subtopic_scores.values()]
        
        # Select subtopic based on probabilities
        return np.random.choice(available_subtopics, p=probs)

    def update_performance(self, topic, subtopic, performance_score, time_taken):
        """Update performance metrics with enhanced time management."""
        try:
            # Normalize subtopic
            normalized_subtopic = self._normalize_subtopic(subtopic)
            
            # Ensure topic exists
            if topic not in self.topics:
                topic = 'ds'  # Fallback to data structures
            
            # Update time tracking
            self.topic_time_spent[topic] += time_taken
            self.time_efficiency = min(300 / max(time_taken, 1), 2.0)  # Cap at 2x efficiency
            self.time_efficiency_history.append(self.time_efficiency)
            
            # Update performance metrics
            self.current_score = performance_score
            if performance_score >= 0.7:  # Good performance threshold
                self.current_streak += 1
            else:
                self.current_streak = 0
            
            # Update topic and subtopic performances
            self.performances.append(performance_score)
            
            # Ensure topic exists in performances
            if topic not in self.topic_performances:
                self.topic_performances[topic] = []
            self.topic_performances[topic].append(performance_score)
            
            # Ensure topic and subtopic exist in subtopic_performances
            if topic not in self.subtopic_performances:
                self.subtopic_performances[topic] = {s: [] for s in self.subtopics[topic]}
            if normalized_subtopic not in self.subtopic_performances[topic]:
                self.subtopic_performances[topic][normalized_subtopic] = []
            self.subtopic_performances[topic][normalized_subtopic].append(performance_score)
            
            # Calculate time-weighted performance
            time_penalty = max(0, (time_taken - 300) / 300)  # Penalty for taking too long
            adjusted_score = performance_score * (1.0 - time_penalty)
            
            # Update question history with time-weighted scores
            self.question_history[topic] = (
                self.question_history[topic] * 0.7 +  # Historical weight
                adjusted_score * 0.3  # Current performance weight
            )
            
            # Calculate reward with time consideration
            reward = self._calculate_reward(topic, adjusted_score, self.time_efficiency)
            
            # Update agent
            self.agent.store_transition(reward, False)
            
            # Train if enough samples
            if len(self.agent.states) >= 32:  # Minimum batch size
                metrics = self.agent.train()  # Train on all available data
                if metrics and isinstance(metrics, dict):
                    self.current_step += 1  # Increment step counter after successful training
                    
                    # Log training metrics
                    logging.info(f"Training step {self.current_step} - "
                               f"Actor Loss: {metrics.get('actor_loss', 0.0):.3f}, "
                               f"Value Loss: {metrics.get('value_loss', 0.0):.3f}, "
                               f"Entropy Loss: {metrics.get('entropy_loss', 0.0):.3f}, "
                               f"Mean Reward: {metrics.get('mean_reward', 0.0):.3f}")
                    
                    return metrics  # Return metrics for monitoring
            
            return None  # Return None if no training occurred
            
        except Exception as e:
            logging.error(f"Error in update_performance: {e}")
            return None  # Return None on error
    
    def _calculate_reward(self, topic, performance_score, time_efficiency):
        """Calculate enhanced reward with multiple components and safety constraints."""
        # Base performance reward with difficulty consideration
        difficulty_factor = self.current_difficulty / 10.0
        topic_difficulty_factor = self.topic_difficulty[topic] / 3.0  # Normalize to 0-1
        
        # Penalize aggressive difficulty increases when performance is poor
        difficulty_change = self.current_difficulty - self.previous_difficulty if hasattr(self, 'previous_difficulty') else 0
        difficulty_penalty = max(0, difficulty_change/5.0) if performance_score < 0.3 else 0
        
        # Base reward considering difficulty appropriateness
        base_reward = performance_score * (1 - abs(difficulty_factor - performance_score))
        
        # Streak consideration
        streak_bonus = 0.2 * min(self.current_streak, 3)  # Cap streak bonus
        
        # Topic appropriateness reward
        topic_appropriateness = 1.0 - abs(topic_difficulty_factor - difficulty_factor)
        topic_reward = 0.3 * topic_appropriateness
        
        # Time efficiency bonus (reduced impact)
        time_bonus = 0.1 if time_efficiency > 0.8 else 0.05 if time_efficiency > 0.6 else 0
        
        # Topic coverage bonus (reduced impact)
        covered_topics = len([t for t in self.topics if self.question_history[t] > 0])
        coverage_bonus = 0.1 * (covered_topics / len(self.topics))
        
        # Exploration bonus for trying new topics (with difficulty consideration)
        exploration_bonus = 0.2 if len(self.topic_performances[topic]) <= 1 and difficulty_factor <= topic_difficulty_factor else 0
        
        # Combine all components
        total_reward = (base_reward 
                       + streak_bonus 
                       + topic_reward
                       + time_bonus 
                       + coverage_bonus 
                       + exploration_bonus 
                       - difficulty_penalty)  # Subtract the difficulty penalty
        
        # Store current difficulty for next calculation
        self.previous_difficulty = self.current_difficulty
        
        return np.clip(total_reward, -1.0, 2.0)  # Clip reward to reasonable range
    
    def get_interview_stats(self):
        """Get enhanced interview statistics."""
        # Calculate topic coverage
        covered_topics = len([t for t in self.topics if self.topic_performances[t]])
        topic_coverage = covered_topics / len(self.topics)
        
        # Calculate average scores per topic
        topic_averages = {}
        for topic in self.topics:
            scores = self.topic_performances[topic]
            topic_averages[topic] = float(np.mean(scores)) if scores else 0.0
        
        # Calculate overall statistics
        return {
            'average_score': float(np.mean(self.performances)) if self.performances else 0.0,
            'time_efficiency': float(np.mean(self.time_efficiency_history)) if self.time_efficiency_history else 1.0,
            'topic_performances': topic_averages,
            'current_streak': int(self.current_streak),
            'difficulty_level': float(self.current_difficulty),
            'fatigue_level': float(self.fatigue_manager.fatigue_factor),
            'topic_coverage': float(topic_coverage),
            'total_questions': len(self.performances)
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
            # Try direct path first
            if os.path.exists(version):
                model_path = version
            else:
                # Try in versions directory
                versions_dir = os.path.join(self.models_dir, 'versions')
                model_path = os.path.join(versions_dir, f"{version}.npy")
                if not os.path.exists(model_path):
                    # Try in main models directory
                    model_path = os.path.join(self.models_dir, f"{version}.npy")
            
            if not os.path.exists(model_path):
                raise ValueError(f"Model version {version} not found at {model_path}")
            
            self.agent.load_model(model_path)
            logging.info(f"Model loaded from: {model_path}")
            
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise 