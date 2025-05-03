import numpy as np
import torch
import json
from collections import deque
from datetime import datetime
from .ppo_agent import PPOAgent
from .model_handler import ModelHandler
from config import INTERVIEW_CONFIG
import random

class InterviewAgent:
    def __init__(self, state_dim=None, model_version=None):
        """Initialize interview agent with PPO."""
        # Use config values or defaults
        self.state_dim = state_dim or INTERVIEW_CONFIG['model']['state_dim']
        self.action_dim = INTERVIEW_CONFIG['model']['action_dim']
        
        # Initialize PPO agent
        self.agent = PPOAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=INTERVIEW_CONFIG['model']['hidden_dim'],
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
        
        # Load the specified model version or default from config
        self.model_version = model_version or INTERVIEW_CONFIG['model']['version']
        if self.model_version:
            self.load_model(self.model_version)
        
        # Initialize interview state
        self.topics = ['ds', 'algo', 'oops', 'dbms', 'os', 'cn', 'system_design']
        
        # Initialize subtopics dictionary
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
        
        self.reset_interview_state()
    
    def reset_interview_state(self):
        """Reset interview state for new interview."""
        self.current_step = 0                                                                                                               
        self.current_score = 0.0
        self.current_streak = 0
        self.time_efficiency = 1.0
        self.current_difficulty = 5  # Start with medium difficulty
        self.question_history = {topic: 0.0 for topic in self.topics}
        self.performances = []
        
        # Initialize topic and subtopic performance tracking
        self.topic_performances = {topic: [] for topic in self.topics}
        self.subtopic_performances = {
            topic: {subtopic: [] for subtopic in subtopics}
            for topic, subtopics in self.subtopics.items()
        }
        self.subtopic_history = {
            topic: {subtopic: 0.0 for subtopic in subtopics}
            for topic, subtopics in self.subtopics.items()
        }
        
        self.time_efficiency_history = []
        self.current_state = None
        self.current_action = None
        self.current_value = None
        self.current_log_prob = None
    
    def select_next_topic(self, exclude_topics=None):
        """Select next topic based on performance weights and exploration."""
        available_topics = [t for t in self.topics if not exclude_topics or t not in exclude_topics]
        if not available_topics:
            return self.topics[0]  # Fallback to first topic if all excluded
            
        # For first few questions, prefer fundamental topics
        if self.current_step < 3:
            fundamental_topics = ['ds', 'algo', 'oops']
            basic_topics = [t for t in fundamental_topics 
                          if not exclude_topics or t not in exclude_topics]
            if basic_topics:
                # Use performance weights even for basic topics
                weights = [1.0 / (1.0 + self.question_history[t]) for t in basic_topics]
                weights = np.array(weights) / np.sum(weights)
                return np.random.choice(basic_topics, p=weights)
            
        # Calculate performance-based weights
        performance_weights = [1.0 / (1.0 + self.question_history[t]) for t in available_topics]
        
        # Calculate exploration factor
        exploration_weights = [1.0 / (1.0 + len(self.topic_performances[t])) for t in available_topics]
        
        # Add topic complexity penalty for early questions
        complexity_weights = np.ones(len(available_topics))
        if self.current_step < 5:  # First 5 questions
            for i, topic in enumerate(available_topics):
                if topic in ['system_design', 'cn']:  # Complex topics
                    complexity_weights[i] = 0.3  # Reduce probability
        
        # Combine all weights
        combined_weights = (np.array(performance_weights) * 
                          np.array(exploration_weights) * 
                          complexity_weights)
        
        # Normalize weights
        weights = combined_weights / np.sum(combined_weights)
        
        # Select topic based on weights
        selected_topic = np.random.choice(available_topics, p=weights)
        
        return selected_topic
    
    def select_subtopic(self, topic, exclude_subtopics=None):
        """Select a subtopic based on performance weights and exploration."""
        available_subtopics = [s for s in self.subtopics[topic] 
                             if not exclude_subtopics or s not in exclude_subtopics]
        
        if not available_subtopics:
            return self.subtopics[topic][0]  # Fallback to first subtopic
        
        # For early questions, prefer basic subtopics
        if self.current_step < 3:
            # Define basic subtopics for each topic
            basic_subtopics = {
                'ds': ["Arrays", "Strings", "Linked Lists"],
                'algo': ["Sorting", "Searching", "Two Pointer"],
                'oops': ["Classes and Objects", "Encapsulation", "Inheritance"],
                'dbms': ["Basic SQL", "Joins", "Normalization"],
                'os': ["Process Management", "Memory Management", "CPU Scheduling"],
                'cn': ["OSI Model", "IP Addressing", "TCP/UDP"],
                'system_design': ["Scalability", "Load Balancing", "Caching"]
            }
            
            # Filter available basic subtopics
            basic_available = [s for s in basic_subtopics.get(topic, [])
                             if s in available_subtopics]
            if basic_available:
                return random.choice(basic_available)
        
        # Calculate weights based on subtopic performance and exploration
        performance_weights = [1.0 / (1.0 + self.subtopic_history[topic][s]) 
                             for s in available_subtopics]
        exploration_weights = [1.0 / (1.0 + len(self.subtopic_performances[topic][s])) 
                             for s in available_subtopics]
        
        # Add complexity weights for subtopics
        complexity_weights = np.ones(len(available_subtopics))
        if self.current_step < 5:  # First 5 questions
            for i, subtopic in enumerate(available_subtopics):
                # Define complex subtopics
                complex_subtopics = {
                    'ds': ["Segment Trees", "Tries"],
                    'algo': ["Dynamic Programming", "Graph Algorithms"],
                    'oops': ["SOLID Principles", "Design Patterns"],
                    'dbms': ["Query Optimization", "Concurrency"],
                    'os': ["Synchronization", "Virtual Memory"],
                    'cn': ["Congestion Control", "Socket Programming"],
                    'system_design': ["High Availability", "System Architecture"]
                }
                if subtopic in complex_subtopics.get(topic, []):
                    complexity_weights[i] = 0.3  # Reduce probability
        
        # Combine weights
        combined_weights = (np.array(performance_weights) * 
                          np.array(exploration_weights) * 
                          complexity_weights)
        weights = combined_weights / np.sum(combined_weights)
        
        # Select subtopic based on weights
        selected_subtopic = np.random.choice(available_subtopics, p=weights)
        
        return selected_subtopic
    
    def _get_state(self, topic):
        """Get current state for decision making."""
        # Convert topic to index safely
        try:
            topic_idx = float(self.topics.index(topic)) / float(len(self.topics) - 1)
        except (ValueError, IndexError):
            topic_idx = 0.0  # Default to first topic if not found
        
        state = np.array([
            topic_idx,                                    # Current topic (normalized)
            self.current_difficulty / 10.0,               # Current difficulty (normalized)
            self.current_score,                           # Average performance
            self.time_efficiency,                         # Time efficiency
            self.current_streak / 10.0,                   # Streak (normalized)
            *[float(val) for val in list(self.question_history.values())[-4:]]  # Last 4 topics' performance
        ], dtype=np.float32)
        
        # Ensure state has correct dimension
        if len(state) < self.state_dim:
            state = np.pad(state, (0, self.state_dim - len(state)), 'constant')
        elif len(state) > self.state_dim:
            state = state[:self.state_dim]
            
        return state
    
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
        diff_adjustment = float((action[0] * 4) - 2)  # Scale from [0,1] to [-2,2]
        
        # Determine max difficulty based on current step
        if self.current_step < 3:  # First 3 questions
            max_difficulty = 5.0
        elif self.current_step < 6:  # Questions 4-6
            max_difficulty = 7.0
        else:  # Later questions
            max_difficulty = 10.0
        
        # Update difficulty with bounds and minimum increase for high performance
        if self.current_score > 0.8:
            # More aggressive difficulty increase for high performance
            performance_factor = (self.current_score - 0.8) / 0.2  # Scale from 0 to 1 for scores 0.8 to 1.0
            difficulty_increase = 1.5 + (performance_factor * 1.5)  # Base increase 1.5 to 3.0
            
            if self.current_streak >= 2:  # Additional boost for consistent performance
                difficulty_increase += 1.0  # Increased streak bonus
                
            new_difficulty = min(
                self.current_difficulty + difficulty_increase,  # More significant increase
                self.current_difficulty * 1.5,  # More aggressive multiplier
                max_difficulty
            )
        elif self.current_score < 0.4:  # Poor performance
            # Reduce difficulty more significantly
            new_difficulty = max(
                1.0,
                self.current_difficulty - 1.0  # Larger reduction
            )
        else:
            # Normal difficulty adjustment with bounds
            new_difficulty = float(np.clip(
                self.current_difficulty + diff_adjustment,
                max(1.0, self.current_difficulty - 1.0),  # Prevent large drops
                min(max_difficulty, self.current_difficulty + 0.5)  # Limit increases
            ))
        
        self.current_difficulty = new_difficulty
        
        return {
            'difficulty': self.current_difficulty,
            'value_estimate': float(value),
            'log_prob': float(log_prob)
        }
    
    def update_performance(self, topic, subtopic, performance_score, time_taken):
        """Update agent with question performance."""
        # Adjust difficulty more conservatively in warm-up phase
        if self.current_step < 3:
            if performance_score < 0.6:
                self.current_difficulty = max(1.0, self.current_difficulty - 1.0)
            elif performance_score > 0.8:
                self.current_difficulty = min(5.0, self.current_difficulty + 0.5)

        # Subtopic mapping for legacy data
        subtopic_mapping = {
            # Data Structures
            "Arrays": "Arrays",
            "Linked Lists": "Linked Lists",
            "Trees": "Trees",
            "Graphs": "Graphs",
            "Heaps": "Heaps",
            
            # Algorithms
            "Sorting": "Sorting",
            "Searching": "Searching",
            "Dynamic Programming": "Dynamic Programming",
            "Greedy": "Greedy",
            "Backtracking": "Backtracking",
            
            # OOPS
            "Classes": "Classes and Objects",
            "Inheritance": "Inheritance",
            "Polymorphism": "Polymorphism",
            "Abstraction": "Abstraction",
            "Design Patterns": "Design Patterns",
            
            # DBMS
            "SQL": "Basic SQL",
            "Normalization": "Normalization",
            "Transactions": "Transactions",
            "Indexing": "Indexes",
            "Query Optimization": "Query Optimization",
            
            # Operating Systems
            "Process Management": "Process Management",
            "Memory Management": "Memory Management",
            "File Systems": "File Systems",
            "Scheduling": "CPU Scheduling",
            "Deadlocks": "Deadlocks",
            
            # Computer Networks
            "TCP/IP": "TCP/UDP",
            "Routing": "Routing",
            "Network Security": "Firewalls",
            "Protocols": "Application Protocols",
            "Socket Programming": "Socket Programming",
            
            # System Design
            "Scalability": "Scalability",
            "Load Balancing": "Load Balancing",
            "Caching": "Caching",
            "Microservices": "System Architecture",
            "Database Design": "Database Design"
        }

        # Map legacy subtopic to new format if needed
        mapped_subtopic = subtopic_mapping.get(subtopic, subtopic)
        
        # Store performance
        self.performances.append({
            'score': performance_score,
            'time_taken': time_taken,
            'topic': topic,
            'subtopic': mapped_subtopic
        })
        
        # Update topic-specific performance
        self.topic_performances[topic].append(performance_score)
        self.question_history[topic] = np.mean(self.topic_performances[topic])
        
        # Update subtopic-specific performance
        try:
            self.subtopic_performances[topic][mapped_subtopic].append(performance_score)
            self.subtopic_history[topic][mapped_subtopic] = np.mean(
                self.subtopic_performances[topic][mapped_subtopic]
            )
        except KeyError as e:
            print(f"Warning: Unknown subtopic mapping for {subtopic} in {topic}")
            # Use a default subtopic for the topic if mapping fails
            default_subtopic = list(self.subtopics[topic])[0]
            self.subtopic_performances[topic][default_subtopic].append(performance_score)
            self.subtopic_history[topic][default_subtopic] = np.mean(
                self.subtopic_performances[topic][default_subtopic]
            )
        
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
        
        # Calculate reward components with improved weights
        base_reward = performance_score
        streak_bonus = 0.2 * self.current_streak  # Increased from 0.1
        
        # Enhanced difficulty bonus
        difficulty_bonus = 0.4 * (self.current_difficulty - 5) / 5 if performance_score > 0.6 else -0.2
        
        # Improved time efficiency bonus
        time_bonus = 0.3 if time_efficiency > 0.8 else 0.1 if time_efficiency > 0.6 else 0
        
        # Enhanced topic coverage reward
        covered_topics = len([t for t in self.topics if self.question_history[t] > 0])
        topic_coverage = 0.2 * (covered_topics / len(self.topics))
        
        # Enhanced subtopic coverage
        covered_subtopics = sum(1 for t in self.topics for s in self.subtopics[t] 
                              if len(self.subtopic_performances[t][s]) > 0)
        total_subtopics = sum(len(subtopics) for subtopics in self.subtopics.values())
        subtopic_coverage = 0.3 * (covered_subtopics / total_subtopics)
        
        # Add exploration bonus
        exploration_bonus = 0.3 * (1 - self.question_history[topic])
        
        # Total reward with all components
        reward = (base_reward + streak_bonus + difficulty_bonus + 
                 time_bonus + topic_coverage + subtopic_coverage + exploration_bonus)
        
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
                'mean_reward': float(np.mean([p['score'] for p in self.performances]) if self.performances else 0.0),
                'policy_loss': float(self.agent.metrics['policy_loss'][-1] if self.agent.metrics['policy_loss'] else 0.0),
                'value_loss': float(self.agent.metrics['value_loss'][-1] if self.agent.metrics['value_loss'] else 0.0),
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
        
        # Convert NumPy types to native Python types
        return {
            'average_score': float(np.mean([p['score'] for p in self.performances])),
            'time_efficiency': float(np.mean(self.time_efficiency_history)),
            'topic_performances': {k: float(v) for k, v in self.question_history.items()},
            'current_streak': int(self.current_streak),
            'difficulty_level': float(self.current_difficulty)
        } 