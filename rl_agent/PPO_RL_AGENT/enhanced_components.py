import numpy as np
from collections import deque

class DifficultyManager:
    def __init__(self):
        self.history = deque(maxlen=10)  # Track last 10 difficulty-performance pairs
        self.recovery_mode = False
        self.confidence_threshold = 0.7
        self.recovery_steps = 3
        self.steps_remaining = 0
        
    def adjust_difficulty(self, current_diff, performance, consecutive_success):
        """Adjust difficulty based on performance with smoother transitions."""
        self.history.append((current_diff, performance))
        
        # Check if we need to enter recovery mode
        if performance < 0.2 and not self.recovery_mode:
            self.enter_recovery()
            return self.get_recovery_difficulty(current_diff)
            
        # Update recovery mode if active
        if self.recovery_mode:
            return self.handle_recovery_mode(performance, current_diff)
            
        # Normal difficulty adjustment
        return self.calculate_new_difficulty(current_diff, performance, consecutive_success)
    
    def enter_recovery(self):
        """Enter recovery mode after poor performance."""
        self.recovery_mode = True
        self.steps_remaining = self.recovery_steps
        
    def get_recovery_difficulty(self, original_difficulty):
        """Calculate recovery mode difficulty."""
        return max(1.0, original_difficulty * 0.6)  # Drop to 60% of original
        
    def handle_recovery_mode(self, performance, current_diff):
        """Handle difficulty adjustment during recovery mode."""
        if performance >= self.confidence_threshold:
            self.steps_remaining -= 1
            if self.steps_remaining <= 0:
                self.recovery_mode = False
                return min(current_diff + 0.5, current_diff * 1.2)
        else:
            self.steps_remaining = self.recovery_steps  # Reset recovery if still struggling
            
        return current_diff
        
    def calculate_new_difficulty(self, current_diff, performance, consecutive_success):
        """Calculate new difficulty for normal mode."""
        # Get recent performance trend
        recent_performances = [p for _, p in self.history]
        avg_recent_perf = np.mean(recent_performances) if recent_performances else performance
        
        # Calculate maximum safe jump based on history
        max_jump = min(0.5, 0.2 * consecutive_success)
        
        if performance >= 0.8 and avg_recent_perf >= 0.7:
            # Excellent performance with good history
            return min(current_diff + max_jump, 10.0)
        elif performance >= 0.6:
            # Good performance
            return min(current_diff + 0.3, current_diff * 1.1)
        elif performance <= 0.3:
            # Poor performance
            return max(current_diff - 0.5, 1.0)
        else:
            # Moderate performance - small adjustments
            return current_diff + (performance - 0.5) * 0.2

class TopicManager:
    def __init__(self, topics, subtopics):
        self.topics = topics
        self.subtopics = subtopics
        self.topic_history = {topic: [] for topic in topics}
        self.topic_relationships = {
            'ds': ['algo'],
            'algo': ['ds', 'system_design'],
            'oops': ['system_design', 'dbms'],
            'dbms': ['system_design', 'oops'],
            'os': ['system_design', 'cn'],
            'cn': ['os', 'system_design'],
            'system_design': ['oops', 'dbms', 'cn']
        }
        
    def select_next_topic(self, current_topic, performance):
        """Select next topic using enhanced selection strategy."""
        # Update topic history
        if current_topic:
            self.topic_history[current_topic].append(performance)
            
        # Calculate topic scores
        topic_scores = self._calculate_topic_scores()
        
        # Exploration vs exploitation
        if np.random.random() < self._get_exploration_rate():
            return self._explore_topics(current_topic)
        else:
            return self._exploit_topics(topic_scores)
    
    def _calculate_topic_scores(self):
        """Calculate scores for each topic based on history."""
        scores = {}
        for topic in self.topics:
            history = self.topic_history[topic]
            if not history:
                scores[topic] = 1.0  # High score for unexplored topics
            else:
                # Recent performance weighted more heavily
                weights = np.exp(np.linspace(-1, 0, len(history)))
                weighted_avg = np.average(history, weights=weights)
                scores[topic] = weighted_avg
        return scores
    
    def _get_exploration_rate(self):
        """Calculate exploration rate based on coverage."""
        covered_topics = len([t for t in self.topics if self.topic_history[t]])
        return 0.3 + (0.4 * (1 - covered_topics / len(self.topics)))
    
    def _explore_topics(self, current_topic):
        """Select a topic for exploration."""
        # Prioritize related topics that are unexplored
        if current_topic and current_topic in self.topic_relationships:
            related_topics = self.topic_relationships[current_topic]
            unexplored_related = [t for t in related_topics if not self.topic_history[t]]
            if unexplored_related:
                return np.random.choice(unexplored_related)
        
        # Fall back to any unexplored topic
        unexplored = [t for t in self.topics if not self.topic_history[t]]
        if unexplored:
            return np.random.choice(unexplored)
            
        # If all explored, pick random topic
        return np.random.choice(self.topics)
    
    def _exploit_topics(self, topic_scores):
        """Select a topic for exploitation based on scores."""
        # Convert scores to probabilities
        total_score = sum(topic_scores.values())
        probs = {t: s/total_score for t, s in topic_scores.items()}
        
        # Select topic based on probabilities
        return np.random.choice(
            list(probs.keys()),
            p=list(probs.values())
        )

class FatigueManager:
    def __init__(self):
        self.baseline_time = 300  # 5 minutes baseline
        self.fatigue_factor = 1.0
        self.time_history = deque(maxlen=5)
        
    def update_fatigue(self, time_taken, expected_time):
        """Update fatigue based on time taken vs expected."""
        self.time_history.append(time_taken / expected_time)
        
        # Calculate rolling average of time ratios
        avg_time_ratio = np.mean(self.time_history)
        
        # Update fatigue factor
        if avg_time_ratio > 1.5:  # Consistently taking 50% longer
            self.fatigue_factor = min(2.0, self.fatigue_factor * 1.2)
        elif avg_time_ratio < 0.8:  # Consistently faster
            self.fatigue_factor = max(1.0, self.fatigue_factor * 0.9)
    
    def get_adjusted_difficulty(self, base_difficulty):
        """Get difficulty adjusted for fatigue."""
        return base_difficulty / self.fatigue_factor 