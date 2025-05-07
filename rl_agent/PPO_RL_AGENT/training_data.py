import numpy as np
import json
from datetime import datetime

class StudentSimulator:
    def __init__(self):
        """Initialize student simulator with realistic performance patterns."""
        # Student characteristics
        self.base_skills = {
            'ds': np.random.normal(5, 1),      # Data Structures
            'algo': np.random.normal(5, 1),    # Algorithms
            'oops': np.random.normal(5, 1),    # Object-Oriented Programming
            'dbms': np.random.normal(5, 1),    # Database Management
            'os': np.random.normal(5, 1),      # Operating Systems
            'cn': np.random.normal(5, 1),      # Computer Networks
            'system_design': np.random.normal(5, 1)  # System Design
        }
        
        # Learning rate for each topic (how quickly they improve)
        self.learning_rates = {
            topic: np.random.uniform(0.1, 0.3) 
            for topic in self.base_skills.keys()
        }
        
        # Fatigue factor (performance decreases with consecutive questions)
        self.fatigue_rate = np.random.uniform(0.05, 0.15)
        self.current_fatigue = 0
        
        # Streak effects (confidence boost/anxiety)
        self.streak_confidence_boost = np.random.uniform(0.05, 0.15)
        self.current_streak = 0
        
        # Topic relationships (improvement in one topic affects related topics)
        self.topic_relationships = {
            'ds': ['algo'],  # Data structures improvement helps in algorithms
            'algo': ['ds', 'system_design'],
            'oops': ['system_design'],
            'dbms': ['system_design'],
            'os': ['system_design', 'cn'],
            'cn': ['os', 'system_design'],
            'system_design': ['dbms', 'oops']
        }
        
        # Performance history
        self.performance_history = {topic: [] for topic in self.base_skills.keys()}
        
    def reset(self):
        """Reset student state for new interview."""
        self.current_fatigue = 0
        self.current_streak = 0
        
    def get_performance(self, topic, subtopic, difficulty):
        """Get simulated performance for a question."""
        # Base performance based on skill level
        base_skill = self.base_skills[topic]
        
        # Adjust for learning from previous questions
        learning_boost = len(self.performance_history[topic]) * self.learning_rates[topic]
        
        # Related topics boost
        related_boost = 0
        if topic in self.topic_relationships:
            for related_topic in self.topic_relationships[topic]:
                if self.performance_history[related_topic]:
                    related_boost += np.mean(self.performance_history[related_topic]) * 0.1
        
        # Calculate effective skill
        effective_skill = (
            base_skill +
            learning_boost +
            related_boost +
            (self.current_streak * self.streak_confidence_boost) -
            self.current_fatigue
        )
        
        # Performance depends on difference between skill and difficulty
        performance_factor = max(0.1, 1.0 - abs(difficulty - effective_skill) / 10.0)
        
        # Add realistic noise
        noise = np.random.normal(0, 0.1)
        performance = min(1.0, max(0.0, performance_factor + noise))
        
        # Update state
        self.performance_history[topic].append(performance)
        
        if performance > 0.6:
            self.current_streak += 1
            # Small skill increase on successful questions
            self.base_skills[topic] += 0.05
        else:
            self.current_streak = 0
        
        # Increase fatigue
        self.current_fatigue += self.fatigue_rate
        
        # Calculate time taken
        base_time = 5 + difficulty  # Base time increases with difficulty
        time_factor = 1.0
        if self.current_fatigue > 0.5:
            time_factor *= (1 + self.current_fatigue)  # Fatigue increases time
        if self.current_streak > 2:
            time_factor *= 0.9  # Confidence reduces time
        time_taken = base_time * time_factor * (1 + np.random.normal(0, 0.2))
        
        return performance, time_taken
    
    def get_skill_profile(self):
        """Get current skill levels for all topics."""
        return {
            topic: {
                'base_skill': self.base_skills[topic],
                'learning_rate': self.learning_rates[topic],
                'avg_performance': np.mean(self.performance_history[topic]) if self.performance_history[topic] else 0
            }
            for topic in self.base_skills.keys()
        }

def generate_training_episodes(num_episodes=100, num_students=10):
    """Generate training episodes with multiple simulated students."""
    training_data = []
    
    for student_id in range(num_students):
        student = StudentSimulator()
        student_episodes = []
        
        for episode in range(num_episodes // num_students):
            student.reset()
            episode_data = {
                'student_id': student_id,
                'episode_id': episode,
                'questions': [],
                'final_skills': None
            }
            
            # Run 10 questions per episode
            for _ in range(10):
                # Random topic selection (this will be replaced by agent's selection)
                topic = np.random.choice(list(student.base_skills.keys()))
                difficulty = np.random.uniform(3, 8)  # Random difficulty between 3-8
                
                # Get performance
                performance, time_taken = student.get_performance(topic, None, difficulty)
                
                question_data = {
                    'topic': topic,
                    'difficulty': difficulty,
                    'performance': performance,
                    'time_taken': time_taken,
                    'streak': student.current_streak,
                    'fatigue': student.current_fatigue
                }
                episode_data['questions'].append(question_data)
            
            # Record final skill profile
            episode_data['final_skills'] = student.get_skill_profile()
            student_episodes.append(episode_data)
        
        training_data.extend(student_episodes)
    
    return training_data

def save_training_data(data, filename='training_data.json'):
    """Save training data to file."""
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

def load_training_data(filename='training_data.json'):
    """Load training data from file."""
    with open(filename, 'r') as f:
        return json.load(f)

if __name__ == "__main__":
    # Generate training data
    print("Generating training data...")
    data = generate_training_episodes(num_episodes=100, num_students=10)
    
    # Save to file
    filename = f'training_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    save_training_data(data, filename)
    print(f"Training data saved to {filename}")
    
    # Print some statistics
    num_episodes = len(data)
    num_questions = sum(len(episode['questions']) for episode in data)
    avg_performance = np.mean([
        question['performance'] 
        for episode in data 
        for question in episode['questions']
    ])
    
    print(f"\nTraining Data Statistics:")
    print(f"Number of episodes: {num_episodes}")
    print(f"Total questions: {num_questions}")
    print(f"Average performance: {avg_performance:.3f}") 