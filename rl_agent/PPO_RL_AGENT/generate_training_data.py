import numpy as np
import json
from datetime import datetime
import random
import os

class InterviewDataGenerator:
    def __init__(self):
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

    def generate_student_profile(self):
        """Generate a realistic student profile with varying expertise levels."""
        return {
            'topic_expertise': {
                topic: random.uniform(0.3, 0.9) for topic in self.topics
            },
            'learning_rate': random.uniform(0.1, 0.3),
            'stress_tolerance': random.uniform(0.5, 0.9),
            'communication_skill': random.uniform(0.6, 0.9)
        }

    def calculate_performance(self, difficulty, student_expertise, stress_level):
        """Calculate realistic performance based on multiple factors."""
        base_performance = np.clip(
            student_expertise - (difficulty/10) + random.uniform(-0.1, 0.1), 
            0.2, 
            1.0
        )
        
        stress_impact = stress_level * random.uniform(0.1, 0.2)
        final_performance = np.clip(base_performance - stress_impact, 0.0, 1.0)
        
        return final_performance

    def calculate_time_taken(self, difficulty, performance):
        """Calculate realistic time taken based on difficulty and performance."""
        base_time = difficulty * 2  # Base time in minutes
        performance_factor = 1 + (1 - performance)  # Lower performance = more time
        variance = random.uniform(0.8, 1.2)
        
        return base_time * performance_factor * variance

    def generate_question_data(self, student_profile, current_step, prev_performance=None):
        """Generate realistic question data with adaptive difficulty."""
        # Select topic based on performance history
        if prev_performance and prev_performance < 0.6:
            # Stay in same topic area if struggling
            topic = random.choice(self.topics[:3])
        else:
            topic = random.choice(self.topics)

        # Calculate adaptive difficulty
        base_difficulty = 5.0
        if prev_performance:
            if prev_performance > 0.8:
                base_difficulty += random.uniform(1, 2)
            elif prev_performance < 0.6:
                base_difficulty -= random.uniform(1, 2)

        difficulty = np.clip(base_difficulty + random.uniform(-1, 1), 1, 10)

        # Calculate stress/fatigue
        stress_level = min(0.3 + (current_step * 0.07), 0.8)
        
        # Generate performance
        performance = self.calculate_performance(
            difficulty,
            student_profile['topic_expertise'][topic],
            stress_level
        )

        # Calculate realistic time taken
        time_taken = self.calculate_time_taken(difficulty, performance)

        # Generate evaluation metrics
        evaluation_metrics = {
            'technical_accuracy': np.clip(performance + random.uniform(-0.1, 0.1), 0, 1),
            'communication_clarity': np.clip(student_profile['communication_skill'] + random.uniform(-0.1, 0.1), 0, 1),
            'problem_solving': np.clip(performance * 1.1, 0, 1),
            'completeness': np.clip(performance * 0.9, 0, 1)
        }

        # Ensure we use a valid subtopic from our list
        subtopic = random.choice(self.subtopics[topic])

        return {
            'topic': topic,
            'subtopic': subtopic,  # Using validated subtopic
            'difficulty': difficulty,
            'performance': performance,
            'time_taken': time_taken,
            'stress_level': stress_level,
            'evaluation_metrics': evaluation_metrics
        }

    def generate_interview_episode(self, student_id, episode_id):
        """Generate a complete interview episode."""
        student_profile = self.generate_student_profile()
        questions = []
        prev_performance = None

        for step in range(10):  # 10 questions per interview
            question_data = self.generate_question_data(
                student_profile,
                step,
                prev_performance
            )
            questions.append(question_data)
            prev_performance = question_data['performance']

        return {
            'student_id': student_id,
            'episode_id': episode_id,
            'student_profile': student_profile,
            'questions': questions,
            'overall_performance': np.mean([q['performance'] for q in questions]),
            'topic_coverage': len(set(q['topic'] for q in questions)) / len(self.topics)
        }

    def generate_dataset(self, num_students=100, episodes_per_student=5):
        """Generate full training dataset."""
        dataset = []
        
        for student_id in range(num_students):
            for episode_id in range(episodes_per_student):
                episode_data = self.generate_interview_episode(student_id, episode_id)
                dataset.append(episode_data)

        # Create data directory if it doesn't exist
        data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'data',
            'training_data'
        )
        os.makedirs(data_dir, exist_ok=True)

        # Save dataset
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(data_dir, f'training_data_{timestamp}.json')
        
        with open(filename, 'w') as f:
            json.dump(dataset, f, indent=2)

        return filename

if __name__ == '__main__':
    generator = InterviewDataGenerator()
    filename = generator.generate_dataset(num_students=100, episodes_per_student=5)
    print(f"Generated dataset saved to: {filename}") 