import json
import os
import random
import numpy as np

class QuestionBank:
    def __init__(self, questions_file="questions.json"):
        """Initialize question bank."""
        self.questions_file = questions_file
        self.questions = self.load_questions()
        
        # Create index for quick access
        self.question_index = {}
        for question in self.questions:
            key = (question['topic'], question['subtopic'])
            if key not in self.question_index:
                self.question_index[key] = []
            self.question_index[key].append(question)
    
    def load_questions(self):
        """Load questions from JSON file."""
        if not os.path.exists(self.questions_file):
            # Create sample questions if file doesn't exist
            sample_questions = self.create_sample_questions()
            with open(self.questions_file, 'w') as f:
                json.dump(sample_questions, f, indent=2)
            return sample_questions
        
        with open(self.questions_file, 'r') as f:
            return json.load(f)
    
    def create_sample_questions(self):
        """Create sample questions for each topic and subtopic."""
        topics = {
            'ds': ["Arrays", "Strings", "Linked Lists", "Stacks and Queues", 
                  "Hashing", "Trees", "Heaps", "Tries", "Graphs"],
            'algo': ["Sorting", "Searching", "Two Pointer", "Recursion", 
                    "Backtracking", "Greedy", "Dynamic Programming"],
            'oops': ["Classes and Objects", "Inheritance", "Polymorphism", 
                    "Abstraction", "Interfaces", "Design Patterns"],
            'dbms': ["Basic SQL", "Joins", "Normalization", "Indexes", 
                    "Transactions", "Query Optimization"],
            'os': ["Process Management", "Memory Management", "CPU Scheduling", 
                  "Deadlocks", "File Systems", "Multithreading"],
            'cn': ["OSI Model", "TCP/UDP", "IP Addressing", "Routing", 
                  "HTTP/HTTPS", "Socket Programming"],
            'system_design': ["Scalability", "Load Balancing", "Caching", 
                            "Database Design", "API Design", "Microservices"]
        }
        
        questions = []
        for topic, subtopics in topics.items():
            for subtopic in subtopics:
                # Create questions of varying difficulty
                for diff in [3, 5, 7]:  # Easy, Medium, Hard
                    questions.append({
                        'topic': topic,
                        'subtopic': subtopic,
                        'difficulty': diff,
                        'text': f"Sample {subtopic} question (Difficulty: {diff}/10)",
                        'expected_time': 5 + diff // 2,  # Higher difficulty = more time
                        'hints': [
                            f"Hint 1 for {subtopic}",
                            f"Hint 2 for {subtopic}"
                        ],
                        'example': f"Example for {subtopic} question"
                    })
        
        return questions
    
    def get_question(self, topic, subtopic, difficulty):
        """Get a question matching the criteria."""
        # Get questions for topic and subtopic
        key = (topic, subtopic)
        available_questions = self.question_index.get(key, [])
        
        if not available_questions:
            # Fallback to a generic question if none available
            return {
                'topic': topic,
                'subtopic': subtopic,
                'difficulty': difficulty,
                'text': f"Generic question for {subtopic} in {topic}",
                'expected_time': 5,
                'hints': ["Think about the basic concepts"],
                'example': "Basic example"
            }
        
        # Find questions close to target difficulty
        difficulties = np.array([q['difficulty'] for q in available_questions])
        diff_scores = np.abs(difficulties - difficulty)
        best_matches = [q for q, score in zip(available_questions, diff_scores) 
                       if score <= 2]  # Within 2 points of target difficulty
        
        if not best_matches:
            best_matches = available_questions  # Fallback to all questions
        
        # Randomly select from best matches
        return random.choice(best_matches)
    
    def add_question(self, question):
        """Add a new question to the bank."""
        self.questions.append(question)
        
        # Update index
        key = (question['topic'], question['subtopic'])
        if key not in self.question_index:
            self.question_index[key] = []
        self.question_index[key].append(question)
        
        # Save to file
        with open(self.questions_file, 'w') as f:
            json.dump(self.questions, f, indent=2)
    
    def get_topics(self):
        """Get list of available topics."""
        return list(set(q['topic'] for q in self.questions))
    
    def get_subtopics(self, topic):
        """Get list of subtopics for a topic."""
        return list(set(q['subtopic'] for q in self.questions 
                       if q['topic'] == topic)) 