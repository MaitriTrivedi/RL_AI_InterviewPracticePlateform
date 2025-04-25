import json
import random
from typing import List, Dict, Any

class QuestionGenerator:
    def __init__(self):
        self.topics = {
            "data_structures": {
                "subtopics": ["arrays", "linked_lists", "trees", "graphs", "hash_tables", "stacks", "queues"],
                "complexities": ["O(1)", "O(n)", "O(log n)", "O(n log n)", "O(n²)"]
            },
            "algorithms": {
                "subtopics": ["sorting", "searching", "dynamic_programming", "greedy", "backtracking"],
                "techniques": ["two_pointer", "sliding_window", "divide_and_conquer", "recursion"]
            },
            "system_design": {
                "components": ["load_balancer", "cache", "database", "message_queue", "api_gateway"],
                "concepts": ["scalability", "reliability", "availability", "consistency"]
            },
            "python": {
                "concepts": ["generators", "decorators", "context_managers", "metaclasses", "async_programming"],
                "features": ["list_comprehension", "lambda_functions", "iterators", "collections"]
            }
        }

    def generate_ds_question(self, id_num: int) -> Dict[str, Any]:
        topic = "data_structures"
        subtopic = random.choice(self.topics[topic]["subtopics"])
        complexity = random.choice(self.topics[topic]["complexities"])
        
        templates = [
            f"Implement a {subtopic.replace('_', ' ')} with {complexity} time complexity for operations.",
            f"Design a {subtopic.replace('_', ' ')} class with methods for insertion, deletion, and search.",
            f"Compare the performance implications of using {subtopic.replace('_', ' ')} vs arrays.",
        ]
        
        return {
            "id": f"ds_{str(id_num).zfill(3)}",
            "topic": topic,
            "difficulty": random.randint(1, 5),
            "question": random.choice(templates),
            "tags": [subtopic, complexity.lower(), "implementation"],
            "expected_time_minutes": random.randint(10, 30)
        }

    def generate_algo_question(self, id_num: int) -> Dict[str, Any]:
        topic = "algorithms"
        technique = random.choice(self.topics[topic]["techniques"])
        subtopic = random.choice(self.topics[topic]["subtopics"])
        
        templates = [
            f"Design an algorithm using {technique.replace('_', ' ')} approach to solve...",
            f"Optimize the {subtopic.replace('_', ' ')} algorithm to improve time complexity.",
            f"Implement a {subtopic.replace('_', ' ')} solution using {technique.replace('_', ' ')}.",
        ]
        
        return {
            "id": f"algo_{str(id_num).zfill(3)}",
            "topic": topic,
            "difficulty": random.randint(2, 8),
            "question": random.choice(templates),
            "tags": [subtopic, technique, "optimization"],
            "expected_time_minutes": random.randint(15, 45)
        }

    def generate_system_design_question(self, id_num: int) -> Dict[str, Any]:
        topic = "system_design"
        component = random.choice(self.topics[topic]["components"])
        concept = random.choice(self.topics[topic]["concepts"])
        
        templates = [
            f"Design a {concept} {component.replace('_', ' ')} system that can handle millions of requests.",
            f"How would you implement a distributed {component.replace('_', ' ')} with {concept}?",
            f"Architect a {concept} solution using {component.replace('_', ' ')} as the primary component.",
        ]
        
        return {
            "id": f"sys_{str(id_num).zfill(3)}",
            "topic": topic,
            "difficulty": random.randint(5, 10),
            "question": random.choice(templates),
            "tags": [component, concept, "architecture"],
            "expected_time_minutes": random.randint(30, 60)
        }

    def generate_python_question(self, id_num: int) -> Dict[str, Any]:
        topic = "python"
        concept = random.choice(self.topics[topic]["concepts"])
        feature = random.choice(self.topics[topic]["features"])
        
        templates = [
            f"Explain and implement {concept.replace('_', ' ')} in Python.",
            f"How would you use {feature.replace('_', ' ')} to optimize this code?",
            f"Write a program demonstrating {concept.replace('_', ' ')} with {feature.replace('_', ' ')}.",
        ]
        
        return {
            "id": f"py_{str(id_num).zfill(3)}",
            "topic": topic,
            "difficulty": random.randint(1, 7),
            "question": random.choice(templates),
            "tags": [concept, feature, "implementation"],
            "expected_time_minutes": random.randint(10, 40)
        }

    def generate_dataset(self, num_questions: int = 50) -> List[Dict[str, Any]]:
        questions = []
        generators = [
            self.generate_ds_question,
            self.generate_algo_question,
            self.generate_system_design_question,
            self.generate_python_question
        ]
        
        for i in range(num_questions):
            generator = random.choice(generators)
            question = generator(i)
            
            # Add historical performance data
            question["historical_performance"] = {
                "attempts": random.randint(100, 3000),
                "avg_score": round(random.uniform(0.5, 0.95), 2),
                "avg_time_minutes": round(random.uniform(
                    0.8 * question["expected_time_minutes"],
                    1.2 * question["expected_time_minutes"]
                ), 1)
            }
            
            questions.append(question)
        
        return questions

def main():
    generator = QuestionGenerator()
    questions = generator.generate_dataset()
    
    # Save to JSON file
    output_file = "interview_questions.json"
    with open(output_file, 'w') as f:
        json.dump(questions, f, indent=4)
    
    print(f"Generated {len(questions)} questions and saved to {output_file}")

if __name__ == "__main__":
    main() 