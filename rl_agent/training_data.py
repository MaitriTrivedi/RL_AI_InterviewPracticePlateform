import json
import numpy as np
from typing import Dict, List
from sentence_transformers import SentenceTransformer
import torch

class TrainingDataHandler:
    def __init__(self):
        # Load pre-trained model for text embeddings
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def generate_question_embeddings(self, question: str) -> np.ndarray:
        """Generate embeddings for question text"""
        return self.embedding_model.encode(question)
    
    def generate_answer_embeddings(self, answer: str) -> np.ndarray:
        """Generate embeddings for answer text"""
        return self.embedding_model.encode(answer)
    
    def calculate_answer_similarity(self, user_answer: str, expected_answer: str) -> float:
        """Calculate similarity between user's answer and expected answer"""
        user_embedding = self.generate_answer_embeddings(user_answer)
        expected_embedding = self.generate_answer_embeddings(expected_answer)
        
        # Compute cosine similarity
        similarity = np.dot(user_embedding, expected_embedding) / \
                    (np.linalg.norm(user_embedding) * np.linalg.norm(expected_embedding))
        return float(similarity)
    
    def simulate_user_answer(self, question: Dict, user_skill_level: float) -> Dict:
        """Simulate user's answer based on their skill level and question difficulty"""
        # Calculate base success probability
        diff = user_skill_level - question['difficulty']
        success_prob = 1 / (1 + np.exp(-diff))  # Sigmoid function
        
        # Simulate answer quality
        answer_quality = np.random.normal(success_prob, 0.1)
        answer_quality = np.clip(answer_quality, 0, 1)
        
        # Simulate time taken (in minutes)
        base_time = question['expected_time_minutes']
        time_taken = base_time * np.random.normal(1.0, 0.2)  # ±20% variation
        
        # Generate simulated answer text (for training purposes)
        if answer_quality > 0.7:
            answer_text = question['expected_answer']
        elif answer_quality > 0.4:
            # Simulate partial answer by taking part of expected answer
            words = question['expected_answer'].split()
            partial_length = int(len(words) * answer_quality)
            answer_text = ' '.join(words[:partial_length])
        else:
            answer_text = "Incomplete or incorrect answer"
        
        return {
            'answer_text': answer_text,
            'time_taken': time_taken,
            'quality_score': answer_quality
        }
    
    def evaluate_answer(self, question: Dict, user_answer: Dict) -> Dict:
        """Evaluate user's answer and return metrics"""
        # Calculate similarity with expected answer
        similarity = self.calculate_answer_similarity(
            user_answer['answer_text'], 
            question['expected_answer']
        )
        
        # Calculate time efficiency
        time_efficiency = question['expected_time_minutes'] / user_answer['time_taken']
        time_efficiency = np.clip(time_efficiency, 0, 1.5)  # Cap at 150% efficiency
        
        # Calculate overall score
        score = 0.7 * similarity + 0.3 * time_efficiency
        
        return {
            'similarity_score': similarity,
            'time_efficiency': time_efficiency,
            'overall_score': score
        }

def create_training_environment():
    """Create training environment with simulated user interactions"""
    handler = TrainingDataHandler()
    
    # Load questions from JSON
    with open('rl_agent/interview_questions.json', 'r') as f:
        questions = json.load(f)
    
    # Pre-compute embeddings for all questions and answers
    for question in questions:
        question['question_embedding'] = handler.generate_question_embeddings(
            question['question']
        ).tolist()
        question['answer_embedding'] = handler.generate_answer_embeddings(
            question['expected_answer']
        ).tolist()
    
    # Save enhanced dataset
    with open('rl_agent/enhanced_questions.json', 'w') as f:
        json.dump(questions, f, indent=4)
    
    return handler

if __name__ == "__main__":
    # Create training environment
    handler = create_training_environment()
    
    # Example of simulated interaction
    with open('rl_agent/interview_questions.json', 'r') as f:
        questions = json.load(f)
    
    # Simulate answers for different skill levels
    skill_levels = [3.0, 5.0, 8.0]  # Beginner, Intermediate, Expert
    
    for skill in skill_levels:
        print(f"\nSimulating answers for skill level: {skill}")
        question = questions[0]  # Take first question as example
        
        # Simulate user answer
        simulated_answer = handler.simulate_user_answer(question, skill)
        
        # Evaluate answer
        evaluation = handler.evaluate_answer(question, simulated_answer)
        
        print(f"Answer quality: {simulated_answer['quality_score']:.2f}")
        print(f"Time taken: {simulated_answer['time_taken']:.2f} minutes")
        print(f"Similarity score: {evaluation['similarity_score']:.2f}")
        print(f"Time efficiency: {evaluation['time_efficiency']:.2f}")
        print(f"Overall score: {evaluation['overall_score']:.2f}") 