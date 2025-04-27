from interview_dataset import InterviewQuestionBank, InterviewSimulator
import json
import random
import numpy as np

def generate_realistic_performance(difficulty: int, skill_level: float) -> float:
    """Generate realistic performance score based on difficulty and skill level"""
    # Base probability of success
    base_prob = 1 / (1 + np.exp(-(skill_level - difficulty)))  # Sigmoid function
    
    # Add some noise
    noise = np.random.normal(0, 0.1)
    performance = base_prob + noise
    
    # Clip to valid range
    return np.clip(performance, 0.0, 1.0)

def generate_interview_episode(simulator: InterviewSimulator, skill_level: float) -> dict:
    """Generate one complete interview episode"""
    episode = {
        "questions": [],
        "performances": [],
        "skill_level": skill_level
    }
    
    # Get first question
    question = simulator.start_interview()
    
    # Simulate 10 questions
    for i in range(10):
        # Generate performance based on skill level and question difficulty
        performance = generate_realistic_performance(question['difficulty'], skill_level)
        
        # Record question and performance
        episode["questions"].append({
            "id": question['id'],
            "topic": question['topic'],
            "subtopic": question['subtopic'],
            "difficulty": question['difficulty'],
            "question": question['question'],
            "time_allocated": question['expected_time_minutes']
        })
        episode["performances"].append({
            "score": float(performance),
            "time_taken": question['expected_time_minutes'] * random.uniform(0.8, 1.2)
        })
        
        # Get next question based on performance
        if i < 9:  # Don't get next question after the last one
            prev_topic = question['topic']
            question = simulator.next_question(performance, prev_topic)
    
    return episode

def generate_training_dataset(num_episodes: int = 10):
    """Generate multiple interview episodes for training"""
    # Initialize question bank and simulator
    question_bank = InterviewQuestionBank()
    simulator = InterviewSimulator(question_bank)
    
    # Generate episodes with varying skill levels
    episodes = []
    
    # Define different skill level profiles
    skill_profiles = [
        3.0,   # Beginner
        4.0,   # Advanced Beginner
        5.0,   # Intermediate
        6.0,   # Advanced Intermediate
        7.0,   # Advanced
        8.0,   # Expert
        # Add some random skill levels
        random.uniform(3.0, 8.0),
        random.uniform(3.0, 8.0),
        random.uniform(3.0, 8.0),
        random.uniform(3.0, 8.0)
    ]
    
    # Generate episodes
    for skill_level in skill_profiles:
        episode = generate_interview_episode(simulator, skill_level)
        episodes.append(episode)
    
    # Save to file
    dataset = {
        "num_episodes": len(episodes),
        "episodes": episodes
    }
    
    with open('training_episodes                                                                                                                            .json', 'w') as f:
        json.dump(dataset, f, indent=2)
    
    # Print summary statistics
    print(f"\nGenerated {len(episodes)} interview episodes:")
    for i, episode in enumerate(episodes):
        avg_performance = np.mean([p['score'] for p in episode['performances']])
        avg_difficulty = np.mean([q['difficulty'] for q in episode['questions']])
        print(f"\nEpisode {i+1}:")
        print(f"Skill Level: {episode['skill_level']:.2f}")
        print(f"Average Performance: {avg_performance:.2f}")
        print(f"Average Question Difficulty: {avg_difficulty:.2f}")
        print("Topic Sequence:", [q['topic'] for q in episode['questions']])

if __name__ == "__main__":
    generate_training_dataset() 