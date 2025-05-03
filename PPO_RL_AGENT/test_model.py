import argparse
import numpy as np
from interview_agent import InterviewAgent

def simulate_student(base_skill, topic_strengths=None):
    """Simulate a student with given skill level and topic strengths."""
    if topic_strengths is None:
        topic_strengths = {}
    
    def get_performance(topic, difficulty, time_factor=1.0):
        # Get topic-specific skill adjustment
        topic_skill = topic_strengths.get(topic, 0.0)
        effective_skill = base_skill + topic_skill
        
        # Calculate performance based on difficulty match
        performance_factor = max(0.1, 1.0 - abs(difficulty - effective_skill) / 10.0)
        performance_noise = np.random.normal(0, 0.1)
        performance = min(1.0, max(0.0, performance_factor + performance_noise))
        
        # Calculate time taken
        expected_time = (5 + difficulty) * time_factor
        time_taken = expected_time * (1 + np.random.normal(0, 0.2))
        
        return performance, time_taken
    
    return get_performance

def test_agent(model_version, num_interviews=5):
    """Test the trained agent with different student profiles."""
    # Initialize agent with trained model
    agent = InterviewAgent(state_dim=9, model_version=model_version)
    
    # Define different student profiles
    student_profiles = [
        {
            'name': 'Strong Student',
            'base_skill': 8.0,
            'topic_strengths': {
                'ds': 1.0,
                'algo': 1.0,
                'system_design': 0.5
            },
            'time_factor': 0.8
        },
        {
            'name': 'Average Student',
            'base_skill': 5.0,
            'topic_strengths': {
                'oops': 0.5,
                'dbms': 0.5
            },
            'time_factor': 1.0
        },
        {
            'name': 'Struggling Student',
            'base_skill': 3.0,
            'topic_strengths': {
                'os': -0.5,
                'cn': -0.5
            },
            'time_factor': 1.2
        },
        {
            'name': 'Inconsistent Student',
            'base_skill': 6.0,
            'topic_strengths': {
                'ds': 2.0,
                'algo': -1.0,
                'dbms': 1.0,
                'os': -1.0
            },
            'time_factor': 1.1
        },
        {
            'name': 'Improving Student',
            'base_skill': 4.0,
            'topic_strengths': {},
            'time_factor': 0.9
        }
    ]
    
    results = {}
    
    # Test each student profile
    for profile in student_profiles:
        print(f"\nTesting with {profile['name']}:")
        profile_results = []
        
        # Create student simulator
        student = simulate_student(
            profile['base_skill'],
            profile['topic_strengths']
        )
        
        # Run multiple interviews
        for interview in range(num_interviews):
            print(f"\nInterview {interview + 1}/{num_interviews}")
            
            # Reset agent state
            agent.reset_interview_state()
            interview_scores = []
            topics_covered = []
            difficulties = []
            
            # Run one complete interview (10 questions)
            for step in range(10):
                # Get current topic based on coverage
                topic_weights = [1.0 / (1.0 + agent.question_history[t]) for t in agent.topics]
                topic = np.random.choice(agent.topics, p=np.array(topic_weights)/sum(topic_weights))
                topics_covered.append(topic)
                
                # Get next question difficulty
                action_info = agent.get_next_question(topic)
                difficulty = action_info['difficulty']
                difficulties.append(difficulty)
                
                # Get simulated performance
                performance_score, time_taken = student(
                    topic,
                    difficulty,
                    profile['time_factor']
                )
                
                # Update agent
                agent.update_performance(
                    topic=topic,
                    performance_score=performance_score,
                    time_taken=time_taken
                )
                
                interview_scores.append(performance_score)
                
                # Print question details
                print(f"Q{step+1}: Topic={topic}, Difficulty={difficulty:.1f}, Score={performance_score:.2f}")
            
            # Calculate interview statistics
            stats = agent.get_interview_stats()
            profile_results.append({
                'average_score': np.mean(interview_scores),
                'final_difficulty': stats['difficulty_level'],
                'topics_covered': set(topics_covered),
                'difficulty_range': (min(difficulties), max(difficulties)),
                'time_efficiency': stats['time_efficiency'],
                'topic_performances': stats['topic_performances']
            })
            
            # Print interview summary
            print(f"\nInterview {interview + 1} Summary:")
            print(f"Average Score: {np.mean(interview_scores):.2f}")
            print(f"Final Difficulty: {stats['difficulty_level']:.1f}")
            print(f"Time Efficiency: {stats['time_efficiency']:.2f}")
            print("Topic Performances:")
            for topic, score in stats['topic_performances'].items():
                if score > 0:  # Only show covered topics
                    print(f"  {topic}: {score:.2f}")
        
        # Store results for this profile
        results[profile['name']] = profile_results
    
    # Print overall analysis
    print("\nOverall Analysis:")
    for profile_name, profile_results in results.items():
        avg_scores = [r['average_score'] for r in profile_results]
        avg_difficulties = [r['final_difficulty'] for r in profile_results]
        
        print(f"\n{profile_name}:")
        print(f"Average Score: {np.mean(avg_scores):.2f} ± {np.std(avg_scores):.2f}")
        print(f"Average Final Difficulty: {np.mean(avg_difficulties):.1f} ± {np.std(avg_difficulties):.1f}")
        
        # Analyze adaptation
        if len(avg_scores) > 1:
            score_trend = np.polyfit(range(len(avg_scores)), avg_scores, 1)[0]
            diff_trend = np.polyfit(range(len(avg_difficulties)), avg_difficulties, 1)[0]
            print(f"Score Trend: {'Improving' if score_trend > 0 else 'Declining'} ({score_trend:.3f}/interview)")
            print(f"Difficulty Trend: {'Increasing' if diff_trend > 0 else 'Decreasing'} ({diff_trend:.3f}/interview)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-version", type=str, required=True,
                      help="Version of the model to test")
    parser.add_argument("--num-interviews", type=int, default=5,
                      help="Number of interviews per student profile")
    args = parser.parse_args()
    
    test_agent(
        model_version=args.model_version,
        num_interviews=args.num_interviews
    ) 