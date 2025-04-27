import argparse
import numpy as np
from interview_agent import InterviewAgent

def train_agent(
    num_episodes=20,
    save_dir="models"
):
    """Train the PPO agent for interview question selection."""
    # Initialize agent
    agent = InterviewAgent(state_dim=9)  # 5 base features + 4 topic history
    
    print("\nStarting fresh training (no previous models found)")
    print(f"\nStarting training for {num_episodes} episodes...")
    
    metrics_history = []
    best_reward = float('-inf')
    topics = agent.topics
    
    for episode in range(num_episodes):
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        
        # Reset agent state
        agent.reset_interview_state()
        episode_rewards = []
        
        # Run one complete interview (10 questions)
        for step in range(10):
            # Get current topic based on performance history
            topic_weights = [1.0 / (1.0 + agent.question_history[t]) for t in topics]
            topic = np.random.choice(topics, p=np.array(topic_weights)/sum(topic_weights))
            
            # Get next question difficulty
            action_info = agent.get_next_question(topic)
            
            # Simulate student performance based on difficulty
            difficulty = action_info['difficulty']
            
            # Base performance depends on difficulty and streak
            base_skill = 5.0 + agent.current_streak * 0.5  # Improve with streak
            performance_factor = max(0.1, 1.0 - abs(difficulty - base_skill) / 10.0)
            performance_noise = np.random.normal(0, 0.1)
            performance_score = min(1.0, max(0.0, performance_factor + performance_noise))
            
            # Simulate time taken based on difficulty
            expected_time = 5 + difficulty  # Base time increases with difficulty
            time_taken = expected_time * (1 + np.random.normal(0, 0.2))  # Add noise
            
            # Update agent with simulated performance
            agent.update_performance(
                topic=topic,
                performance_score=performance_score,
                time_taken=time_taken
            )
            
            episode_rewards.append(performance_score)
        
        # Train after each episode
        if episode > 0 and episode % 2 == 0:
            metrics = agent.train()
            if metrics:
                metrics['episode'] = episode
                metrics['mean_episode_reward'] = np.mean(episode_rewards)
                metrics_history.append(metrics)
                
                print(f"\nTraining metrics:")
                print(f"  Actor Loss: {metrics['actor_loss']:.3f}")
                print(f"  Value Loss: {metrics['value_loss']:.3f}")
                print(f"  Mean Reward: {metrics['mean_episode_reward']:.3f}")
                
                # Save best model
                if metrics['mean_episode_reward'] > best_reward:
                    best_reward = metrics['mean_episode_reward']
                    version = agent.save_model()
                    print(f"  New best model saved (version {version})")
        
        # Print progress periodically
        if episode > 0 and episode % 5 == 0:
            stats = agent.get_interview_stats()
            print("\nCurrent Progress:")
            print(f"  Average Score: {stats['average_score']:.3f}")
            print(f"  Time Efficiency: {stats['time_efficiency']:.3f}")
            print(f"  Current Streak: {stats['current_streak']}")
            print(f"  Difficulty Level: {stats['difficulty_level']:.1f}")
            print("  Topic Performances:")
            for topic, score in stats['topic_performances'].items():
                print(f"    {topic}: {score:.3f}")
    
    # Save final model
    final_version = agent.save_model()
    
    print("\nTraining Complete!")
    print(f"Final model version: {final_version}")
    
    return metrics_history

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--save-dir", type=str, default="models")
    args = parser.parse_args()
    
    metrics = train_agent(
        num_episodes=args.episodes,
        save_dir=args.save_dir
    ) 