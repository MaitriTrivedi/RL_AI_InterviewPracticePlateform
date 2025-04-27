import argparse
import numpy as np
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from PPO_RL_AGENT.interview_agent import InterviewAgent
from environment.interview_env import InterviewEnvironment

def train_agent(num_interviews=100, model_version=None):
    """Train the agent for specified number of interviews."""
    # Initialize environment and agent
    env = InterviewEnvironment()
    agent = InterviewAgent(state_dim=6, model_version=model_version)
    
    # Training loop
    for interview in range(num_interviews):
        print(f"\nStarting Interview {interview + 1}/{num_interviews}")
        
        # Reset environment and agent state
        obs, _ = env.reset()
        agent.reset_interview_state()
        done = False
        
        while not done:
            # Get current topic and state
            topic = env.get_current_topic()
            
            # Get next question difficulty
            question_info = agent.get_next_question(topic)
            difficulty = question_info['difficulty']
            
            # Take step in environment
            obs, reward, terminated, truncated, info = env.step(difficulty)
            done = terminated or truncated
            
            # Update agent with performance
            agent.update_performance(
                topic=topic,
                performance_score=info['performance'],
                time_taken=info['time_taken']
            )
        
        # Train agent after each interview
        metrics = agent.train()
        if metrics:
            print(f"Training metrics - Policy Loss: {metrics['policy_loss']:.3f}, "
                  f"Value Loss: {metrics['value_loss']:.3f}, "
                  f"Mean Reward: {metrics['mean_reward']:.3f}")
        
        # Save checkpoint every 2 interviews
        if (interview + 1) % 2 == 0:
            agent.save_checkpoint(interview + 1)
            stats = agent.get_interview_stats()
            print(f"\nCheckpoint {interview + 1} Stats:")
            print(f"Average Score: {stats['average_score']:.2f}")
            print(f"Time Efficiency: {stats['time_efficiency']:.2f}")
            print("Topic Performances:", {t: f"{p:.2f}" for t, p in stats['topic_performances'].items()})
    
    # Save final model
    version = agent.save_model()
    print(f"\nTraining completed. Model saved with version: {version}")
    return version

def evaluate_agent(version):
    """Evaluate trained agent for 5 complete interviews."""
    env = InterviewEnvironment()
    agent = InterviewAgent(state_dim=6, model_version=version)
    
    total_rewards = []
    episode_lengths = []
    topic_performances = {topic: [] for topic in agent.topics}
    
    for episode in range(5):
        print(f"\nEvaluation Episode {episode + 1}/5")
        obs, _ = env.reset()
        agent.reset_interview_state()
        episode_reward = 0
        steps = 0
        done = False
        
        while not done:
            topic = env.get_current_topic()
            question_info = agent.get_next_question(topic)
            difficulty = question_info['difficulty']
            
            obs, reward, terminated, truncated, info = env.step(difficulty)
            done = terminated or truncated
            
            print(f"\nStep {steps + 1}:")
            print(f"Topic: {topic}")
            print(f"Difficulty: {difficulty:.2f}")
            print(f"Performance: {info['performance']:.2f}")
            print(f"Time: {info['time_taken']:.1f} min")
            print(f"Reward: {reward:.2f}")
            
            agent.update_performance(
                topic=topic,
                performance_score=info['performance'],
                time_taken=info['time_taken']
            )
            
            episode_reward += reward
            steps += 1
            
            # Store topic performance
            topic_performances[topic].append(info['performance'])
        
        print(f"\nEpisode {episode + 1} Total Reward: {episode_reward:.2f}")
        total_rewards.append(episode_reward)
        episode_lengths.append(steps)
    
    # Print evaluation summary
    print("\nEvaluation Summary:")
    print(f"Average Episode Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print("\nTopic-wise Performance:")
    for topic in topic_performances:
        if topic_performances[topic]:
            mean_perf = np.mean(topic_performances[topic])
            std_perf = np.std(topic_performances[topic])
            print(f"{topic}: {mean_perf:.2f} ± {std_perf:.2f}")

def main():
    parser = argparse.ArgumentParser(description='Train or evaluate PPO agent for interview difficulty adjustment')
    parser.add_argument('--mode', choices=['train', 'evaluate'], required=True, help='Mode of operation')
    parser.add_argument('--interviews', type=int, default=100, help='Number of interviews for training')
    parser.add_argument('--version', type=str, help='Model version for evaluation')
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_agent(num_interviews=args.interviews)
    else:
        if not args.version:
            raise ValueError("Model version must be specified for evaluation mode")
        evaluate_agent(args.version)

if __name__ == "__main__":
    main() 