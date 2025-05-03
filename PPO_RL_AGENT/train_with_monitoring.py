import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
import os
from .interview_agent import InterviewAgent
from .training_history import TrainingHistory

def plot_training_curves(history, save_dir):
    """Plot and save training curves."""
    # Create plots directory
    plots_dir = os.path.join(save_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot episode rewards
    plt.figure(figsize=(10, 6))
    plt.plot(history.episode_rewards)
    plt.title('Episode Rewards Over Time')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.savefig(os.path.join(plots_dir, 'rewards.png'))
    plt.close()
    
    # Plot topic performances
    plt.figure(figsize=(12, 6))
    for topic, scores in history.topic_performances.items():
        plt.plot(scores, label=topic)
    plt.title('Topic Performance Over Time')
    plt.xlabel('Question Number')
    plt.ylabel('Performance Score')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'topic_performance.png'))
    plt.close()
    
    # Plot training losses
    plt.figure(figsize=(10, 6))
    plt.plot(history.policy_losses, label='Policy Loss')
    plt.plot(history.value_losses, label='Value Loss')
    plt.title('Training Losses')
    plt.xlabel('Update Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'losses.png'))
    plt.close()
    
    # Plot difficulty progression
    plt.figure(figsize=(10, 6))
    plt.plot(history.difficulty_history)
    plt.title('Question Difficulty Over Time')
    plt.xlabel('Question Number')
    plt.ylabel('Difficulty Level')
    plt.savefig(os.path.join(plots_dir, 'difficulty.png'))
    plt.close()

def train_agent(
    num_episodes=100,
    save_dir="training_results",
    checkpoint_frequency=10,
    eval_frequency=5,
    use_simulated_data=True
):
    """Train the PPO agent with detailed monitoring."""
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize agent and history tracker
    agent = InterviewAgent(state_dim=9)
    history = TrainingHistory()
    
    # Initialize student simulator if using simulated data
    if use_simulated_data:
        from .training_data import StudentSimulator
        student = StudentSimulator()
    
    print(f"\nStarting training for {num_episodes} episodes...")
    print(f"Results will be saved in: {save_dir}")
    
    best_reward = float('-inf')
    topics = agent.topics
    
    for episode in range(num_episodes):
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        
        # Reset agent and student state
        agent.reset_interview_state()
        if use_simulated_data:
            student.reset()
            
        episode_rewards = []
        episode_difficulties = []
        episode_times = []
        topic_scores = {topic: [] for topic in topics}
        
        # Run one complete interview (10 questions)
        for step in range(10):
            # Select topic based on performance history
            topic_weights = [1.0 / (1.0 + agent.question_history[t]) for t in topics]
            topic = np.random.choice(topics, p=np.array(topic_weights)/sum(topic_weights))
            
            # Select subtopic
            subtopic = agent.select_subtopic(topic)
            
            # Get next question difficulty
            action_info = agent.get_next_question(topic)
            difficulty = action_info['difficulty']
            episode_difficulties.append(difficulty)
            
            if use_simulated_data:
                # Get performance from simulated student
                performance_score, time_taken = student.get_performance(topic, subtopic, difficulty)
            else:
                # Use simple simulation
                base_skill = 5.0 + agent.current_streak * 0.5
                performance_factor = max(0.1, 1.0 - abs(difficulty - base_skill) / 10.0)
                performance_noise = np.random.normal(0, 0.1)
                performance_score = min(1.0, max(0.0, performance_factor + performance_noise))
                
                # Simulate time taken
                expected_time = 5 + difficulty
                time_taken = expected_time * (1 + np.random.normal(0, 0.2))
            
            episode_times.append(time_taken)
            
            # Update agent
            agent.update_performance(
                topic=topic,
                subtopic=subtopic,
                performance_score=performance_score,
                time_taken=time_taken
            )
            
            # Record metrics
            episode_rewards.append(performance_score)
            topic_scores[topic].append(performance_score)
        
        # Calculate episode statistics
        mean_episode_reward = np.mean(episode_rewards)
        topic_performance = {t: np.mean(s) if s else 0 for t, s in topic_scores.items()}
        
        # Update history
        history.add_episode(
            total_reward=mean_episode_reward,
            length=len(episode_rewards),
            topic_scores=topic_performance,
            difficulties=episode_difficulties,
            time_stats=episode_times
        )
        
        # Train agent
        if episode > 0 and episode % 2 == 0:
            metrics = agent.train()
            if metrics:
                history.add_training_metrics(
                    policy_loss=metrics['actor_loss'],
                    value_loss=metrics['value_loss'],
                    entropy_loss=0.0  # Not tracked in current implementation
                )
                
                print(f"\nTraining metrics:")
                print(f"  Actor Loss: {metrics['actor_loss']:.3f}")
                print(f"  Value Loss: {metrics['value_loss']:.3f}")
                print(f"  Mean Reward: {mean_episode_reward:.3f}")
                
                # Save best model
                if mean_episode_reward > best_reward:
                    best_reward = mean_episode_reward
                    version = agent.save_model()
                    print(f"  New best model saved (version {version})")
        
        # Create checkpoint
        if episode > 0 and episode % checkpoint_frequency == 0:
            checkpoint_id = agent.save_checkpoint(episode)
            history.create_snapshot()
            print(f"\nCheckpoint saved: {checkpoint_id}")
        
        # Print evaluation metrics
        if episode > 0 and episode % eval_frequency == 0:
            stats = agent.get_interview_stats()
            print("\nCurrent Progress:")
            print(f"  Average Score: {stats['average_score']:.3f}")
            print(f"  Time Efficiency: {stats['time_efficiency']:.3f}")
            print(f"  Current Streak: {stats['current_streak']}")
            print(f"  Difficulty Level: {stats['difficulty_level']:.1f}")
            print("  Topic Performances:")
            for topic, score in stats['topic_performances'].items():
                print(f"    {topic}: {score:.3f}")
            
            # Plot current progress
            plot_training_curves(history, save_dir)
    
    # Save final model and history
    final_version = agent.save_model()
    history.save_to_file(os.path.join(save_dir, 'training_history.npz'))
    
    # Generate final plots
    plot_training_curves(history, save_dir)
    
    print("\nTraining Complete!")
    print(f"Final model version: {final_version}")
    print(f"Training history and plots saved in: {save_dir}")
    
    return history

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--save-dir", type=str, default="training_results")
    parser.add_argument("--checkpoint-freq", type=int, default=10)
    parser.add_argument("--eval-freq", type=int, default=5)
    parser.add_argument("--use-simulated-data", action="store_true")
    args = parser.parse_args()
    
    history = train_agent(
        num_episodes=args.episodes,
        save_dir=args.save_dir,
        checkpoint_frequency=args.checkpoint_freq,
        eval_frequency=args.eval_freq,
        use_simulated_data=args.use_simulated_data
    ) 