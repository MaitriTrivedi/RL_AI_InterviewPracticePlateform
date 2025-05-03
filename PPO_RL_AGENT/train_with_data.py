import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
import os
import logging
from .interview_agent import InterviewAgent
from .training_history import TrainingHistory
from .training_data import load_training_data

def setup_monitoring(base_dir="monitor"):
    """Setup monitoring directory and logging."""
    # Create timestamp for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    monitor_dir = os.path.join(base_dir, timestamp)
    
    # Create directory structure
    os.makedirs(monitor_dir, exist_ok=True)
    os.makedirs(os.path.join(monitor_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(monitor_dir, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(monitor_dir, "checkpoints"), exist_ok=True)
    
    # Setup logging
    log_file = os.path.join(monitor_dir, "training.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also print to console
        ]
    )
    
    return monitor_dir

def plot_training_curves(history, monitor_dir):
    """Plot and save training curves."""
    plots_dir = os.path.join(monitor_dir, 'plots')
    
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

def save_metrics(metrics, monitor_dir, episode_idx):
    """Save metrics to JSON file."""
    metrics_file = os.path.join(monitor_dir, "metrics", f"metrics_episode_{episode_idx}.json")
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)

def train_agent(
    training_data_file,
    num_epochs=5,
    save_dir="training_results",
    checkpoint_frequency=10,
    eval_frequency=5
):
    """Train the PPO agent using pre-generated training data with detailed monitoring."""
    # Setup monitoring
    monitor_dir = setup_monitoring()
    logging.info(f"Starting training session. Monitoring directory: {monitor_dir}")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Load training data
    logging.info(f"Loading training data from {training_data_file}...")
    with open(training_data_file, 'r') as f:
        training_data = json.load(f)
    
    # Initialize agent and history tracker
    agent = InterviewAgent(state_dim=9)
    history = TrainingHistory()
    
    logging.info(f"Starting training for {num_epochs} epochs...")
    
    best_reward = float('-inf')
    topics = agent.topics
    total_episodes = 0
    
    # Training loop over epochs
    for epoch in range(num_epochs):
        logging.info(f"Starting Epoch {epoch + 1}/{num_epochs}")
        
        # Shuffle episodes for each epoch
        episode_indices = np.random.permutation(len(training_data))
        
        # Train on each episode
        for idx, episode_idx in enumerate(episode_indices):
            total_episodes += 1
            episode_data = training_data[episode_idx]
            
            # Reset agent state
            agent.reset_interview_state()
            
            episode_rewards = []
            episode_difficulties = []
            episode_times = []
            topic_scores = {topic: [] for topic in topics}
            
            # Process each question in the episode
            for question_data in episode_data['questions']:
                topic = question_data['topic']
                subtopic = question_data.get('subtopic', agent.subtopics[topic][0])
                
                # Get next question difficulty from agent
                action_info = agent.get_next_question(topic)
                difficulty = action_info['difficulty']
                episode_difficulties.append(difficulty)
                
                # Get performance from training data
                performance_score = question_data['performance']
                time_taken = question_data['time_taken']
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
            
            # Train agent after each episode
            metrics = agent.train()
            if metrics:
                history.add_training_metrics(
                    policy_loss=metrics['actor_loss'],
                    value_loss=metrics['value_loss'],
                    entropy_loss=0.0
                )
                
                # Save metrics
                metrics_data = {
                    'total_episodes': total_episodes,
                    'epoch': epoch + 1,
                    'epoch_episode': idx + 1,
                    'actor_loss': float(metrics['actor_loss']),
                    'value_loss': float(metrics['value_loss']),
                    'mean_reward': float(mean_episode_reward),
                    'topic_performance': {k: float(v) for k, v in topic_performance.items()}
                }
                save_metrics(metrics_data, monitor_dir, total_episodes)
                
                logging.info(f"Epoch {epoch + 1}, Episode {idx + 1} (Total: {total_episodes}) - Mean Reward: {mean_episode_reward:.3f}")
                
                # Save best model
                if mean_episode_reward > best_reward:
                    best_reward = mean_episode_reward
                    version = agent.save_model()
                    logging.info(f"New best model saved (version {version}, reward: {best_reward:.3f})")
            
            # Create checkpoint
            if total_episodes % checkpoint_frequency == 0:
                checkpoint_path = os.path.join(monitor_dir, "checkpoints", f"checkpoint_episode_{total_episodes}")
                os.makedirs(checkpoint_path, exist_ok=True)
                
                # Save checkpoint
                checkpoint_data = {
                    'episode': total_episodes,
                    'epoch': epoch + 1,
                    'mean_reward': float(mean_episode_reward),
                    'best_reward': float(best_reward),
                    'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
                }
                
                # Save checkpoint metadata
                with open(os.path.join(checkpoint_path, 'metadata.json'), 'w') as f:
                    json.dump(checkpoint_data, f, indent=2)
                
                # Save model checkpoint
                agent.save_checkpoint(total_episodes)
                history.create_snapshot()
                logging.info(f"Checkpoint saved at episode {total_episodes}")
                
                # Save current plots
                plot_training_curves(history, monitor_dir)
            
            # Evaluation metrics
            if total_episodes % eval_frequency == 0:
                stats = agent.get_interview_stats()
                eval_metrics = {
                    'total_episodes': total_episodes,
                    'epoch': epoch + 1,
                    'epoch_episode': idx + 1,
                    'average_score': float(stats['average_score']),
                    'time_efficiency': float(stats['time_efficiency']),
                    'current_streak': int(stats['current_streak']),
                    'difficulty_level': float(stats['difficulty_level']),
                    'topic_performances': {k: float(v) for k, v in stats['topic_performances'].items()}
                }
                save_metrics(eval_metrics, monitor_dir, f"{total_episodes}_eval")
    
    # Save final model and history
    final_version = agent.save_model()
    history.save_to_file(os.path.join(monitor_dir, 'training_history.npz'))
    
    # Generate final plots
    plot_training_curves(history, monitor_dir)
    
    logging.info("Training Complete!")
    logging.info(f"Final model version: {final_version}")
    logging.info(f"Best reward achieved: {best_reward:.3f}")
    logging.info(f"Total episodes trained: {total_episodes}")
    logging.info(f"Training history and monitoring data saved in: {monitor_dir}")
    
    return history

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-data", type=str, required=True,
                      help="Path to the training data JSON file")
    parser.add_argument("--epochs", type=int, default=5,
                      help="Number of epochs to train")
    parser.add_argument("--save-dir", type=str, default="training_results",
                      help="Directory to save results")
    parser.add_argument("--checkpoint-freq", type=int, default=10,
                      help="Save checkpoint every N episodes")
    parser.add_argument("--eval-freq", type=int, default=5,
                      help="Print evaluation metrics every N episodes")
    args = parser.parse_args()
    
    history = train_agent(
        training_data_file=args.training_data,
        num_epochs=args.epochs,
        save_dir=args.save_dir,
        checkpoint_frequency=args.checkpoint_freq,
        eval_frequency=args.eval_freq
    ) 