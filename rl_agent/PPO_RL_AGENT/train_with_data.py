import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
import os
import logging
from .interview_agent import InterviewAgent
from .training_history import TrainingHistory
from .training_data import load_training_data

def setup_monitoring(base_dir=None):
    """Setup monitoring directory and logging."""
    # Use main data/monitor directory if not specified
    if base_dir is None:
        base_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'data',
            'monitor'
        )
    
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
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot training losses
    policy_losses = [float(x) for x in history.policy_losses if x is not None]
    value_losses = [float(x) for x in history.value_losses if x is not None]
    entropy_losses = [float(x) for x in history.entropy_losses if x is not None]
    
    if policy_losses and value_losses:  # Only plot if we have valid data
        plt.figure(figsize=(10, 6))
        plt.plot(policy_losses, label='Policy Loss', alpha=0.8)
        plt.plot(value_losses, label='Value Loss', alpha=0.8)
        if entropy_losses:
            plt.plot(entropy_losses, label='Entropy Loss', alpha=0.8)
        plt.grid(True, alpha=0.3)
        plt.title('Training Losses')
        plt.xlabel('Update Step')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'losses.png'))
        plt.close()
        logging.info(f"Saved loss plot with {len(policy_losses)} points")
    
    # Plot episode rewards
    if history.episode_rewards:
        rewards = [float(x) for x in history.episode_rewards if x is not None]
        if rewards:
            plt.figure(figsize=(10, 6))
            plt.plot(rewards, alpha=0.8)
            plt.grid(True, alpha=0.3)
            plt.title('Episode Rewards Over Time')
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'rewards.png'))
            plt.close()
    
    # Plot topic performances
    if history.topic_performances:
        valid_topics = False
        plt.figure(figsize=(12, 6))
        for topic, scores in history.topic_performances.items():
            if scores:
                # Flatten nested lists and convert to float
                flattened_scores = []
                for score_list in scores:
                    if isinstance(score_list, (list, tuple, np.ndarray)):
                        flattened_scores.extend([float(s) for s in score_list if s is not None])
                    elif score_list is not None:
                        flattened_scores.append(float(score_list))
                if flattened_scores:  # Only plot if we have data
                    plt.plot(flattened_scores, label=topic, alpha=0.8)
                    valid_topics = True
        if valid_topics:
            plt.grid(True, alpha=0.3)
            plt.title('Topic Performance Over Time')
            plt.xlabel('Question Number')
            plt.ylabel('Performance Score')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'topic_performance.png'))
        plt.close()
    
    # Plot difficulty progression
    if history.difficulty_history:
        difficulties = []
        for diff in history.difficulty_history:
            if isinstance(diff, (list, tuple, np.ndarray)):
                difficulties.extend([float(d) for d in diff if d is not None])
            elif diff is not None:
                difficulties.append(float(diff))
        if difficulties:
            plt.figure(figsize=(10, 6))
            plt.plot(difficulties, alpha=0.8)
            plt.grid(True, alpha=0.3)
            plt.title('Question Difficulty Over Time')
            plt.xlabel('Question Number')
            plt.ylabel('Difficulty Level')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'difficulty.png'))
            plt.close()

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    return obj

def save_metrics(metrics, monitor_dir, episode_idx):
    """Save metrics to JSON file."""
    # Convert numpy types to Python native types
    metrics = convert_numpy_types(metrics)
    
    metrics_file = os.path.join(monitor_dir, "metrics", f"metrics_episode_{episode_idx}.json")
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)

def train_agent(
    training_data_file,
    num_epochs=5,
    save_dir=None,
    checkpoint_frequency=10,
    eval_frequency=5
):
    """Train the PPO agent using pre-generated training data with detailed monitoring."""
    # Use main data directory if save_dir not specified
    if save_dir is None:
        save_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'data',
            'training_results'
        )
    
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
            for question_idx, question_data in enumerate(episode_data['questions']):
                topic = question_data['topic']
                subtopic = question_data.get('subtopic', agent.subtopics[topic][0])
                
                # Get agent's action (difficulty level) with safety constraints
                action_tuple = agent.get_next_question(topic)
                raw_difficulty = float(action_tuple[0]) if isinstance(action_tuple, (tuple, list)) else 5.0
                
                # Apply safety constraints based on performance history
                if question_idx > 0:
                    # Get recent performance metrics
                    recent_scores = episode_rewards[-3:]
                    avg_recent_score = float(np.mean(recent_scores)) if recent_scores else 0.5
                    
                    # Count consecutive low scores
                    consecutive_low_scores = sum(1 for score in reversed(episode_rewards) if score < 0.3)
                    
                    # Apply safety constraints
                    if consecutive_low_scores >= 2:
                        # Force difficulty reduction on multiple low scores
                        difficulty = max(1.0, raw_difficulty * 0.6)
                    else:
                        # Normal difficulty progression
                        max_increase = min(2.0, avg_recent_score * 3)
                        if raw_difficulty > agent.current_difficulty:
                            difficulty = min(raw_difficulty, agent.current_difficulty + max_increase)
                        else:
                            difficulty = raw_difficulty
                else:
                    difficulty = raw_difficulty
                
                # Consider topic difficulty in final action
                topic_base_diff = agent.topic_difficulty[topic]['base']
                topic_max_diff = agent.topic_difficulty[topic]['max']
                topic_factor = (topic_base_diff / 3.0)  # Normalize base difficulty to 0-1
                max_topic_difficulty = topic_max_diff  # Use topic's max difficulty as ceiling
                difficulty = min(difficulty, max_topic_difficulty)
                
                # Ensure difficulty is within valid range
                difficulty = float(np.clip(difficulty, 1.0, 10.0))
                
                # Get performance from training data
                performance_score = float(question_data['performance'])
                time_taken = float(question_data['time_taken'])
                
                # Record metrics BEFORE updating agent
                episode_rewards.append(float(performance_score))
                episode_difficulties.append(float(difficulty))
                topic_scores[topic].append(float(performance_score))
                episode_times.append(float(time_taken))
                
                # Update agent with enhanced reward calculation
                metrics = agent.update_performance(
                    topic=topic,
                    subtopic=subtopic,
                    performance_score=performance_score,
                    time_taken=time_taken
                )
                
                # Log training metrics if available
                if metrics and isinstance(metrics, dict):
                    # Store training metrics in history
                    history.add_training_metrics(
                        policy_loss=float(metrics.get('policy_loss', 0.0)),
                        value_loss=float(metrics.get('value_loss', 0.0)),
                        entropy_loss=float(metrics.get('entropy_loss', 0.0))
                    )
                    
                    # Log metrics
                    logging.info(
                        f"Training step {total_episodes} - "
                        f"Policy Loss: {metrics.get('policy_loss', 0.0):.3f}, "
                        f"Value Loss: {metrics.get('value_loss', 0.0):.3f}, "
                        f"Entropy Loss: {metrics.get('entropy_loss', 0.0):.3f}, "
                        f"Mean Reward: {performance_score:.3f}"
                    )
                    
                    # Calculate mean scores for each topic
                    topic_means = {
                        k: float(np.mean([float(s) for s in v])) if v else 0.0 
                        for k, v in topic_scores.items()
                    }
                    
                    # Save detailed metrics
                    metrics_data = {
                        'total_episodes': int(total_episodes),
                        'epoch': int(epoch + 1),
                        'epoch_episode': int(idx + 1),
                        'question_number': int(question_idx + 1),
                        'topic': topic,
                        'subtopic': subtopic,
                        'difficulty': float(difficulty),
                        'performance': float(performance_score),
                        'time_taken': float(time_taken),
                        'policy_loss': float(metrics.get('policy_loss', 0.0)),
                        'value_loss': float(metrics.get('value_loss', 0.0)),
                        'entropy_loss': float(metrics.get('entropy_loss', 0.0)),
                        'topic_means': topic_means,
                        'consecutive_low_scores': int(consecutive_low_scores if question_idx > 0 else 0),
                        'safety_constraints_applied': bool(difficulty != raw_difficulty)
                    }
                    save_metrics(metrics_data, monitor_dir, total_episodes)
            
            # End of episode processing - only calculate means if we have data
            if episode_rewards:
                episode_mean_reward = float(np.mean([float(r) for r in episode_rewards]))
                episode_mean_difficulty = float(np.mean([float(d) for d in episode_difficulties]))
            else:
                episode_mean_reward = 0.0
                episode_mean_difficulty = 0.0
            
            # Update history
            history.add_episode(
                total_reward=episode_mean_reward,
                length=len(episode_rewards),
                topic_scores={k: [float(s) for s in v] for k, v in topic_scores.items()},
                difficulties=[float(d) for d in episode_difficulties],
                time_stats=[float(t) for t in episode_times]
            )
            
            # Log episode summary with more details
            logging.info(
                f"Epoch {epoch + 1}, Episode {idx + 1} - "
                f"Mean Reward: {episode_mean_reward:.3f}, "
                f"Mean Difficulty: {episode_mean_difficulty:.3f}, "
                f"Topic: {topic}, "
                f"Questions: {len(episode_rewards)}"
            )

            # Log detailed topic performance only if we have data
            if topic_scores:
                topic_averages = {
                    k: float(np.mean([float(s) for s in v])) if v else 0.0 
                    for k, v in topic_scores.items()
                }
                for t, avg in topic_averages.items():
                    if avg > 0:  # Only log topics that were covered
                        logging.info(f"  {t}: {avg:.3f}")
                
            # Save checkpoint if best performance
            if episode_mean_reward > best_reward:
                best_reward = episode_mean_reward
                checkpoint_path = os.path.join(monitor_dir, "checkpoints", "best_model.pt")
                agent.save_checkpoint(checkpoint_path)
                logging.info(f"New best model saved with reward: {best_reward:.3f}")
            
            # Regular checkpoint saving
            if total_episodes % checkpoint_frequency == 0:
                checkpoint_path = os.path.join(
                    monitor_dir, 
                    "checkpoints", 
                    f"checkpoint_ep{total_episodes}.pt"
                )
                agent.save_checkpoint(checkpoint_path)
                # Generate plots at checkpoints
                plot_training_curves(history, monitor_dir)
                logging.info(f"Generated training plots at episode {total_episodes}")
        
        # End of epoch processing
        # Save final model for this epoch
        model_path = os.path.join(save_dir, f"model_epoch_{epoch + 1}.pt")
        agent.save_model()
        logging.info(f"Saved model for epoch {epoch + 1}")
        
        # Generate plots at end of each epoch
        plot_training_curves(history, monitor_dir)
        logging.info(f"Generated training plots for epoch {epoch + 1}")
    
    # Final plotting after all training is complete
    plot_training_curves(history, monitor_dir)
    logging.info("Generated final training plots")
    return agent, history

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-data", type=str, required=True,
                      help="Path to the training data JSON file")
    parser.add_argument("--epochs", type=int, default=5,
                      help="Number of epochs to train")
    parser.add_argument("--save-dir", type=str, default=None,
                      help="Directory to save results (defaults to data/training_results)")
    parser.add_argument("--checkpoint-freq", type=int, default=100,
                      help="Save checkpoint every N episodes")
    parser.add_argument("--eval-freq", type=int, default=20,
                      help="Print evaluation metrics every N episodes")
    args = parser.parse_args()
    
    agent, history = train_agent(
        training_data_file=args.training_data,
        num_epochs=args.epochs,
        save_dir=args.save_dir,
        checkpoint_frequency=args.checkpoint_freq,
        eval_frequency=args.eval_freq
    ) 