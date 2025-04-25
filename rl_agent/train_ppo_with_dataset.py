import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import json
from typing import Dict, List, Tuple
import torch.nn as nn
from gymnasium import spaces
from model_handler import ModelHandler

class InterviewEnvWithDataset(gym.Env):
    def __init__(self, dataset_path: str = 'training_episodes.json'):
        super().__init__()
        
        # Load dataset
        with open(dataset_path, 'r') as f:
            self.dataset = json.load(f)
        
        # Environment setup
        self.current_episode_idx = 0
        self.current_step = 0
        self.max_steps = 10  # 10 questions per interview
        
        # State space: [current_topic_onehot(7), current_difficulty, avg_performance_so_far, 
        #              time_efficiency, questions_remaining]
        self.observation_space = spaces.Box(
            low=np.array([0]*7 + [1, 0, 0, 0], dtype=np.float32),
            high=np.array([1]*7 + [10, 1, 2, 10], dtype=np.float32)
        )
        
        # Action space: difficulty adjustment (-2 to +2)
        self.action_space = spaces.Box(
            low=np.array([-2], dtype=np.float32),
            high=np.array([2], dtype=np.float32)
        )
        
        # Topic mapping
        self.topics = ['ds', 'algo', 'oops', 'dbms', 'os', 'cn', 'system_design']
        
    def _get_topic_onehot(self, topic: str) -> np.ndarray:
        """Convert topic to one-hot encoding"""
        onehot = np.zeros(len(self.topics))
        topic_idx = self.topics.index(topic)
        onehot[topic_idx] = 1
        return onehot
    
    def reset(self) -> np.ndarray:
        """Reset environment and return initial state"""
        # Get random episode
        self.current_episode_idx = np.random.randint(0, len(self.dataset['episodes']))
        self.current_episode = self.dataset['episodes'][self.current_episode_idx]
        self.current_step = 0
        
        # Get initial state
        return self._get_observation()
    
    def _get_observation(self) -> np.ndarray:
        """Get current state observation"""
        current_question = self.current_episode['questions'][self.current_step]
        performances = self.current_episode['performances'][:self.current_step]
        
        # Get topic one-hot encoding
        topic_onehot = self._get_topic_onehot(current_question['topic'])
        
        # Calculate average performance so far
        avg_performance = np.mean([p['score'] for p in performances]) if performances else 0.5
        
        # Calculate time efficiency
        time_efficiency = np.mean([p['time_taken'] / q['time_allocated'] 
                                 for p, q in zip(performances, self.current_episode['questions'][:self.current_step])]) if performances else 1.0
        
        return np.concatenate([
            topic_onehot,
            [current_question['difficulty'] / 10.0],  # Normalize difficulty
            [avg_performance],
            [time_efficiency],
            [(self.max_steps - self.current_step) / 10.0]  # Normalize questions remaining
        ]).astype(np.float32)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one step in the environment"""
        # Get current question and performance
        current_question = self.current_episode['questions'][self.current_step]
        current_performance = self.current_episode['performances'][self.current_step]
        
        # Calculate reward
        reward = self._calculate_reward(action[0], current_question, current_performance)
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        return self._get_observation(), reward, done, {
            'question': current_question,
            'performance': current_performance
        }
    
    def _calculate_reward(self, action: float, question: Dict, performance: Dict) -> float:
        """Calculate reward based on action and actual performance"""
        reward = 0
        
        # Base reward from performance
        reward += performance['score']
        
        # Penalty for inappropriate difficulty adjustment
        difficulty_mismatch = abs(action - (performance['score'] - 0.5) * 4)
        reward -= 0.2 * difficulty_mismatch
        
        # Time efficiency bonus/penalty
        time_efficiency = performance['time_taken'] / question['time_allocated']
        if 0.8 <= time_efficiency <= 1.2:
            reward += 0.2
        else:
            reward -= 0.1
        
        return reward

def make_env():
    """Create and wrap the interview environment"""
    return InterviewEnvWithDataset()

def train_ppo(total_interviews: int = 10, checkpoint_freq_interviews: int = 2, continue_version: str = None):
    """Train the PPO agent with checkpoints based on number of complete interviews
    
    Args:
        total_interviews: Total number of complete interview sets to train for (default: 10 interviews per day)
        checkpoint_freq_interviews: Save checkpoint every N complete interviews (default: every 2 interviews)
        continue_version: If provided, continue training from this model version
    """
    # Create model handler
    model_handler = ModelHandler()
    
    # Create vectorized environment
    env = DummyVecEnv([make_env])
    
    # Calculate timesteps based on interviews (each interview is 10 questions/steps)
    STEPS_PER_INTERVIEW = 10
    total_timesteps = total_interviews * STEPS_PER_INTERVIEW
    checkpoint_freq = checkpoint_freq_interviews * STEPS_PER_INTERVIEW
    
    if continue_version:
        # Load existing model and its metrics
        model, prev_metrics = model_handler.load_model(continue_version)
        print(f"Continuing training from model version: {continue_version}")
        print(f"Previous training interviews: {prev_metrics.get('total_interviews', 0)}")
        
        # Update model's environment
        model.set_env(env)
    else:
        # Initialize new PPO agent
        policy_kwargs = dict(
            net_arch=[dict(pi=[128, 64], vf=[128, 64])]
        )
        
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=1
        )
    
    # Training loop with checkpoints
    for interview_num in range(0, total_interviews, checkpoint_freq_interviews):
        current_timesteps = checkpoint_freq
        
        # Train for checkpoint_freq steps
        model.learn(total_timesteps=current_timesteps)
        
        # Evaluate current performance
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5)
        
        # Calculate interviews completed
        interviews_completed = (interview_num + checkpoint_freq_interviews)
        
        # Save checkpoint with metrics
        metrics = {
            'interviews_completed': interviews_completed,
            'timestep': (interview_num + checkpoint_freq_interviews) * STEPS_PER_INTERVIEW,
            'mean_reward': float(mean_reward),
            'std_reward': float(std_reward),
            'training_params': {
                'learning_rate': float(model.learning_rate),
                'n_steps': model.n_steps,
                'batch_size': model.batch_size,
                'n_epochs': model.n_epochs
            }
        }
        
        if continue_version:
            metrics['continued_from'] = continue_version
            metrics['previous_interviews'] = prev_metrics.get('total_interviews', 0)
        
        model_handler.save_checkpoint(model, metrics, interviews_completed)
        print(f"\nCheckpoint saved after {interviews_completed} interviews")
        print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    # Final evaluation
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    
    # Save final model with metrics
    final_metrics = {
        'total_interviews': total_interviews,
        'total_timesteps': total_timesteps,
        'final_mean_reward': float(mean_reward),
        'final_std_reward': float(std_reward),
        'training_params': {
            'learning_rate': float(model.learning_rate),
            'n_steps': model.n_steps,
            'batch_size': model.batch_size,
            'n_epochs': model.n_epochs
        }
    }
    
    if continue_version:
        final_metrics['continued_from'] = continue_version
        final_metrics['previous_interviews'] = prev_metrics.get('total_interviews', 0)
        final_metrics['total_cumulative_interviews'] = total_interviews + prev_metrics.get('total_interviews', 0)
    
    version = model_handler.save_model(model, final_metrics)
    print(f"\nTraining completed and model saved with version: {version}")
    print(f"Final mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    return model, version

def evaluate_agent(version: str = None, n_eval_episodes: int = 5):
    """Evaluate a specific version of the trained agent"""
    model_handler = ModelHandler()
    
    # Load model
    if version is None:
        model, metrics = model_handler.load_latest_model()
        print("Loaded latest model")
    else:
        model, metrics = model_handler.load_model(version)
        print(f"Loaded model version: {version}")
        
    # Print training history if available
    if metrics.get('training_history'):
        print("\nTraining History:")
        print(f"Total interviews trained on: {metrics.get('total_interviews', 0)}")
        if metrics.get('continued_from'):
            print(f"Continued from version: {metrics.get('continued_from')}")
            print(f"Total cumulative interviews: {metrics.get('total_cumulative_interviews', 0)}")
    
    env = make_env()
    
    # Tracking metrics across episodes
    all_rewards = []
    topic_performance = {topic: [] for topic in env.topics}
    difficulty_adjustments = []
    time_efficiency = []
    
    for episode in range(n_eval_episodes):
        obs = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        print(f"\nEvaluation Episode {episode + 1}")
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            
            # Track metrics
            topic_performance[info['question']['topic']].append(info['performance']['score'])
            difficulty_adjustments.append(action[0])
            time_efficiency.append(info['performance']['time_taken'] / info['question']['time_allocated'])
            
            print(f"\nStep {step + 1}:")
            print(f"Question Topic: {info['question']['topic']}")
            print(f"Question Difficulty: {info['question']['difficulty']}")
            print(f"Student Performance: {info['performance']['score']:.2f}")
            print(f"Time Taken: {info['performance']['time_taken']:.1f} minutes")
            print(f"Time Efficiency: {(info['performance']['time_taken'] / info['question']['time_allocated']):.2f}")
            print(f"Action (Difficulty Adjustment): {action[0]:.2f}")
            print(f"Reward: {reward:.2f}")
            
            step += 1
        
        all_rewards.append(episode_reward)
        print(f"\nEpisode Total Reward: {episode_reward:.2f}")
    
    # Print summary statistics
    print("\n=== Evaluation Summary ===")
    print(f"Average Episode Reward: {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}")
    print("\nTopic-wise Performance:")
    for topic, scores in topic_performance.items():
        if scores:  # Only print if we have data for this topic
            print(f"{topic}: {np.mean(scores):.2f} ± {np.std(scores):.2f}")
    
    print("\nDifficulty Adjustment Stats:")
    print(f"Average adjustment: {np.mean(difficulty_adjustments):.2f}")
    print(f"Adjustment range: [{np.min(difficulty_adjustments):.2f}, {np.max(difficulty_adjustments):.2f}]")
    
    print("\nTime Efficiency Stats:")
    print(f"Average efficiency: {np.mean(time_efficiency):.2f}")
    print(f"Efficiency range: [{np.min(time_efficiency):.2f}, {np.max(time_efficiency):.2f}]")
    
    # Compare with previous version if this is a continued training
    if metrics.get('continued_from'):
        print("\nComparison with Previous Version:")
        prev_model, prev_metrics = model_handler.load_model(metrics['continued_from'])
        print(f"Previous version mean reward: {prev_metrics.get('final_mean_reward', 'N/A')}")
        print(f"Current version mean reward: {metrics.get('final_mean_reward', 'N/A')}")
        if all(isinstance(x, (int, float)) for x in [prev_metrics.get('final_mean_reward'), metrics.get('final_mean_reward')]):
            improvement = ((metrics.get('final_mean_reward') - prev_metrics.get('final_mean_reward')) / 
                         abs(prev_metrics.get('final_mean_reward'))) * 100
            print(f"Improvement: {improvement:.1f}%")

def list_available_models():
    """List all available trained models"""
    model_handler = ModelHandler()
    models = model_handler.list_available_models()
    
    print("\nAvailable Models:")
    for version, info in models.items():
        print(f"\nVersion: {version}")
        print(f"Created at: {info['created_at']}")
        print(f"Average Reward: {info['metrics_summary']['avg_reward']:.2f}")
        print(f"Success Rate: {info['metrics_summary']['success_rate']:.2%}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train or evaluate PPO agent')
    parser.add_argument('--mode', choices=['train', 'evaluate', 'list'], default='train',
                      help='Mode: train, evaluate, or list models')
    parser.add_argument('--version', type=str, help='Model version to evaluate or continue training from')
    parser.add_argument('--interviews', type=int, default=10,
                      help='Number of complete interviews to train on (default: 10 per day)')
    parser.add_argument('--continue-training', action='store_true',
                      help='Continue training from an existing model version')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("Starting training...")
        if args.continue_training and args.version:
            model, version = train_ppo(total_interviews=args.interviews, continue_version=args.version)
            print(f"Continued training completed. New model version: {version}")
        else:
            model, version = train_ppo(total_interviews=args.interviews)
            print(f"Training completed. Model version: {version}")
    
    elif args.mode == 'evaluate':
        print("Starting evaluation...")
        evaluate_agent(version=args.version)
    
    else:  # list mode
        list_available_models() 