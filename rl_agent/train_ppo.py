import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from interview_env import InterviewEnvironment

def make_env():
    """Create and wrap the interview environment."""
    env = InterviewEnvironment()
    return env

def train_ppo(total_timesteps: int = 100000):
    """Train the PPO agent."""
    # Create vectorized environment
    env = DummyVecEnv([make_env])
    
    # Initialize PPO agent
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1
    )
    
    # Train the agent
    model.learn(total_timesteps=total_timesteps)
    
    # Save the trained model
    model.save("interview_ppo_model")
    
def evaluate_agent(n_episodes: int = 10):
    """Evaluate the trained PPO agent."""
    # Load trained model
    model = PPO.load("interview_ppo_model")
    env = InterviewEnvironment()
    
    total_reward = 0
    success_rate = 0
    
    for episode in range(n_episodes):
        obs = env.reset()
        episode_reward = 0
        episode_success = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            if info["completed"]:
                episode_success += 1
                
        avg_success = episode_success / env.max_steps
        print(f"Episode {episode + 1}")
        print(f"Total Reward: {episode_reward:.2f}")
        print(f"Success Rate: {avg_success:.2%}")
        print("---")
        
        total_reward += episode_reward
        success_rate += avg_success
    
    print("\nOverall Performance:")
    print(f"Average Reward: {total_reward / n_episodes:.2f}")
    print(f"Average Success Rate: {success_rate / n_episodes:.2%}")

if __name__ == "__main__":
    # Train the agent
    train_ppo()
    
    # Evaluate the trained agent
    evaluate_agent() 