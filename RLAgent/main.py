"""
main.py - Main Application Module for RL Interview System

This is the main entry point for the RL-based interview practice platform.
It connects all components and provides a command-line interface.
"""

import os
import sys
import json
import argparse
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

# Add parent directory to path to import from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from RLAgent.environment import InterviewEnvironment
from RLAgent.agent import QLearningAgent
from RLAgent.connector import SystemConnector

def setup_argparse() -> argparse.ArgumentParser:
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(
        description="AI Interview Practice Platform with Reinforcement Learning"
    )
    
    # Resume file path (required)
    parser.add_argument(
        "resume", type=str, help="Path to the resume file (PDF)"
    )
    
    # Mode selection
    parser.add_argument(
        "--mode", type=str, choices=["train", "interview"], default="interview",
        help="Mode: 'train' to train the agent, 'interview' for real interview"
    )
    
    # Training parameters
    parser.add_argument(
        "--episodes", type=int, default=100,
        help="Number of training episodes (for train mode)"
    )
    parser.add_argument(
        "--max-questions", type=int, default=10,
        help="Maximum number of questions per interview"
    )
    
    # Agent parameters
    parser.add_argument(
        "--agent-file", type=str, default="models/agent.pkl",
        help="Path to save/load the agent"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.1,
        help="Learning rate for the agent"
    )
    parser.add_argument(
        "--exploration", type=float, default=1.0,
        help="Initial exploration rate"
    )
    
    # Interview parameters
    parser.add_argument(
        "--topic", type=str, 
        help="Override topic extraction with a specific topic"
    )
    parser.add_argument(
        "--difficulty", type=float, default=5.0,
        help="Target difficulty for questions (0-10)"
    )
    
    return parser

def train_agent(
    resume_data: Dict[str, Any],
    agent_file: str,
    episodes: int,
    max_questions: int,
    learning_rate: float,
    exploration_rate: float
) -> QLearningAgent:
    """
    Train the RL agent to select good questions.
    
    Args:
        resume_data: Parsed resume data
        agent_file: Path to save the trained agent
        episodes: Number of training episodes
        max_questions: Max questions per interview
        learning_rate: Agent learning rate
        exploration_rate: Initial exploration rate
    
    Returns:
        Trained agent
    """
    print("Initializing environment...")
    env = InterviewEnvironment(
        resume_data=resume_data,
        max_questions=max_questions
    )
    
    print("Initializing agent...")
    agent = QLearningAgent(
        state_dim=env.get_state_dimension(),
        action_dim=env.get_action_dimension(),
        learning_rate=learning_rate,
        exploration_rate=exploration_rate
    )
    
    # Try to load existing agent if available
    if os.path.exists(agent_file):
        print(f"Loading existing agent from {agent_file}...")
        success = agent.load(agent_file)
        if not success:
            print("Failed to load agent, starting with a new one.")
    
    print(f"Starting training for {episodes} episodes...")
    rewards_history = []
    
    # Start training
    for episode in range(1, episodes + 1):
        # Reset environment for new episode
        state = env.reset()
        done = False
        episode_reward = 0
        questions_asked = 0
        
        print(f"\nEpisode {episode}/{episodes}")
        
        # Episode loop
        while not done:
            # Select action
            action = agent.select_action(state)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Update agent
            agent.update(state, action, reward, next_state, done)
            
            # Update for next iteration
            state = next_state
            episode_reward += reward
            questions_asked += 1
            
            # Print info about this step
            print(f"Question {questions_asked}: Difficulty target: {info['difficulty']}")
            print(f"Reward: {reward:.2f}")
        
        # Episode summary
        rewards_history.append(episode_reward)
        print(f"Episode {episode} finished:")
        print(f"  Questions asked: {questions_asked}")
        print(f"  Total reward: {episode_reward:.2f}")
        print(f"  Average reward per question: {episode_reward/questions_asked:.2f}")
        print(f"  Current exploration rate: {agent.exploration_rate:.3f}")
        
        # Save agent every 10 episodes
        if episode % 10 == 0:
            print(f"Saving agent to {agent_file}...")
            agent.save(agent_file)
            
            # Show training progress
            avg_reward = np.mean(rewards_history[-10:])
            print(f"Average reward over last 10 episodes: {avg_reward:.2f}")
    
    # Final save
    print(f"Training completed. Saving agent to {agent_file}...")
    agent.save(agent_file)
    
    return agent

def run_interview(
    resume_data: Dict[str, Any],
    agent_file: str,
    max_questions: int,
    topic: Optional[str],
    difficulty: float
) -> None:
    """
    Run an actual interview using a trained agent.
    
    Args:
        resume_data: Parsed resume data
        agent_file: Path to the trained agent
        max_questions: Maximum questions to ask
        topic: Optional topic override
        difficulty: Target difficulty level
    """
    # Create system connector for live interview
    system = SystemConnector()
    
    # Try to load agent
    if not os.path.exists(agent_file):
        print(f"No trained agent found at {agent_file}. Starting with default parameters.")
        use_agent = False
    else:
        print(f"Loading trained agent from {agent_file}...")
        env = InterviewEnvironment(
            resume_data=resume_data,
            max_questions=max_questions
        )
        agent = QLearningAgent(
            state_dim=env.get_state_dimension(),
            action_dim=env.get_action_dimension()
        )
        success = agent.load(agent_file)
        use_agent = success
        
        if not success:
            print("Failed to load agent. Starting with default parameters.")
            use_agent = False
    
    # Determine topic
    if topic is None:
        topic = system.generate_topic_from_resume(resume_data)
        print(f"Selected topic based on resume: {topic}")
    else:
        print(f"Using provided topic: {topic}")
    
    # Initialize interview state
    current_difficulty = difficulty
    total_score = 0
    questions_asked = 0
    
    # Initialize environment for state tracking if using agent
    if use_agent:
        state = env.reset()
    
    print("\n" + "="*60)
    print(f"WELCOME TO THE AI INTERVIEW PRACTICE PLATFORM")
    print("="*60)
    print(f"Topic: {topic}")
    print(f"We'll ask up to {max_questions} questions.")
    print("Answer each question as best you can.")
    print("="*60 + "\n")
    
    while questions_asked < max_questions:
        # Determine difficulty using agent or parameter
        if use_agent:
            action = agent.select_action(state, training=False)
            target_difficulty = 1.0 + action[0] * 9.0  # Scale [0,1] to [1,10]
            print(f"[Agent selected difficulty: {target_difficulty:.1f}]")
        else:
            target_difficulty = current_difficulty
        
        # Generate questions
        print(f"\nGenerating questions (difficulty ~{target_difficulty:.1f})...")
        try:
            questions = system.generate_questions(
                resume_data=resume_data,
                topic=topic,
                difficulty=target_difficulty,
                num_questions=3  # Generate a few options
            )
            
            if not questions:
                print("Failed to generate questions. Trying again with default prompt...")
                questions = system.question_connector.generate_questions(
                    topic=topic,
                    difficulty=target_difficulty,
                    num_questions=3
                )
                
                if not questions:
                    print("Still failed to generate questions. Please check the question generator.")
                    break
            
            # If using agent, select the best question, otherwise use the first one
            if use_agent and len(questions) > 1:
                # In production, we would implement a selection method that considers
                # question diversity, relevance to resume, etc.
                selected_question = questions[0]
            else:
                selected_question = questions[0]
            
            # Display question
            questions_asked += 1
            print(f"\nQUESTION {questions_asked} (Difficulty: {selected_question['difficulty']}/10)")
            print(f"{selected_question['question']}\n")
            
            # Get answer from user
            print("Your answer (type on one or multiple lines, end with a line with just '.' or CTRL+D):")
            answer_lines = []
            while True:
                try:
                    line = input()
                    if line == '.':
                        break
                    answer_lines.append(line)
                except EOFError:
                    break
            
            answer = "\n".join(answer_lines)
            
            # Evaluate answer
            print("\nEvaluating your answer...")
            score = system.evaluate_answer(
                question=selected_question['question'],
                answer=answer,
                topic=topic,
                resume_data=resume_data
            )
            
            print(f"Score: {score:.1f}/10.0")
            
            total_score += score
            
            # Update environment and agent state if using agent
            if use_agent:
                # Mock the environment step
                question_data = {
                    'question': selected_question['question'],
                    'difficulty': selected_question['difficulty'],
                    'questionNo': str(questions_asked)
                }
                
                # Update environment state
                env.question_history.append(question_data)
                env.answer_history.append(answer)
                env.score_history.append(score)
                env.difficulty_history.append(float(selected_question['difficulty']))
                env.current_question_idx = questions_asked
                
                # Update state
                next_state = env._create_state_representation()
                
                # Calculate reward
                reward = env._calculate_reward(question_data, score)
                
                # Check if done
                done = questions_asked >= max_questions
                
                # No need to update agent during interview, but update state
                state = next_state
                
                print(f"Reward: {reward:.2f}")
            
            # Adapt difficulty for next question if not using agent
            if not use_agent:
                # Simple adaptive difficulty:
                # If score > 7, increase difficulty
                # If score < 4, decrease difficulty
                if score > 7.0:
                    current_difficulty = min(10.0, current_difficulty + 1.0)
                elif score < 4.0:
                    current_difficulty = max(1.0, current_difficulty - 1.0)
        
        except Exception as e:
            print(f"Error during question generation/evaluation: {e}")
            break
        
        # Pause between questions
        if questions_asked < max_questions:
            print("\nPress Enter for the next question...")
            input()
    
    # Interview summary
    print("\n" + "="*60)
    print("INTERVIEW SUMMARY")
    print("="*60)
    print(f"Questions asked: {questions_asked}")
    print(f"Average score: {total_score/questions_asked:.1f}/10.0")
    print("="*60)
    print("\nThank you for using the AI Interview Practice Platform!")
    print("Keep practicing to improve your skills.")
    print("="*60 + "\n")

def main():
    """Main application entry point."""
    # Parse command line arguments
    parser = setup_argparse()
    args = parser.parse_args()
    
    # Check if resume file exists
    if not os.path.exists(args.resume):
        print(f"Error: Resume file not found at {args.resume}")
        return 1
    
    # Create system connector
    system = SystemConnector()
    
    # Parse resume
    print(f"Parsing resume from {args.resume}...")
    try:
        resume_data = system.parse_resume(args.resume)
        print("Resume parsed successfully.")
    except Exception as e:
        print(f"Error parsing resume: {e}")
        return 1
    
    # Create directory for agent if it doesn't exist
    os.makedirs(os.path.dirname(args.agent_file), exist_ok=True)
    
    # Run in selected mode
    if args.mode == "train":
        print("Running in TRAINING mode.")
        train_agent(
            resume_data=resume_data,
            agent_file=args.agent_file,
            episodes=args.episodes,
            max_questions=args.max_questions,
            learning_rate=args.learning_rate,
            exploration_rate=args.exploration
        )
    else:  # Interview mode
        print("Running in INTERVIEW mode.")
        run_interview(
            resume_data=resume_data,
            agent_file=args.agent_file,
            max_questions=args.max_questions,
            topic=args.topic,
            difficulty=args.difficulty
        )
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 