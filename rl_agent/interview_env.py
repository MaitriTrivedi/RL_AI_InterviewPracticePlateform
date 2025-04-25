import numpy as np
import gym
from gym import spaces
from interview_dataset import InterviewQuestionBank, InterviewSimulator
from training_data import TrainingDataHandler

class InterviewEnvironment(gym.Env):
    def __init__(self):
        super().__init__()
        
        # Initialize question bank and simulator
        self.question_bank = InterviewQuestionBank()
        self.simulator = InterviewSimulator(self.question_bank)
        self.training_handler = TrainingDataHandler()
        
        # State space: [current_topic_index, current_difficulty, avg_score, 
        #              time_efficiency, streak, question_history(4 topics)]
        self.observation_space = spaces.Box(
            low=np.array([0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([3, 10, 1, 1, 10, 1, 1, 1, 1], dtype=np.float32)
        )
        
        # Action space: difficulty adjustment (-2 to +2)
        self.action_space = spaces.Box(
            low=np.array([-2], dtype=np.float32),
            high=np.array([2], dtype=np.float32)
        )
        
        self.reset()
        
    def reset(self):
        """Reset environment for new interview"""
        self.current_step = 0
        self.current_question = self.simulator.start_interview()
        self.question_history = {topic: 0.0 for topic in self.question_bank.topics}
        self.current_score = 0.0
        self.current_streak = 0
        self.time_efficiency = 1.0
        
        return self._get_observation()
        
    def _get_observation(self):
        """Convert current state to observation vector"""
        topic_idx = self.question_bank.topics.index(self.current_question['topic'])
        
        return np.array([
            topic_idx / (len(self.question_bank.topics) - 1),  # Normalize topic index
            self.current_question['difficulty'] / 10.0,  # Normalize difficulty
            self.current_score,
            self.time_efficiency,
            self.current_streak / 10.0,  # Normalize streak
            *list(self.question_history.values())  # Topic performance history
        ], dtype=np.float32)
        
    def step(self, action):
        """Execute one step in the environment"""
        # Simulate user attempt with current skill level
        skill_level = 5.0 + self.current_streak * 0.5  # Base skill + streak bonus
        simulated_answer = self.training_handler.simulate_user_answer(
            self.current_question, 
            skill_level
        )
        
        # Evaluate answer
        evaluation = self.training_handler.evaluate_answer(
            self.current_question,
            simulated_answer
        )
        
        # Update metrics
        self.current_score = evaluation['overall_score']
        self.time_efficiency = evaluation['time_efficiency']
        self.question_history[self.current_question['topic']] = self.current_score
        
        # Update streak
        if self.current_score > 0.6:
            self.current_streak = min(10, self.current_streak + 1)
        else:
            self.current_streak = 0
            
        # Get next question
        prev_topic = self.current_question['topic']
        self.current_question = self.simulator.next_question(
            self.current_score,
            prev_topic
        )
        
        # Calculate reward
        reward = self._calculate_reward(evaluation)
        
        # Check if interview is done
        self.current_step += 1
        done = self.current_step >= self.simulator.questions_per_interview
        
        return self._get_observation(), reward, done, {
            'question': self.current_question,
            'score': self.current_score,
            'time_efficiency': self.time_efficiency
        }
        
    def _calculate_reward(self, evaluation):
        """Calculate reward based on performance"""
        reward = 0
        
        # Base reward from answer quality
        reward += evaluation['overall_score']
        
        # Streak bonus
        reward += 0.1 * self.current_streak
        
        # Difficulty bonus (reward for handling harder questions)
        if self.current_score > 0.6 and self.current_question['difficulty'] > 5:
            reward += 0.2 * (self.current_question['difficulty'] - 5) / 5
            
        # Time efficiency bonus
        if evaluation['time_efficiency'] > 0.8:
            reward += 0.2
            
        # Topic coverage bonus (reward for good performance across topics)
        avg_topic_score = np.mean(list(self.question_history.values()))
        min_topic_score = min(self.question_history.values())
        reward += 0.1 * avg_topic_score + 0.1 * min_topic_score
        
        return reward 