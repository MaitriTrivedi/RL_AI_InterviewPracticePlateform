"""
environment.py - RL Environment for Interview Question Selection

This module defines the environment for the reinforcement learning agent.
It interfaces with the ResumeExtractor and QuestionGenerator to create
a complete interview practice environment.
"""

import numpy as np
import json
import sys
import os
from typing import Dict, List, Tuple, Any, Optional

# Add parent directory to path to import from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class InterviewEnvironment:
    """
    Environment for RL-based interview question selection.
    
    This class handles:
    1. Interfacing with the resume extractor
    2. Communicating with the question generator
    3. Tracking interview state
    4. Providing rewards based on interview performance
    """
    
    def __init__(
        self,
        resume_data: Dict[str, Any],
        difficulty_range: Tuple[float, float] = (1.0, 10.0),
        max_questions: int = 10,
        reward_weights: Dict[str, float] = None
    ):
        """
        Initialize the interview environment.
        
        Args:
            resume_data: Parsed resume data from ResumeExtractor
            difficulty_range: Min and max difficulty for questions
            max_questions: Maximum number of questions in an interview
            reward_weights: Weights for different components in the reward function
        """
        # Store resume data
        self.resume_data = resume_data
        
        # Environment parameters
        self.difficulty_range = difficulty_range
        self.max_questions = max_questions
        
        # Default reward weights
        self.reward_weights = reward_weights or {
            'score': 1.0,                # Weight for the answer score
            'difficulty': 0.5,           # Weight for question difficulty
            'relevance': 0.8,            # Weight for relevance to resume
            'diversity': 0.3,            # Weight for question diversity
            'challenge': 0.6             # Weight for making questions challenging but not impossible
        }
        
        # State tracking
        self.current_question_idx = 0
        self.question_history = []
        self.answer_history = []
        self.score_history = []
        self.difficulty_history = []
        
        # Features extracted from resume for state representation
        self.resume_features = self._extract_resume_features()
        
        # Current state representation (initialized in reset())
        self.current_state = None
        
    def _extract_resume_features(self) -> Dict[str, Any]:
        """
        Extract key features from resume for state representation.
        
        Returns:
            Dictionary of features extracted from resume
        """
        features = {}
        
        # Extract education features
        education = self.resume_data.get('education', [])
        features['education_level'] = self._determine_education_level(education)
        features['education_fields'] = self._extract_fields_of_study(education)
        
        # Extract experience features
        work_experience = self.resume_data.get('work_experience', [])
        features['years_experience'] = self._calculate_years_experience(work_experience)
        features['job_roles'] = self._extract_job_roles(work_experience)
        
        # Extract skills and technologies
        projects = self.resume_data.get('projects', [])
        features['technologies'] = self._extract_technologies(projects, work_experience)
        
        return features
    
    def _determine_education_level(self, education: List[Dict[str, str]]) -> int:
        """Determine highest education level from 1-5 (1=High School, 5=PhD)"""
        if not education:
            return 1
        
        levels = {
            'high school': 1,
            'associate': 2,
            'bachelor': 3,
            'master': 4,
            'phd': 5,
            'doctorate': 5
        }
        
        highest_level = 1
        for edu in education:
            degree = edu.get('degree', '').lower()
            for key, value in levels.items():
                if key in degree and value > highest_level:
                    highest_level = value
        
        return highest_level
    
    def _extract_fields_of_study(self, education: List[Dict[str, str]]) -> List[str]:
        """Extract fields of study from education history"""
        fields = []
        for edu in education:
            degree = edu.get('degree', '')
            # Simple extraction - this could be improved with NLP
            if 'computer science' in degree.lower():
                fields.append('computer science')
            elif 'engineering' in degree.lower():
                fields.append('engineering')
            # Add more field extraction logic as needed
        
        return list(set(fields))  # Remove duplicates
    
    def _calculate_years_experience(self, work_experience: List[Dict[str, str]]) -> float:
        """Calculate total years of work experience"""
        total_years = 0.0
        for exp in work_experience:
            year_info = exp.get('year', '')
            # Parse year ranges like "2018-2020" or "2018-Present"
            if '-' in year_info:
                try:
                    start, end = year_info.split('-')
                    start = int(start.strip())
                    end = 2023 if end.strip().lower() == 'present' else int(end.strip())
                    total_years += (end - start)
                except (ValueError, TypeError):
                    # If parsing fails, estimate 1 year per position
                    total_years += 1
            else:
                # If no range is provided, estimate 1 year per position
                total_years += 1
        
        return total_years
    
    def _extract_job_roles(self, work_experience: List[Dict[str, str]]) -> List[str]:
        """Extract job roles from work experience"""
        roles = []
        # This is a simplified approach - could be enhanced with NLP
        common_roles = [
            'software engineer', 'developer', 'manager', 'analyst',
            'data scientist', 'designer', 'architect', 'administrator'
        ]
        
        for exp in work_experience:
            description = exp.get('description', '').lower()
            for role in common_roles:
                if role in description and role not in roles:
                    roles.append(role)
        
        return roles
    
    def _extract_technologies(self, projects: List[Dict[str, str]], 
                             work_experience: List[Dict[str, str]]) -> List[str]:
        """Extract technologies and skills from projects and work experience"""
        technologies = []
        
        # Extract from projects
        for project in projects:
            if 'technologies' in project and isinstance(project['technologies'], list):
                technologies.extend(project['technologies'])
            
            # Also try to find technologies in descriptions
            desc = project.get('description', '').lower()
            technologies.extend(self._find_tech_in_text(desc))
        
        # Extract from work experience descriptions
        for exp in work_experience:
            desc = exp.get('description', '').lower()
            technologies.extend(self._find_tech_in_text(desc))
        
        # Remove duplicates and return
        return list(set(technologies))
    
    def _find_tech_in_text(self, text: str) -> List[str]:
        """Find technology keywords in text"""
        # Simple keyword matching - could be enhanced with NLP
        tech_keywords = [
            'python', 'java', 'javascript', 'js', 'react', 'angular', 'vue',
            'node', 'express', 'django', 'flask', 'spring', 'ruby', 'rails',
            'php', 'laravel', 'html', 'css', 'sql', 'nosql', 'mongodb',
            'postgres', 'mysql', 'oracle', 'aws', 'azure', 'gcp', 'docker',
            'kubernetes', 'k8s', 'git', 'ci/cd', 'jenkins', 'terraform',
            'machine learning', 'ml', 'ai', 'data science', 'tensorflow',
            'pytorch', 'pandas', 'numpy', 'scipy', 'r', 'tableau', 'power bi'
        ]
        
        found_techs = []
        for tech in tech_keywords:
            if tech in text:
                found_techs.append(tech)
        
        return found_techs
    
    def reset(self) -> np.ndarray:
        """
        Reset the environment to start a new interview.
        
        Returns:
            Initial state representation as numpy array
        """
        # Reset state tracking
        self.current_question_idx = 0
        self.question_history = []
        self.answer_history = []
        self.score_history = []
        self.difficulty_history = []
        
        # Create initial state representation
        self.current_state = self._create_state_representation()
        
        return self.current_state
    
    def _create_state_representation(self) -> np.ndarray:
        """
        Create a state representation for the RL agent.
        
        The state includes:
        - Information from the resume
        - Question history
        - Answer performance history
        
        Returns:
            Numpy array representing the current state
        """
        # Features from resume (normalized)
        education_level = self.resume_features['education_level'] / 5.0  # Normalize to [0,1]
        years_experience = min(self.resume_features['years_experience'] / 10.0, 1.0)  # Cap at 10 years
        num_technologies = min(len(self.resume_features['technologies']) / 20.0, 1.0)  # Cap at 20 technologies
        
        # Features from interview progress
        progress = self.current_question_idx / self.max_questions
        
        # Features from previous questions/answers
        avg_score = np.mean(self.score_history) / 10.0 if self.score_history else 0.5
        avg_difficulty = np.mean(self.difficulty_history) / 10.0 if self.difficulty_history else 0.5
        std_score = np.std(self.score_history) / 10.0 if len(self.score_history) > 1 else 0.1
        
        # Combine features into state vector
        state = np.array([
            education_level,
            years_experience,
            num_technologies,
            progress,
            avg_score,
            avg_difficulty,
            std_score
        ])
        
        return state
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take a step in the environment by selecting and asking a question.
        
        Args:
            action: Action from the agent (e.g., targeted difficulty and topic)
        
        Returns:
            next_state: New state representation
            reward: Reward signal based on answer quality and question selection
            done: Whether the interview is complete
            info: Additional information dictionary
        """
        # Decode action
        target_difficulty, topic_idx = self._decode_action(action)
        
        # Generate question options based on action
        questions = self._generate_questions(target_difficulty, topic_idx)
        
        # Select the best question based on our criteria
        selected_question = self._select_best_question(questions)
        
        # Ask the question and get answer and score
        answer, score = self._ask_question(selected_question)
        
        # Update history
        self.question_history.append(selected_question)
        self.answer_history.append(answer)
        self.score_history.append(score)
        self.difficulty_history.append(float(selected_question['difficulty']))
        self.current_question_idx += 1
        
        # Calculate reward
        reward = self._calculate_reward(selected_question, score)
        
        # Update state
        self.current_state = self._create_state_representation()
        
        # Check if interview is complete
        done = self.current_question_idx >= self.max_questions
        
        # Additional info
        info = {
            'question': selected_question['question'],
            'difficulty': selected_question['difficulty'],
            'score': score,
            'questions_asked': self.current_question_idx
        }
        
        return self.current_state, reward, done, info
    
    def _decode_action(self, action: np.ndarray) -> Tuple[float, int]:
        """
        Decode action vector from the agent.
        
        Args:
            action: Action vector from the agent
            
        Returns:
            target_difficulty: Target difficulty level [1-10]
            topic_idx: Index representing preferred topic
        """
        # Example implementation - modify based on your action space design
        target_difficulty = self.difficulty_range[0] + action[0] * (
            self.difficulty_range[1] - self.difficulty_range[0]
        )
        
        # Simple topic selection (can be expanded)
        topic_idx = int(action[1] * 10) if len(action) > 1 else 0
        
        return target_difficulty, topic_idx
    
    def _generate_questions(self, target_difficulty: float, topic_idx: int) -> List[Dict[str, Any]]:
        """
        Generate candidate questions based on target difficulty and topic.
        
        In a real implementation, this would call the QuestionGenerator service.
        For now, we'll stub this with a placeholder implementation.
        
        Args:
            target_difficulty: Target difficulty level [1-10]
            topic_idx: Index representing preferred topic
            
        Returns:
            List of question dictionaries
        """
        # This is a placeholder - in production, call the actual QuestionGenerator
        # Implementation will depend on how we integrate with the JS question generator
        
        # Placeholder implementation
        questions = []
        # In actual implementation, this would call the question generator API
        
        # For now, we'll return a placeholder list
        questions = [
            {
                'question': f"Placeholder question at difficulty {target_difficulty:.1f}",
                'difficulty': f"{target_difficulty:.1f}",
                'questionNo': "1"
            }
        ]
        
        return questions
    
    def _select_best_question(self, questions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Select the best question from candidates based on relevance and diversity.
        
        Args:
            questions: List of candidate questions
            
        Returns:
            Selected question dictionary
        """
        # For now, just return the first question
        # In a complete implementation, we would score questions by relevance to resume,
        # diversity compared to previous questions, etc.
        return questions[0]
    
    def _ask_question(self, question: Dict[str, Any]) -> Tuple[str, float]:
        """
        Ask the selected question and get the answer and score.
        
        In a real implementation, this would interface with the user interface.
        For now, we'll stub this with a placeholder.
        
        Args:
            question: The selected question dictionary
            
        Returns:
            answer: The user's answer text
            score: The score for the answer [0-10]
        """
        # Placeholder implementation
        # In a real system, this would:
        # 1. Display the question to the user
        # 2. Collect their answer
        # 3. Send both to the evaluation service
        # 4. Return the answer and score
        
        # For testing only:
        answer = "Placeholder answer"
        score = 7.5  # Placeholder score
        
        return answer, score
    
    def _calculate_reward(self, question: Dict[str, Any], score: float) -> float:
        """
        Calculate reward based on question selection and answer score.
        
        Args:
            question: The question that was asked
            score: The score for the user's answer [0-10]
            
        Returns:
            reward: The calculated reward signal
        """
        # Base reward is the score
        base_reward = score / 10.0  # Normalize to [0,1]
        
        # Question difficulty component
        difficulty = float(question['difficulty'])
        difficulty_factor = 1.0 - abs(difficulty - 5.0) / 5.0  # Reward for appropriate difficulty
        
        # Relevance component (would use actual relevance in full implementation)
        relevance_factor = 0.8  # Placeholder
        
        # Diversity component (would calculate in full implementation)
        diversity_factor = 0.7  # Placeholder
        
        # Calculate weighted reward
        reward = (
            self.reward_weights['score'] * base_reward +
            self.reward_weights['difficulty'] * difficulty_factor +
            self.reward_weights['relevance'] * relevance_factor +
            self.reward_weights['diversity'] * diversity_factor
        )
        
        # Challenge factor - reward more for doing well on difficult questions
        if score > 7.0 and difficulty > 7.0:
            reward *= 1.2  # Bonus for doing well on hard questions
        
        return reward
    
    def get_state_dimension(self) -> int:
        """
        Get the dimension of the state representation.
        
        Returns:
            Dimension of state vector
        """
        return len(self._create_state_representation())
    
    def get_action_dimension(self) -> int:
        """
        Get the dimension of the action space.
        
        Returns:
            Dimension of action vector
        """
        # Current implementation: [difficulty, topic_preference]
        return 2 