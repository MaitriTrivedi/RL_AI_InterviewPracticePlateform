import os
import json
import numpy as np
from datetime import datetime
from .interview_agent import InterviewAgent
from .question_bank import QuestionBank  # You'll need to implement this
import logging

class InterviewSession:
    def __init__(self, model_dir="models"):
        """Initialize interview session."""
        self.model_dir = model_dir
        self.setup_logging()
        
        # Load the latest model
        self.model_version = self.get_latest_model_version()
        self.agent = InterviewAgent(state_dim=9, model_version=self.model_version)
        
        # Initialize question bank
        self.question_bank = QuestionBank()
        
        # Initialize interview state
        self.current_topic = None
        self.current_subtopic = None
        self.current_difficulty = 5.0  # Start with medium difficulty
        self.interview_history = []
        
        logging.info(f"Interview session initialized with model version: {self.model_version}")
    
    def setup_logging(self):
        """Setup logging for the interview session."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join("interview_logs", timestamp)
        os.makedirs(log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, 'interview.log')),
                logging.StreamHandler()
            ]
        )
        self.log_dir = log_dir
    
    def get_latest_model_version(self):
        """Get the latest model version from the models directory."""
        if not os.path.exists(self.model_dir):
            raise ValueError(f"Model directory {self.model_dir} does not exist!")
        
        model_versions = [d for d in os.listdir(self.model_dir) 
                        if os.path.isdir(os.path.join(self.model_dir, d)) 
                        and d.startswith('model_v1_')]
        
        if not model_versions:
            raise ValueError("No trained models found!")
        
        # Sort by timestamp and reward
        latest_version = sorted(model_versions, reverse=True)[0]
        return latest_version
    
    def start_interview(self):
        """Start a new interview session."""
        logging.info("Starting new interview session")
        self.interview_history = []
        self.agent.reset_interview_state()
        
        # Select first topic
        self.current_topic = self.agent.select_next_topic()
        self.current_subtopic = self.agent.select_subtopic(self.current_topic)
        
        return self.get_next_question()
    
    def get_next_question(self):
        """Get the next question based on current state."""
        # Get difficulty adjustment from agent
        action_info = self.agent.get_next_question(self.current_topic)
        self.current_difficulty = action_info['difficulty']
        
        # Get question from question bank
        question = self.question_bank.get_question(
            topic=self.current_topic,
            subtopic=self.current_subtopic,
            difficulty=self.current_difficulty
        )
        
        # Log question details
        logging.info(f"Selected question - Topic: {self.current_topic}, "
                    f"Subtopic: {self.current_subtopic}, "
                    f"Difficulty: {self.current_difficulty:.1f}")
        
        return {
            'question': question['text'],
            'topic': self.current_topic,
            'subtopic': self.current_subtopic,
            'difficulty': self.current_difficulty,
            'expected_time': question.get('expected_time', 5),
            'hints': question.get('hints', []),
            'example': question.get('example', None)
        }
    
    def submit_answer(self, performance_score, time_taken):
        """Submit answer and get feedback."""
        # Update agent with performance
        self.agent.update_performance(
            topic=self.current_topic,
            subtopic=self.current_subtopic,
            performance_score=performance_score,
            time_taken=time_taken
        )
        
        # Record in interview history
        self.interview_history.append({
            'topic': self.current_topic,
            'subtopic': self.current_subtopic,
            'difficulty': self.current_difficulty,
            'performance': performance_score,
            'time_taken': time_taken
        })
        
        # Log performance
        logging.info(f"Answer submitted - Performance: {performance_score:.2f}, "
                    f"Time taken: {time_taken:.1f}s")
        
        # Check if interview should continue
        if len(self.interview_history) >= 10:  # Limit to 10 questions
            return self.end_interview()
        
        # Select next topic and subtopic
        excluded_topics = [self.current_topic]  # Avoid immediate repetition
        self.current_topic = self.agent.select_next_topic(exclude_topics=excluded_topics)
        self.current_subtopic = self.agent.select_subtopic(self.current_topic)
        
        return self.get_next_question()
    
    def end_interview(self):
        """End the interview and provide summary."""
        stats = self.agent.get_interview_stats()
        
        summary = {
            'total_questions': len(self.interview_history),
            'average_score': float(stats['average_score']),
            'topic_performances': {k: float(v) for k, v in stats['topic_performances'].items()},
            'time_efficiency': float(stats['time_efficiency']),
            'final_difficulty': float(stats['difficulty_level'])
        }
        
        # Save interview results
        results_file = os.path.join(self.log_dir, 'interview_results.json')
        with open(results_file, 'w') as f:
            json.dump({
                'summary': summary,
                'history': self.interview_history,
                'model_version': self.model_version
            }, f, indent=2)
        
        logging.info("Interview completed")
        logging.info(f"Average score: {summary['average_score']:.2f}")
        logging.info(f"Results saved to: {results_file}")
        
        return {
            'status': 'completed',
            'summary': summary,
            'message': "Interview completed. Thank you for your participation!"
        }

def main():
    """Run an interactive interview session."""
    try:
        # Initialize interview session
        session = InterviewSession()
        
        print("\nWelcome to the AI Technical Interview!")
        print("Each answer will be scored from 0 to 1, where:")
        print("  0.0-0.3: Incorrect or very weak answer")
        print("  0.4-0.6: Partially correct answer")
        print("  0.7-0.8: Good answer")
        print("  0.9-1.0: Excellent answer\n")
        
        # Start interview
        question_data = session.start_interview()
        
        while True:
            # Display question
            print("\n" + "="*50)
            print(f"Topic: {question_data['topic']}")
            print(f"Subtopic: {question_data['subtopic']}")
            print(f"Difficulty: {question_data['difficulty']:.1f}/10")
            print(f"Expected time: {question_data['expected_time']} minutes")
            print("-"*50)
            print("Question:", question_data['question'])
            
            if question_data.get('example'):
                print("\nExample:", question_data['example'])
            
            if question_data.get('hints'):
                print("\nHints available! Type 'hint' to see them.")
            
            # Get answer and score
            input("\nPress Enter when you're ready to answer...")
            start_time = datetime.now()
            
            input("\nPress Enter when you've finished answering...")
            time_taken = (datetime.now() - start_time).total_seconds() / 60  # Convert to minutes
            
            # Get performance score
            while True:
                try:
                    score = float(input("\nEnter your performance score (0-1): "))
                    if 0 <= score <= 1:
                        break
                    print("Score must be between 0 and 1")
                except ValueError:
                    print("Please enter a valid number")
            
            # Submit answer and get next question
            question_data = session.submit_answer(score, time_taken)
            
            # Check if interview is complete
            if question_data.get('status') == 'completed':
                print("\n" + "="*50)
                print("Interview Complete!")
                print(f"Average Score: {question_data['summary']['average_score']:.2f}")
                print("\nTopic Performances:")
                for topic, score in question_data['summary']['topic_performances'].items():
                    print(f"  {topic}: {score:.2f}")
                print(f"\nTime Efficiency: {question_data['summary']['time_efficiency']:.2f}")
                print("="*50)
                break
    
    except Exception as e:
        logging.error(f"Error during interview: {e}")
        print("\nAn error occurred during the interview. Please check the logs for details.")

if __name__ == "__main__":
    main() 