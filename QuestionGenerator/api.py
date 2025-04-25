#!/usr/bin/env python3
"""
api.py - Programmatic API for Question Generation and Evaluation

This module provides a clean API interface for the QuestionGenerator
to be used by the RL agent without the interactive command-line interface.
"""

import os
import json
import sys
from typing import Dict, List, Any, Optional, Union

# Check for Google Generative AI package
try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
except ImportError:
    print("ERROR: google-generativeai package not found.")
    print("Please install it with: pip install google-generativeai")
    sys.exit(1)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()  # Load variables from .env file

class QuestionGenerator:
    """API for generating interview questions and evaluating answers."""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-1.5-flash-latest"):
        """
        Initialize the QuestionGenerator API.
        
        Args:
            api_key: Optional Gemini API key (if not provided, uses GEMINI_API_KEY env var)
            model_name: The Gemini model to use
        """
        # Get API key from argument or environment
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set and no API key provided.")
        
        # Configure the model
        self.model_name = model_name
        
        # Safety settings
        self.safety_settings = [
            {
                "category": HarmCategory.HARM_CATEGORY_HARASSMENT,
                "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
            },
            {
                "category": HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
            },
            {
                "category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
            },
            {
                "category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
            }
        ]
        
        # Initialize the model
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(
            model_name=self.model_name, 
            safety_settings=self.safety_settings
        )
        
        # Chat sessions for different topics
        self.chat_sessions = {}
    
    def _get_or_create_chat_session(self, topic: str) -> Any:
        """
        Get an existing chat session for a topic or create a new one.
        
        Args:
            topic: The topic for the Q&A session
            
        Returns:
            Chat session object
        """
        if topic not in self.chat_sessions:
            # Set up initial chat history
            chat = self.model.start_chat(history=[
                {
                    "role": "user",
                    "parts": [f"We are starting an interactive Q&A session about '{topic}'. I will ask you to generate technical questions around a specific difficulty score (0-10). I will then choose one, provide an answer, and you will evaluate my answer based on technical accuracy and key points, providing a score out of 10 (allowing decimals). Please remember the topic and the questions already asked throughout our conversation to avoid repetition. Respond strictly in the JSON formats requested."]
                },
                {
                    "role": "model",
                    "parts": [f"Understood. I am ready to begin the Q&A session on '{topic}'. I will generate unique questions targeting the difficulty score you provide, wait for your answer to a chosen question, and then evaluate it based on technical accuracy and key points, providing a score out of 10 (including decimals) in the requested JSON format."]
                }
            ])
            self.chat_sessions[topic] = chat
        
        return self.chat_sessions[topic]
    
    def _clean_json_response(self, text: str) -> str:
        """
        Clean JSON response from the model.
        
        Args:
            text: Raw text response from the model
            
        Returns:
            Cleaned JSON text
        """
        if not text or not isinstance(text, str):
            return text
        
        # Extract JSON from markdown code blocks
        import re
        match = re.search(r'```json\s*([\s\S]*?)\s*```', text)
        if match and match.group(1):
            return match.group(1).strip()
        
        # Check if it's already a JSON structure
        cleaned_text = text.strip()
        if (cleaned_text.startswith('[') and cleaned_text.endswith(']')) or \
           (cleaned_text.startswith('{') and cleaned_text.endswith('}')):
            return cleaned_text
        
        return cleaned_text
    
    def _parse_json_response(self, response_text: str, expected_type: type) -> Union[List, Dict, None]:
        """
        Parse JSON response from the model.
        
        Args:
            response_text: Raw text response from the model
            expected_type: Expected type (list or dict)
            
        Returns:
            Parsed JSON object or None if parsing fails
        """
        if response_text is None:
            return None
        
        cleaned_text = self._clean_json_response(response_text)
        try:
            data = json.loads(cleaned_text)
            
            # Validate the expected type
            if expected_type == list and isinstance(data, list):
                return data
            elif expected_type == dict and isinstance(data, dict) and not isinstance(data, list):
                return data
            else:
                return None
        
        except (json.JSONDecodeError, Exception):
            return None
    
    def generate_questions(
        self, 
        topic: str, 
        difficulty: float,
        num_questions: int = 5,
        custom_prompt: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate interview questions for a given topic and difficulty.
        
        Args:
            topic: Topic for questions (e.g., "Python", "Machine Learning")
            difficulty: Target difficulty level (0-10)
            num_questions: Number of questions to generate
            custom_prompt: Optional custom prompt for the LLM
            
        Returns:
            List of question dictionaries with 'question', 'difficulty', and 'questionNo' keys
        """
        chat_session = self._get_or_create_chat_session(topic)
        
        # Format difficulty for consistency
        difficulty_str = f"{difficulty:.1f}"
        
        # Create prompt
        if custom_prompt:
            prompt = custom_prompt
        else:
            prompt = f"""
            Generate exactly {num_questions} new, unique technical questions about the topic '{topic}'.
            Target difficulty score: Aim for questions around {difficulty_str} on a scale of 0.0 (very easy) to 10.0 (expert).

            For each generated question:
            1.  Provide the specific question text.
            2.  Estimate *its* specific difficulty score on the 0.0 to 10.0 scale, formatted to one decimal place.
            3.  Assign a sequential question number for this batch, starting from 1.

            IMPORTANT: Ensure these questions are different from any previously asked in this chat session.

            Format the output STRICTLY as a JSON list of objects, like this example:
            [
                {{"question": "What is RAM?", "difficulty": "1.5", "questionNo": "1"}},
                {{"question": "Explain memory virtualization.", "difficulty": "7.8", "questionNo": "2"}},
                {{"question": "...", "difficulty": "...", "questionNo": "3"}}
            ]
            The output MUST be ONLY the valid JSON list and nothing else.
            """
        
        # Send the prompt
        try:
            response = chat_session.send_message(prompt)
            response_text = response.text
            questions_data_raw = self._parse_json_response(response_text, list)
            
            if not questions_data_raw:
                return []
            
            # Validate and filter questions
            valid_questions = []
            for q in questions_data_raw:
                if q and isinstance(q, dict) and 'question' in q and 'difficulty' in q and 'questionNo' in q:
                    try:
                        q_text = str(q['question']).strip()
                        q_diff_str = str(q['difficulty'])
                        q_num_str = str(q['questionNo'])
                        
                        # Validate format
                        q_diff_num = float(q_diff_str)
                        q_num_int = int(q_num_str)
                        
                        if not q_text or q_diff_num < 0 or q_diff_num > 10:
                            continue
                        
                        # Store with formatted difficulty
                        valid_questions.append({
                            'question': q_text,
                            'difficulty': f"{q_diff_num:.1f}",
                            'questionNo': q_num_str
                        })
                    except (ValueError, TypeError):
                        continue
            
            return valid_questions
        
        except Exception as e:
            print(f"Error generating questions: {str(e)}")
            return []
    
    def evaluate_answer(
        self, 
        topic: str,
        question: str,
        answer: str
    ) -> float:
        """
        Evaluate an answer to a question.
        
        Args:
            topic: The topic of the question
            question: The question text
            answer: The user's answer text
            
        Returns:
            Score for the answer (0-10), defaults to 5.0 if evaluation fails
        """
        chat_session = self._get_or_create_chat_session(topic)
        
        # Create prompt
        prompt = f"""
        You are an expert technical evaluator for the topic '{topic}'.

        The specific question was:
        "{question}"

        The provided answer is:
        "{answer}"

        **Evaluation Task:**
        1.  Identify key technical elements based on the question.
        2.  Analyze the answer for accuracy and use of key technical elements.
        3.  Score based on technical accuracy between 0.0 and 10.0, allowing decimal values.
            *   High (8.0-10.0): Strong understanding, accurate terms/concepts.
            *   Medium (5.0-7.9): Partial understanding, some correct elements but gaps.
            *   Low (0.0-4.9): Misses main technical points, incorrect/irrelevant.

        **Output Format:**
        Respond ONLY with a valid JSON object containing the score, like this:
        {"score": "8.5"}
        Do not include any other text, just the JSON object.
        """
        
        try:
            # Send the prompt
            response = chat_session.send_message(prompt)
            response_text = response.text
            score_data = self._parse_json_response(response_text, dict)
            
            # Extract and validate the score
            if score_data and 'score' in score_data:
                try:
                    return float(score_data['score'])
                except ValueError:
                    return 5.0  # Default if score isn't a valid float
            else:
                return 5.0  # Default if response is missing or malformed
        
        except Exception as e:
            print(f"Error evaluating answer: {str(e)}")
            return 5.0  # Default score on error


# Example usage
if __name__ == "__main__":
    # Example of using the API programmatically
    
    try:
        generator = QuestionGenerator()
        
        topic = "Python Programming"
        difficulty = 7.5
        
        print(f"Generating questions about {topic} at difficulty {difficulty}...")
        questions = generator.generate_questions(topic, difficulty, num_questions=3)
        
        if questions:
            print("\nGenerated Questions:")
            for q in questions:
                print(f"\n{q['questionNo']}. (Difficulty: {q['difficulty']})")
                print(f"   Q: {q['question']}")
            
            # Example evaluation
            sample_question = questions[0]['question']
            sample_answer = "This is a sample answer that would be provided by a user."
            
            print("\nEvaluating sample answer...")
            score = generator.evaluate_answer(topic, sample_question, sample_answer)
            print(f"Score: {score:.1f}/10.0")
        else:
            print("Failed to generate questions.")
    
    except Exception as e:
        print(f"Error in example usage: {str(e)}") 