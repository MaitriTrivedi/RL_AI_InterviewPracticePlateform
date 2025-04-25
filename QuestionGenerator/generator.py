#!/usr/bin/env python3
"""
generator.py - Python version of the Gemini Q&A Evaluator

This module provides question generation and answer evaluation 
functionality using Google's Generative AI (Gemini API).
"""

import os
import sys
import json
import argparse
import readline
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

# --- Configuration ---
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    print("ERROR: GEMINI_API_KEY environment variable not set.")
    print("Please create a .env file with GEMINI_API_KEY=YOUR_KEY or set the environment variable.")
    sys.exit(1)

# Configure the model
MODEL_NAME = "gemini-1.5-flash-latest"  # Or "gemini-pro"

# Configure safety settings (optional)
SAFETY_SETTINGS = [
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
try:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel(model_name=MODEL_NAME, safety_settings=SAFETY_SETTINGS)
    print(f"Using model: {MODEL_NAME}")
except Exception as e:
    print(f"Error initializing the Generative Model '{MODEL_NAME}': {str(e)}")
    sys.exit(1)

# --- Helper Functions ---

def ask_question(query: str) -> str:
    """Ask a question and get user input."""
    return input(query)

def clean_json_response(text: str) -> str:
    """Clean JSON response from the model."""
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

def call_gemini_chat(chat_session, prompt: str) -> Optional[str]:
    """Send a message to the Gemini model and get the response."""
    try:
        response = chat_session.send_message(prompt)
        response_text = response.text
        
        # Check for safety blocks
        if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
            if hasattr(response.prompt_feedback, 'block_reason') and response.prompt_feedback.block_reason:
                print("\n--- ERROR: Prompt Blocked by API ---")
                print(f"Reason: {response.prompt_feedback.block_reason}")
                # Log ratings if available
                if hasattr(response.prompt_feedback, 'safety_ratings') and response.prompt_feedback.safety_ratings:
                    for rating in response.prompt_feedback.safety_ratings:
                        print(f"  {rating.category}: {rating.probability}")
                return None
        
        # Check for response blocks
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'finish_reason') and candidate.finish_reason and candidate.finish_reason != 'STOP':
                print("\n--- Warning: Response Terminated Abnormally ---")
                print(f"Reason: {candidate.finish_reason}")
                # Log ratings if available
                if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings:
                    for rating in candidate.safety_ratings:
                        print(f"  {rating.category}: {rating.probability}")
        
        return response_text
    
    except Exception as e:
        print("\n--- An API Error Occurred During Chat ---")
        print(f"Error details: {str(e)}")
        return None

def parse_json_response(response_text: str, expected_type: type) -> Union[List, Dict, None]:
    """Parse JSON response from the model."""
    if response_text is None:
        print("Debug: Cannot parse JSON, input response_text is null.")
        return None
    
    cleaned_text = clean_json_response(response_text)
    try:
        data = json.loads(cleaned_text)
        
        # Validate the expected type
        if expected_type == list and isinstance(data, list):
            return data
        elif expected_type == dict and isinstance(data, dict) and not isinstance(data, list):
            return data
        else:
            print("\n--- Error: Unexpected JSON structure ---")
            print(f"Expected type: {expected_type.__name__}, Got type: {type(data).__name__}")
            print(f"Received data structure: {data}")
            print(f"Original Response Text (before cleaning):\n{response_text}")
            return None
    
    except json.JSONDecodeError as e:
        print("\n--- Error: Failed to decode JSON (SyntaxError) ---")
        print(f"Error message: {str(e)}")
        print(f"Text attempted to parse (after cleaning):\n{cleaned_text}")
        print(f"Original Response Text (before cleaning):\n{response_text}")
        return None
    
    except Exception as e:
        print("\n--- An unexpected error occurred during JSON parsing ---")
        print(f"Error details: {str(e)}")
        return None

# --- Main Application Logic ---

def run():
    """Main application function for the Q&A evaluator."""
    print("Welcome to the Gemini Q&A Evaluator!")
    print("=" * 40)
    
    topic = ""
    chat_session = None
    
    # --- 1. Initialize Chat / Get Topic ---
    while not topic:
        topic = ask_question("Enter the topic for the Q&A session (e.g., Docker, Java, Kubernetes): ").strip()
        if not topic:
            print("Topic cannot be empty.")
    
    print(f"\nInitializing chat session for topic: '{topic}'...")
    try:
        # Set up initial chat history
        chat = model.start_chat(history=[
            {
                "role": "user",
                "parts": [f"We are starting an interactive Q&A session about '{topic}'. I will ask you to generate technical questions around a specific difficulty score (0-10). I will then choose one, provide an answer, and you will evaluate my answer based on technical accuracy and key points, providing a score out of 10 (allowing decimals). Please remember the topic and the questions already asked throughout our conversation to avoid repetition. Respond strictly in the JSON formats requested."]
            },
            {
                "role": "model",
                "parts": [f"Understood. I am ready to begin the Q&A session on '{topic}'. I will generate unique questions targeting the difficulty score you provide, wait for your answer to a chosen question, and then evaluate it based on technical accuracy and key points, providing a score out of 10 (including decimals) in the requested JSON format."]
            }
        ])
        chat_session = chat
        print(f"Chat session started for '{topic}'.")
    except Exception as e:
        print(f"Error starting chat session: {str(e)}")
        sys.exit(1)
    
    # --- Main Loop ---
    while True:
        # --- 2. Get Difficulty Score ---
        difficulty_score = None
        while difficulty_score is None:
            difficulty_input = ask_question(f"\nEnter desired difficulty score for '{topic}' questions (0.0 - 10.0): ").strip()
            try:
                score = float(difficulty_input)
                if 0 <= score <= 10:
                    difficulty_score = score
                else:
                    print("Invalid input. Please enter a number between 0.0 and 10.0.")
            except ValueError:
                print("Invalid input. Please enter a number between 0.0 and 10.0.")
        
        # Format for consistency
        difficulty_score_str = f"{difficulty_score:.1f}"
        
        # --- 3. Generate Questions ---
        print(f"\nGenerating questions for '{topic}' targeting difficulty score ~{difficulty_score_str}...")
        prompt_questions = f"""
        Generate exactly 5 new, unique technical questions about the topic '{topic}' based on our ongoing conversation.
        Target difficulty score: Aim for questions around {difficulty_score_str} on a scale of 0.0 (very easy) to 10.0 (expert).

        For each generated question:
        1.  Provide the specific question text.
        2.  Estimate *its* specific difficulty score on the 0.0 to 10.0 scale, formatted to one decimal place.
        3.  Assign a sequential question number for this batch, starting from 1.

        IMPORTANT: Ensure these questions are different from any previously asked in this chat session. Do not include introductory text.

        Format the output STRICTLY as a JSON list of objects, like this example:
        [
            {"question": "What is RAM?", "difficulty": "1.5", "questionNo": "1"},
            {"question": "Explain memory virtualization.", "difficulty": "7.8", "questionNo": "2"},
            {"question": "...", "difficulty": "...", "questionNo": "3"},
            {"question": "...", "difficulty": "...", "questionNo": "4"},
            {"question": "...", "difficulty": "...", "questionNo": "5"}
        ]
        The output MUST be ONLY the valid JSON list and nothing else.
        """
        
        response_text_questions = call_gemini_chat(chat_session, prompt_questions)
        questions_data_raw = parse_json_response(response_text_questions, list)
        
        if not questions_data_raw:
            print("Could not get valid questions from Gemini. Trying again.")
            continue
        
        # --- Validate and Filter Questions ---
        valid_questions = []
        malformed_found = False
        
        for i, q in enumerate(questions_data_raw):
            if q and isinstance(q, dict) and 'question' in q and 'difficulty' in q and 'questionNo' in q:
                try:
                    q_text = str(q['question']).strip()
                    q_diff_str = str(q['difficulty'])
                    q_num_str = str(q['questionNo'])
                    
                    # Validate difficulty and number format
                    q_diff_num = float(q_diff_str)
                    q_num_int = int(q_num_str)
                    
                    if not q_text:
                        raise ValueError("Empty question text")
                    if q_diff_num < 0 or q_diff_num > 10:
                        raise ValueError("Invalid difficulty range")
                    
                    # Store with formatted difficulty
                    valid_questions.append({
                        'question': q_text,
                        'difficulty': f"{q_diff_num:.1f}",
                        'questionNo': q_num_str
                    })
                
                except (ValueError, TypeError) as e:
                    print(f"Warning: Skipping question {i + 1} due to invalid format/type/range ({str(e)}): {q}")
                    malformed_found = True
            else:
                print(f"Warning: Skipping question {i + 1} due to missing keys or incorrect type: {q}")
                malformed_found = True
        
        if not valid_questions:
            print("No valid questions were extracted from the response. Asking again.")
            continue
        
        if malformed_found:
            print("Note: Some potential questions returned by the API were skipped.")
        
        # Organize questions by number
        current_batch_questions = {}
        for q in valid_questions:
            current_batch_questions[q['questionNo']] = q
        
        # --- 4. Display Questions ---
        print("\n--- Please choose a question to answer ---")
        for q_data in sorted(valid_questions, key=lambda x: int(x['questionNo'])):
            print(f"\n{q_data['questionNo']}. (Est. Difficulty: {q_data['difficulty']})")
            print(f"   Q: {q_data['question']}")
        
        # --- 5. User Selects and Answers ---
        selected_q_num = None
        selected_question_data = None
        
        while selected_q_num is None:
            choice = ask_question("\nEnter the number of the question you want to answer: ").strip()
            if choice in current_batch_questions:
                selected_q_num = choice
                selected_question_data = current_batch_questions[choice]
            else:
                print(f"Invalid choice '{choice}'. Please enter one of: {', '.join(current_batch_questions.keys())}")
        
        print(f"\nYou selected Question {selected_q_num}:")
        print(f"Q: {selected_question_data['question']}")
        
        user_answer = ""
        while not user_answer:
            user_answer = ask_question("Your Answer: ").strip()
            if not user_answer:
                print("Answer cannot be empty.")
        
        # --- 6. Evaluate Answer and Get Score ---
        print("\nEvaluating your answer (focusing on key points and technical terms)...")
        prompt_score = f"""
        You are an expert technical evaluator for the topic '{topic}'.
        I previously asked you for questions, and you provided a list. I chose to answer question number {selected_q_num} from that list.

        The specific question was:
        "{selected_question_data['question']}"

        My provided answer is:
        "{user_answer}"

        **Evaluation Task:**
        1.  **Identify Key Technical Elements:** Based on the QUESTION, determine the essential concepts, terms, commands, etc., required for a correct answer.
        2.  **Analyze My Answer:** Examine MY ANSWER for the presence, accuracy, and appropriate use of these key technical elements.
        3.  **Score Based on Technical Accuracy:** Assign a score between 0.0 and 10.0, **allowing decimal values** (e.g., 7.5, 8.0, 9.2). This score should **primarily** reflect the technical accuracy, correct use of key terms, and understanding of core concepts.
            *   High (8.0-10.0): Strong understanding, accurate terms/concepts.
            *   Medium (5.0-7.9): Partial understanding, some correct elements but notable gaps/inaccuracies.
            *   Low (0.0-4.9): Misses main technical points, incorrect/irrelevant.
        4.  **Focus:** Prioritize technical substance and precision.

        **Output Format:**
        Respond ONLY with a valid JSON object containing the score, like this:
        {"score": "SCORE"}
        (e.g., {"score": "8.5"} or {"score": "7.0"} or {"score": "9"})
        Do not include any other text, just the JSON object.
        """
        
        response_text_score = call_gemini_chat(chat_session, prompt_score)
        score_data = parse_json_response(response_text_score, dict)
        
        if score_data and 'score' in score_data:
            print("\n--- Evaluation Complete ---")
            score_value = score_data['score']
            try:
                score_num = float(score_value)
                # Format to 1 decimal place for consistency
                print(f"Score (based on key points/terms): {score_num:.1f}/10.0")
            except ValueError:
                # Show raw if not a number
                print(f"Score (based on key points/terms): {score_value}/10.0")
        else:
            print("\n--- Could not get a valid score ---")
            print("Response might have been blocked, malformed, or missing the 'score' key.")
            if response_text_score:
                print(f"Gemini's raw response text was:\n{response_text_score}")
            else:
                print("(No response text received)")
        
        # --- 7. Continue or Exit ---
        print("-" * 40)
        next_action = ""
        while next_action not in ['y', 'n']:
            next_action = ask_question(f"Ask another question on '{topic}' ('y') or exit ('n')? [y/n]: ").lower().strip()
            if next_action not in ['y', 'n']:
                print("Please enter 'y' or 'n'.")
        
        if next_action == 'n':
            print("\nExiting the Q&A session. Goodbye!")
            break
        # Loop continues...

def main():
    """Entry point for the application."""
    try:
        run()
    except KeyboardInterrupt:
        print("\n\nSession interrupted. Exiting...")
    except Exception as e:
        print(f"\nAn unexpected error occurred in the main execution: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 