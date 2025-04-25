from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import uuid
import logging
from werkzeug.utils import secure_filename
from ResumeExtractor.main import ResumeParser
import google.generativeai as genai
from dotenv import load_dotenv
import json
import random  # For initial random selection, will be replaced with RL
import numpy as np
from rl_agent.ppo_agent import PPOAgent
from rl_agent.interview_agent import InterviewAgent
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure Gemini
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY not found in environment variables")
    raise ValueError("GEMINI_API_KEY not found in environment variables")

try:
    genai.configure(api_key=GEMINI_API_KEY)
    # List available models
    for m in genai.list_models():
        logger.info(f"Available model: {m.name}")
    
    # Use the correct model name
    model = genai.GenerativeModel('gemini-2.0-flash')
except Exception as e:
    logger.error(f"Failed to configure Gemini API: {str(e)}")
    raise

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define allowed extensions
ALLOWED_EXTENSIONS = {'pdf'}

# In-memory storage for interviews (replace with database in production)
interviews = {}

# Initialize PPO agent
STATE_DIM = 17  # [avg_score, num_questions, current_difficulty, time_trend, performance_trend] + 12 topic weights
ACTION_DIM = 1  # Desired difficulty level
ppo_agent = PPOAgent(STATE_DIM, ACTION_DIM)

# Initialize RL agent
interview_agent = InterviewAgent()

# Load pre-trained model if exists
model_path = "rl_agent/interview_model.pth"
if os.path.exists(model_path):
    interview_agent.load(model_path)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/resume/upload', methods=['POST'])
def upload_resume():
    # Check if the post request has the file part
    if 'resume' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['resume']
    
    # If user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Generate unique filename
        filename = secure_filename(file.filename)
        resume_id = str(uuid.uuid4())
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{resume_id}_{filename}")
        
        # Save file
        file.save(file_path)
        
        try:
            # Parse resume
            parser = ResumeParser()
            parsed_resume = parser.parse_resume(file_path)
            
            # Return the parsed resume data and resume ID
            return jsonify({
                'resumeId': resume_id,
                'resumeData': parsed_resume
            }), 200
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/api/resume/<resume_id>', methods=['GET'])
def get_resume(resume_id):
    # In a real implementation, you would fetch the resume data from a database
    # For this example, we'll just return a mock response
    return jsonify({
        'resumeId': resume_id,
        'resumeData': {
            'education': [
                {
                    'institution': 'Example University',
                    'degree': 'Computer Science',
                    'year': '2018-2022',
                    'gpa': '3.8/4.0'
                }
            ],
            'work_experience': [
                {
                    'company': 'Tech Company',
                    'year': '2022-Present',
                    'description': 'Software Engineer working on backend systems'
                }
            ],
            'projects': [
                {
                    'name': 'Project Example',
                    'description': 'A machine learning project',
                    'technologies': ['Python', 'TensorFlow']
                }
            ]
        }
    }), 200

def generate_question(topic, params):
    """Generate a single question based on RL agent's parameters"""
    try:
        prompt = f"""Generate a {params['depth']} technical interview question about {params['topic']} in {topic}.
        Focus on {params['focus']}.
        The question should:
        1. Be at difficulty level {params['difficulty']}/10
        2. Test practical knowledge and problem-solving
        3. Be clearly stated and unambiguous
        4. Include specific requirements or constraints
        
        Format the response as:
        Question: [The actual question]
        Expected Time: [Estimated time to solve]
        Topic Focus: [Specific aspect of {params['topic']} being tested]
        """
        
        response = model.generate_content(prompt)
        if response and response.text:
            return {
                'questionId': str(uuid.uuid4()),
                'question': response.text,
                'difficulty': params['difficulty'],
                'topic': params['topic'],
                'depth': params['depth']
            }
        else:
            raise Exception("Failed to generate question")
            
    except Exception as e:
        logger.error(f"Error generating question: {str(e)}")
        raise

def select_next_question(interview):
    """Use RL agent to determine next question parameters and generate question"""
    try:
        # Get next question parameters from RL agent
        question_params = ppo_agent.get_next_question_params(interview)
        
        # Generate question based on these parameters
        question = generate_question(interview['topic'], question_params)
        
        # Update interview state
        interview['current_question'] = question
        if 'questions' not in interview:
            interview['questions'] = []
        interview['questions'].append(question)
        
        return interview
    except Exception as e:
        logger.error(f"Error in select_next_question: {str(e)}")
        raise

@app.route('/api/interview/new', methods=['POST'])
def create_interview():
    try:
        data = request.json
        if not data:
            logger.error("No JSON data received in request")
            return jsonify({'error': 'No data provided'}), 400
            
        if 'topic' not in data:
            logger.error("Topic not found in request data")
            return jsonify({'error': 'Topic is required'}), 400
        
        interview_id = str(uuid.uuid4())
        interview = {
            'interviewId': interview_id,
            'topic': data['topic'],
            'current_question': None,
            'questions': [],
            'answers': [],
            'status': 'in_progress'
        }
        
        # Generate questions
        interview = select_next_question(interview)
        interviews[interview_id] = interview
        
        return jsonify(interview), 200
        
    except Exception as e:
        logger.error(f"Error in create_interview: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/interview/<interview_id>/next-question', methods=['GET'])
def get_next_question(interview_id):
    if interview_id not in interviews:
        return jsonify({'error': 'Interview not found'}), 404
    
    interview = interviews[interview_id]
    if interview['status'] == 'completed':
        return jsonify({'error': 'Interview is already completed'}), 400
    
    # Select next question using RL
    interview = select_next_question(interview)
    interviews[interview_id] = interview
    
    return jsonify(interview), 200

def evaluate_answer(question: dict, answer: str) -> dict:
    """Evaluate the candidate's answer using Gemini."""
    prompt = f"""You are an expert technical interviewer evaluating a candidate's answer. Your response must be in valid JSON format.

Question:
{question['content']}

Candidate's Answer:
{answer}

Evaluate the answer and respond with ONLY a JSON object in the following format (no other text):
{{
    "score": <number between 1-10>,
    "feedback": "<your detailed feedback>",
    "correct_answer": "<brief sample solution if score < 7, otherwise null>"
}}

Base the score on:
- Technical accuracy (40%)
- Problem-solving approach (30%)
- Communication clarity (20%)
- Edge case consideration (10%)

IMPORTANT: 
1. Your entire response must be ONLY the JSON object
2. Use double quotes for strings
3. Use null instead of empty strings
4. Do not include any explanatory text outside the JSON
5. Ensure the feedback is a single line with escaped quotes"""

    try:
        response = model.generate_content(prompt)
        if not response or not response.text:
            raise ValueError("Empty response from Gemini")
            
        # Clean the response text
        response_text = response.text.strip()
        if not response_text.startswith('{'):
            # Try to find the JSON object in the response
            start = response_text.find('{')
            end = response_text.rfind('}')
            if start != -1 and end != -1:
                response_text = response_text[start:end+1]
            else:
                raise ValueError("No valid JSON object found in response")

        # Parse the JSON
        evaluation = json.loads(response_text)
        
        # Validate the required fields
        if not isinstance(evaluation.get('score'), (int, float)):
            raise ValueError("Invalid score format")
        if not isinstance(evaluation.get('feedback'), str):
            raise ValueError("Invalid feedback format")
            
        # Ensure score is within bounds
        evaluation['score'] = max(1, min(10, float(evaluation['score'])))
        
        return evaluation
    except Exception as e:
        logger.error(f"Error evaluating answer: {str(e)}")
        logger.error(f"Raw response: {response.text if response else 'No response'}")
        return {
            "score": 0,
            "feedback": "Error evaluating answer. Please try again.",
            "correct_answer": None
        }

@app.route('/api/interview/<interview_id>/submit-answer', methods=['POST'])
def submit_answer(interview_id):
    try:
        if interview_id not in interviews:
            return jsonify({'error': 'Interview not found'}), 404
            
        data = request.json
        if not data or 'answer' not in data or 'questionId' not in data:
            return jsonify({'error': 'Answer and questionId are required'}), 400
            
        interview = interviews[interview_id]
        question = next((q for q in interview['questions'] if q['questionId'] == data['questionId']), None)
        
        if not question:
            return jsonify({'error': 'Question not found'}), 404
            
        # Evaluate the answer using Gemini
        evaluation = evaluate_answer(question, data['answer'])
        
        # Store the answer with evaluation
        answer = {
            'questionId': data['questionId'],
            'answer': data['answer'],
            'score': evaluation['score'],
            'feedback': evaluation['feedback'],
            'correct_answer': evaluation['correct_answer']
        }
        
        interview['answers'].append(answer)
        
        return jsonify({
            'interview': interview,
            'evaluation': evaluation
        }), 200
        
    except Exception as e:
        logger.error(f"Error in submit_answer: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/interview/<interview_id>/results', methods=['GET'])
def get_results(interview_id):
    if interview_id not in interviews:
        return jsonify({'error': 'Interview not found'}), 404
    
    interview = interviews[interview_id]
    return jsonify(interview), 200

@app.route('/api/next-question', methods=['POST'])
def get_next_question_rl():
    """Get the next interview question based on RL agent's selection."""
    try:
        data = request.get_json()
        previous_score = data.get('previousScore')
        previous_topic = data.get('previousTopic')
        time_taken = data.get('timeTaken', 300)  # Default to 5 minutes if not provided
        
        question = interview_agent.select_next_question(
            previous_score=previous_score,
            time_taken=time_taken,
            previous_topic=previous_topic
        )
        
        return jsonify(question)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/evaluate-answer', methods=['POST'])
def evaluate_submission():
    """Evaluate the candidate's answer and update the RL agent."""
    try:
        data = request.get_json()
        question_id = data.get('questionId')
        answer = data.get('answer')
        
        # Find the question from the dataset
        question = next(
            (q for q in interview_agent.questions if q['id'] == question_id),
            None
        )
        if not question:
            return jsonify({"error": "Question not found"}), 404
        
        # Evaluate the answer
        start_time = time.time()
        evaluation = evaluate_answer(question, answer)
        time_taken = time.time() - start_time
        
        # Update RL agent's state and policy
        current_state = interview_agent.current_state.copy()
        next_state = interview_agent.current_state.copy()
        next_state['avg_score'] = (current_state['avg_score'] * current_state['num_questions'] + evaluation['score']) / (current_state['num_questions'] + 1)
        next_state['num_questions'] += 1
        
        # Calculate reward based on score and adaptation
        base_reward = evaluation['score'] / 10.0  # Normalize to 0-1
        adaptation_bonus = 0.1 if abs(question['difficulty'] - current_state['current_difficulty']) <= 2 else -0.1
        reward = base_reward + adaptation_bonus
        
        # Train the agent
        interview_agent.train_step(
            state=current_state,
            action=question['difficulty'],
            reward=reward,
            next_state=next_state,
            done=False
        )
        
        # Save the updated model
        interview_agent.save(model_path)
        
        return jsonify(evaluation)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/end-interview', methods=['POST'])
def end_interview():
    """End the interview and reset the agent's state."""
    try:
        # Save final model state
        interview_agent.save(model_path)
        # Reset agent for next interview
        interview_agent.reset_state()
        return jsonify({"message": "Interview ended successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000) 