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
import random
import numpy as np
import time
from PPO_RL_AGENT.interview_agent import InterviewAgent
from PPO_RL_AGENT.ppo_agent import PPOAgent
from config import INTERVIEW_CONFIG

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

# Enable CORS with specific configuration
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:3000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"],
        "supports_credentials": True,
        "expose_headers": ["Content-Type"],
        "max_age": 600
    }
})

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Configure session handling
app.config['SESSION_COOKIE_SECURE'] = False  # Set to True in production
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_DOMAIN'] = 'localhost'

# Store active sessions with TTL
active_sessions = {}
SESSION_TIMEOUT = 3600  # 1 hour timeout

# Available topics with their subtopics
sde_topics = {
    "ds": [
        "Arrays",
        "Strings",
        "Linked Lists (Singly, Doubly)",
        "Stacks and Queues",
        "Hashing (HashMaps, HashSets)",
        "Trees (Binary Tree, BST, Traversals)",
        "Heaps (Min/Max Heap, Priority Queue)",
        "Tries (Prefix Trees)",
        "Graphs (Adjacency List/Matrix, BFS, DFS)",
        "Segment Trees / Binary Indexed Trees"
    ],
    "algo": [
        "Sorting (Bubble, Selection, Insertion)",
        "Searching (Linear, Binary Search)",
        "Two Pointer Techniques",
        "Recursion",
        "Backtracking (N-Queens, Sudoku Solver)",
        "Greedy Algorithms",
        "Divide and Conquer (Merge Sort, Quick Sort)",
        "Sliding Window",
        "Dynamic Programming (Memoization, Tabulation)",
        "Graph Algorithms (Dijkstra, Floyd-Warshall, Topological Sort, Union-Find)"
    ],
    "dbms": [
        "Basic SQL (SELECT, INSERT, UPDATE, DELETE)",
        "Joins (INNER, LEFT, RIGHT, FULL)",
        "Constraints & Normalization (1NF, 2NF, 3NF)",
        "Indexes & Views",
        "Transactions (ACID Properties)",
        "Stored Procedures & Triggers",
        "Concurrency & Locking",
        "Query Optimization",
        "NoSQL vs RDBMS",
        "CAP Theorem & Distributed DB Concepts"
    ],
    "oops": [
        "Classes and Objects",
        "Encapsulation",
        "Inheritance",
        "Polymorphism (Compile-time, Run-time)",
        "Abstraction",
        "Interfaces and Abstract Classes",
        "SOLID Principles",
        "Design Patterns (Singleton, Factory, Observer)",
        "UML & Class Diagrams",
        "Real-world System Modeling"
    ],
    "os": [
        "Process vs Thread",
        "Memory Management (Paging, Segmentation)",
        "CPU Scheduling Algorithms (FCFS, SJF, RR)",
        "Deadlocks (Conditions, Prevention)",
        "Inter-Process Communication (IPC)",
        "Virtual Memory & Thrashing",
        "File Systems & Inodes",
        "Multithreading & Concurrency",
        "Mutex vs Semaphore",
        "Context Switching & Scheduling"
    ],
    "cn": [
        "OSI vs TCP/IP Models",
        "IP Addressing & Subnetting",
        "TCP vs UDP",
        "DNS, DHCP, ARP",
        "HTTP/HTTPS & REST APIs",
        "Routing & Switching Basics",
        "Firewalls & NAT",
        "Congestion Control (TCP Slow Start)",
        "Socket Programming",
        "Application Layer Protocols"
    ],
    "system_design": [
        "Basics of Scalability (Vertical vs Horizontal)",
        "Load Balancers",
        "Caching (Redis, CDN)",
        "Database Sharding & Replication",
        "CAP Theorem",
        "Message Queues (Kafka, RabbitMQ)",
        "Designing RESTful APIs",
        "Rate Limiting & Throttling",
        "High Availability & Fault Tolerance",
        "End-to-End Design of Systems (e.g., URL Shortener, Instagram)"
    ]
}

def cleanup_expired_sessions():
    """Remove expired sessions."""
    current_time = time.time()
    expired_sessions = [
        session_id for session_id, session in active_sessions.items()
        if current_time - session.get('last_activity', 0) > SESSION_TIMEOUT
    ]
    for session_id in expired_sessions:
        logger.info(f"Removing expired session: {session_id}")
        del active_sessions[session_id]

def get_session(session_id):
    """Get session with validation and activity update."""
    session = active_sessions.get(session_id)
    if session:
        session['last_activity'] = time.time()
        return session
    return None

@app.before_request
def before_request():
    """Handle pre-request tasks."""
    cleanup_expired_sessions()
    logger.info(f"Request path: {request.path}")
    logger.info(f"Request method: {request.method}")
    logger.info(f"Request headers: {dict(request.headers)}")
    if request.is_json:
        logger.info(f"Request JSON: {request.get_json()}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_agent():
    """Create a new interview agent instance with the trained model."""
    return InterviewAgent(
        state_dim=9,
        model_version='model_v1_20250427_151606_reward_0.773'  # Using our best trained model
    )

def create_fallback_question(topic: str, difficulty: float) -> dict:
    """Create a fallback question when Gemini fails."""
    return {
        "id": str(uuid.uuid4()),
        "topic": topic,
        "difficulty": difficulty,
        "content": f"Explain the core concepts of {topic} and provide examples of its practical applications.",
        "follow_up_questions": [
            f"What are the main challenges when working with {topic}?",
            f"How would you optimize a solution involving {topic}?",
            "Can you compare this with alternative approaches?"
        ],
        "evaluation_points": [
            "Understanding of core concepts",
            "Quality of examples provided",
            "Analysis of trade-offs"
        ],
        "subtopic": "fundamentals"
    }

def evaluate_answer(answer: str, question: dict) -> dict:
    """Evaluate student's answer using Gemini."""
    try:
        # Construct prompt for evaluation
        prompt = f"""You are an expert technical interviewer. Evaluate the following answer to a technical interview question.

Question: {question['content']}

Evaluation Points:
{chr(10).join(f"- {point}" for point in question['evaluation_points'])}

Student's Answer:
{answer}

Please evaluate the answer and respond with ONLY a JSON object in the following format (no other text):
{{
    "score": <number between 0-10>,
    "feedback": "<detailed feedback>",
    "strengths": [
        "<strength 1>",
        "<strength 2>"
    ],
    "areas_for_improvement": [
        "<area 1>",
        "<area 2>"
    ],
    "follow_up_suggestions": [
        "<suggestion 1>",
        "<suggestion 2>"
    ]
}}

The score should be based on:
1. Technical accuracy (40%)
2. Completeness of answer (30%)
3. Clarity of explanation (20%)
4. Practical application (10%)"""

        # Get evaluation from Gemini
        response = model.generate_content(prompt)
        
        if not response.text:
            logger.warning("Empty response from Gemini for evaluation")
            return create_fallback_evaluation(answer)
            
        # Remove Markdown code block formatting if present
        response_text = response.text.strip()
        if response_text.startswith('```json\n'):
            response_text = response_text[8:]
        if response_text.endswith('\n```'):
            response_text = response_text[:-4]
        
        try:
            evaluation = json.loads(response_text)
            
            # Validate required fields
            required_fields = ['score', 'feedback', 'strengths', 'areas_for_improvement']
            if not all(field in evaluation for field in required_fields):
                logger.warning("Missing required fields in evaluation response")
                return create_fallback_evaluation(answer)
                
            return evaluation
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON from Gemini evaluation: {response_text}")
            logger.error(f"JSON parse error: {str(e)}")
            return create_fallback_evaluation(answer)
            
    except Exception as e:
        logger.error(f"Error in evaluate_answer: {str(e)}")
        return create_fallback_evaluation(answer)

def create_fallback_evaluation(answer: str) -> dict:
    """Create a fallback evaluation when Gemini fails."""
    # Basic evaluation based on answer length and structure
    score = min(7, max(3, len(answer) / 100))  # Score between 3-7 based on length
    
    return {
        "score": score,
        "feedback": "Your answer demonstrates understanding of the topic. Consider providing more detailed examples and explanations.",
        "strengths": [
            "Attempted to address the question",
            "Provided some relevant information"
        ],
        "areas_for_improvement": [
            "Add more specific examples",
            "Explain concepts in more detail"
        ],
        "follow_up_suggestions": [
            "Consider discussing trade-offs",
            "Include real-world applications"
        ]
    }

def generate_question(subtopic: str, difficulty: float) -> dict:
    """Generate a question using Gemini based on subtopic and difficulty."""
    try:
        # Calculate target question length based on difficulty
        if difficulty <= 3:
            word_range = "50-100"
        elif difficulty <= 7:
            word_range = "100-200"
        else:
            word_range = "200-300"

        # Construct prompt for Gemini with strict formatting instructions
        prompt = f"""You MUST respond with ONLY a JSON object in the following format, with no additional text or explanation:

{{
    "content": "<insert main technical interview question here>",
    "follow_up_questions": [
        "<insert follow-up question 1>",
        "<insert follow-up question 2>",
        "<insert follow-up question 3>"
    ],
    "expected_time_minutes": <number>,
    "evaluation_points": [
        "<insert evaluation point 1>",
        "<insert evaluation point 2>",
        "<insert evaluation point 3>"
    ]
}}

Generate a technical interview question about:
Topic: {subtopic}
Difficulty: {difficulty}/10 (where 1 is easiest and 10 is hardest)
Question Length: {word_range} words

The question should be:
1. Challenging but clear for the given difficulty level
2. Within the specified word range
3. More focused and concise for lower difficulties
4. More complex and detailed for higher difficulties
5. Include relevant context and constraints based on the difficulty level

Remember: Respond with ONLY the JSON object, no other text."""

        # Generate response from Gemini
        response = model.generate_content(prompt)
        logger.info(f"Raw Gemini response: {response.text}")
        
        if not response.text:
            logger.warning("Empty response from Gemini, using fallback")
            return create_fallback_question(subtopic, difficulty)
            
        # Remove Markdown code block formatting if present
        response_text = response.text.strip()
        if response_text.startswith('```json\n'):
            response_text = response_text[8:]
        if response_text.endswith('\n```'):
            response_text = response_text[:-4]
        
        try:
            question_data = json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON from Gemini: {response_text}")
            logger.error(f"JSON parse error: {str(e)}")
            return create_fallback_question(subtopic, difficulty)
        
        # Validate required fields
        required_fields = ['content', 'follow_up_questions', 'expected_time_minutes', 'evaluation_points']
        missing_fields = [field for field in required_fields if field not in question_data]
        if missing_fields:
            logger.warning(f"Missing required fields in response: {', '.join(missing_fields)}")
            return create_fallback_question(subtopic, difficulty)
        
        # Add metadata
        question_data['id'] = str(uuid.uuid4())
        question_data['topic'] = subtopic
        question_data['difficulty'] = difficulty
        
        return question_data
    except Exception as e:
        logger.error(f"Failed to generate question: {str(e)}")
        logger.error(f"Full error context: {e.__class__.__name__}")
        return create_fallback_question(subtopic, difficulty)

@app.route('/api/interview/start', methods=['POST'])
def start_interview():
    """Start a new interview session with first medium difficulty question."""
    try:
        data = request.get_json()
        session_id = str(int(time.time() * 1000))
        topic = data.get('topic', INTERVIEW_CONFIG['DEFAULT_TOPIC'])
        difficulty = float(data.get('difficulty', INTERVIEW_CONFIG['DEFAULT_DIFFICULTY']))
        max_questions = int(data.get('maxQuestions', INTERVIEW_CONFIG['MAX_QUESTIONS']))
        
        logger.info(f"Starting new interview session with ID: {session_id}")
        logger.info(f"Request data: {data}")
        logger.info(f"Current active sessions before adding new one: {list(active_sessions.keys())}")
            
        # Get list of topics
        topics = list(sde_topics.keys())
        
        # Create new agent for this session
        agent = create_agent()
        
        # Initialize session state
        active_sessions[session_id] = {
            'agent': agent,
            'current_difficulty': difficulty,
            'questions_asked': 0,
            'total_performance': 0.0,
            'topics': topics,
            'current_topic_index': topics.index(topic) if topic in topics else 0,
            'max_questions': max_questions,
            'history': [],
            'last_activity': time.time()  # Add timestamp for session tracking
        }
        
        session = active_sessions[session_id]
        logger.info(f"Created session for ID {session_id}. Active sessions: {list(active_sessions.keys())}")
        
        try:
            # Get current topic and subtopic
            current_topic = topics[session['current_topic_index']]
            current_subtopic = sde_topics[current_topic][0]  # Start with first subtopic
            
            # Generate first question using Gemini
            question = generate_question(current_subtopic, difficulty)
            session['current_question'] = question
            
            # Format response
            response = {
                'message': 'Interview session started',
                'session_id': session_id,
                'initial_difficulty': difficulty,
                'session_stats': {
                    'questions_asked': 0,
                    'average_performance': 0.0,
                    'max_questions': max_questions,
                    'topics': topics,
                    'current_topic': current_topic,
                    'current_subtopic': current_subtopic
                },
                'first_question': {
                    'id': question['id'],
                    'topic': current_topic,
                    'subtopic': current_subtopic,
                    'difficulty': difficulty,
                    'content': question['content'],
                    'follow_up_questions': question['follow_up_questions'],
                    'evaluation_points': question['evaluation_points']
                }
            }
            
            logger.info(f"Interview session {session_id} initialized successfully")
            return jsonify(response)
            
        except ValueError as e:
            logger.error(f"Error in start_interview for session {session_id}: {str(e)}")
            if session_id in active_sessions:
                del active_sessions[session_id]
            return jsonify({'error': str(e)}), 500
        
    except Exception as e:
        logger.error(f"Error in start_interview: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/interview/<interview_id>/submit-answer', methods=['POST', 'OPTIONS'])
def submit_answer(interview_id):
    """Submit answer performance and update agent."""
    try:
        if request.method == 'OPTIONS':
            # Handle CORS preflight request
            response = jsonify({})
            response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
            response.headers.add('Access-Control-Allow-Methods', 'POST')
            response.headers.add('Access-Control-Allow-Credentials', 'true')
            return response

        logger.info(f"Submitting answer for session {interview_id}")
        logger.info(f"Current active sessions: {list(active_sessions.keys())}")
        
        # Get and validate session
        session = get_session(interview_id)
        if not session:
            logger.error(f"No active session found for ID {interview_id}")
            return jsonify({
                'error': 'No active interview session found. Please start a new interview.',
                'code': 'SESSION_NOT_FOUND'
            }), 404

        # Validate request data
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Request body is required'}), 400

        answer = data.get('answer', '').strip()
        if not answer:
            return jsonify({'error': 'Answer is required'}), 400

        # Get current question and validate
        current_question = session.get('current_question')
        if not current_question:
            logger.error(f"No current question found for session {interview_id}")
            return jsonify({
                'error': 'No current question found. Please start a new interview.',
                'code': 'QUESTION_NOT_FOUND'
            }), 404

        # Get time taken from request data
        time_taken = float(data.get('time_taken', 0))

        # Get current topic
        current_topic = session['topics'][session['current_topic_index']]
        logger.info(f"Processing answer for session {interview_id}, topic: {current_topic}")

        # Evaluate answer
        evaluation = evaluate_answer(answer, current_question)
        performance_score = float(evaluation['score']) / 10.0  # Normalize to 0-1

        # Update session statistics
        session['questions_asked'] += 1
        session['total_performance'] += performance_score
        session['history'].append({
            'question': current_question,
            'answer': answer,
            'evaluation': evaluation,
            'time_taken': time_taken
        })

        # Calculate average performance
        avg_performance = session['total_performance'] / session['questions_asked']

        # Check if interview is complete
        done = session['questions_asked'] >= session['max_questions']

        if not done:
            # Update agent with performance
            session['agent'].update_performance(current_topic, performance_score, time_taken)
            
            # Get next question parameters from agent
            action_info = session['agent'].get_next_question(current_topic)
            next_difficulty = action_info['difficulty']
            
            # Update session
            session['current_difficulty'] = next_difficulty
            session['current_topic_index'] = (session['current_topic_index'] + 1) % len(session['topics'])
            
            # Get next topic and subtopic
            next_topic = session['topics'][session['current_topic_index']]
            next_subtopic = sde_topics[next_topic][0]  # Start with first subtopic
            
            # Generate next question
            next_question = generate_question(next_subtopic, next_difficulty)
            session['current_question'] = next_question

            # Update session stats
            session_stats = {
                'questions_asked': session['questions_asked'],
                'total_questions': session['max_questions'],
                'current_difficulty': session['current_difficulty'],
                'average_performance': avg_performance * 10,  # Convert back to 0-10 scale
                'current_topic': next_topic,
                'current_subtopic': next_subtopic
            }

            # Prepare response
            response_data = {
                'evaluation': evaluation,
                'current_state': {
                    'current_question': current_question,
                    'session_stats': session_stats
                },
                'next_state': {
                    'next_question': next_question,
                    'next_difficulty': next_difficulty,
                    'next_topic': next_topic,
                    'next_subtopic': next_subtopic,
                    'interview_complete': done
                }
            }

            response = jsonify(response_data)
            response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
            response.headers.add('Access-Control-Allow-Credentials', 'true')
            return response
        else:
            # Interview complete
            response_data = {
                'evaluation': evaluation,
                'current_state': {
                    'session_stats': {
                        'questions_asked': session['questions_asked'],
                        'total_questions': session['max_questions'],
                        'average_performance': avg_performance * 10
                    }
                },
                'next_state': {
                    'interview_complete': True
                }
            }

            response = jsonify(response_data)
            response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
            response.headers.add('Access-Control-Allow-Credentials', 'true')
            return response

    except Exception as e:
        logger.error(f"Error in submit_answer: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/interview/end', methods=['POST'])
def end_interview():
    """End the interview session."""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        
        if not session_id:
            return jsonify({'error': 'session_id is required'}), 400
            
        session = get_session(session_id)
        if not session:
            return jsonify({'error': 'No active interview session found'}), 404
            
        # Get final statistics
        stats = {
            'questions_asked': session['questions_asked'],
            'average_performance': session['total_performance'] / max(1, session['questions_asked']),
            'final_difficulty': session['current_difficulty']
        }
        
        # Clean up session
        del active_sessions[session_id]
        
        return jsonify({
            'message': 'Interview session ended',
            'final_stats': stats
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000) 