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

# Question templates for different difficulty levels
question_templates = {
    "ds": {  # Data Structures
        "easy": [
            "Explain the basic concept and implementation of {subtopic}",
            "What are the main operations supported by {subtopic}?",
            "Draw and explain a simple example of {subtopic}"
        ],
        "medium": [
            "Compare and contrast different implementations of {subtopic}",
            "Solve a problem using {subtopic} with time complexity analysis",
            "Implement a specific operation in {subtopic} with error handling"
        ],
        "hard": [
            "Design an optimized version of {subtopic} for a specific use case",
            "Handle edge cases and performance bottlenecks in {subtopic}",
            "Combine multiple concepts with {subtopic} to solve a complex problem"
        ]
    },
    "algo": {  # Algorithms
        "easy": [
            "Explain how {subtopic} works with a simple example",
            "What is the time and space complexity of {subtopic}?",
            "Trace the execution of {subtopic} on a small input"
        ],
        "medium": [
            "Implement {subtopic} with specific constraints",
            "Optimize a basic implementation of {subtopic}",
            "Apply {subtopic} to solve a real-world problem"
        ],
        "hard": [
            "Design a variant of {subtopic} for a specific requirement",
            "Handle special cases and optimize {subtopic} for scale",
            "Combine {subtopic} with other algorithms to solve a complex problem"
        ]
    },
    "dbms": {  # Database Management
        "easy": [
            "Write a basic {subtopic} query",
            "Explain the concept of {subtopic} in DBMS",
            "What are the main components of {subtopic}?"
        ],
        "medium": [
            "Optimize a {subtopic} query for better performance",
            "Handle concurrent operations in {subtopic}",
            "Implement error handling in {subtopic}"
        ],
        "hard": [
            "Design a scalable solution using {subtopic}",
            "Handle distributed scenarios in {subtopic}",
            "Optimize {subtopic} for high-load situations"
        ]
    },
    "oops": {  # Object-Oriented Programming
        "easy": [
            "Explain the concept of {subtopic} in OOP",
            "Give a simple example of {subtopic}",
            "What are the basic principles of {subtopic}?"
        ],
        "medium": [
            "Implement a design pattern using {subtopic}",
            "Refactor code to better utilize {subtopic}",
            "Handle inheritance and polymorphism with {subtopic}"
        ],
        "hard": [
            "Design a complex system using {subtopic}",
            "Solve design problems using multiple OOP concepts including {subtopic}",
            "Optimize and scale a system using {subtopic}"
        ]
    },
    "os": {  # Operating Systems
        "easy": [
            "Explain the basic concept of {subtopic} in OS",
            "What are the main functions of {subtopic}?",
            "How does {subtopic} work in a simple scenario?"
        ],
        "medium": [
            "Handle synchronization in {subtopic}",
            "Implement a solution for {subtopic} with error handling",
            "Optimize resource usage in {subtopic}"
        ],
        "hard": [
            "Design a complex scheduling system for {subtopic}",
            "Handle deadlocks and race conditions in {subtopic}",
            "Implement an optimized version of {subtopic} for multi-core systems"
        ]
    },
    "cn": {  # Computer Networks
        "easy": [
            "Explain the basic concept of {subtopic} in networking",
            "What are the main protocols used in {subtopic}?",
            "How does {subtopic} work in a simple network?"
        ],
        "medium": [
            "Configure and troubleshoot {subtopic}",
            "Implement error handling in {subtopic}",
            "Optimize network performance using {subtopic}"
        ],
        "hard": [
            "Design a scalable network using {subtopic}",
            "Handle security concerns in {subtopic}",
            "Implement advanced protocols for {subtopic}"
        ]
    },
    "system_design": {  # System Design
        "easy": [
            "Explain the basic components needed for {subtopic}",
            "What are the key considerations in {subtopic}?",
            "Design a simple version of {subtopic}"
        ],
        "medium": [
            "Scale {subtopic} to handle more load",
            "Add fault tolerance to {subtopic}",
            "Optimize performance of {subtopic}"
        ],
        "hard": [
            "Design a globally distributed {subtopic}",
            "Handle extreme scale and reliability in {subtopic}",
            "Optimize cost and performance trade-offs in {subtopic}"
        ]
    }
}

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
        # Determine difficulty level category
        if difficulty <= 3:
            difficulty_category = "easy"
        elif difficulty <= 7:
            difficulty_category = "medium"
        else:
            difficulty_category = "hard"

        # Get topic category (ds, algo, dbms, etc.)
        topic_category = None
        for topic, subtopics in sde_topics.items():
            if subtopic in subtopics:
                topic_category = topic
                break
        
        # Get relevant templates
        templates = []
        if topic_category and topic_category in question_templates:
            templates = question_templates[topic_category][difficulty_category]
        
        # Construct prompt for Gemini with templates and focused instructions
        template_suggestions = "\n".join([f"- {t.format(subtopic=subtopic)}" for t in templates]) if templates else ""
        
        prompt = f"""Generate a technical interview question about {subtopic}.
Target difficulty: {difficulty}/10 (where 1 is easiest and 10 is hardest)

Here are some suggested question patterns for this topic and difficulty:
{template_suggestions}

The question should be:
1. Clear and specific to {subtopic}
2. Appropriate for the difficulty level {difficulty}/10
3. Include practical aspects and real-world scenarios when possible
4. Follow the pattern of the suggested templates but with your own specific details
5. For difficulty {difficulty}/10, focus on {'basic concepts and definitions' if difficulty <= 3 else 'application and problem-solving' if difficulty <= 7 else 'advanced concepts and edge cases'}

Respond with ONLY a JSON object in this format:
{{
    "question": "Your main question here",
    "follow_up_questions": [
        "A relevant follow-up question",
        "Another follow-up question"
    ],
    "difficulty_level": "{difficulty}",
    "key_points": [
        "Key concept or term to look for in answer",
        "Another important point to evaluate"
    ],
    "expected_time_minutes": number
}}"""

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
        required_fields = ['question', 'follow_up_questions', 'key_points', 'expected_time_minutes']
        missing_fields = [field for field in required_fields if field not in question_data]
        if missing_fields:
            logger.warning(f"Missing required fields in response: {', '.join(missing_fields)}")
            return create_fallback_question(subtopic, difficulty)
        
        # Format into our standard question structure
        formatted_question = {
            "id": str(uuid.uuid4()),
            "topic": subtopic,
            "difficulty": difficulty,
            "content": question_data['question'],
            "follow_up_questions": question_data['follow_up_questions'][:3],  # Limit to 3 follow-ups
            "evaluation_points": question_data['key_points'],
            "expected_time_minutes": question_data['expected_time_minutes']
        }
        
        return formatted_question
        
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
        logger.info(f"Request JSON: {data}")
        
        session_id = data.get('session_id')
        if not session_id:
            return jsonify({'error': 'session_id is required'}), 400
            
        # Log active sessions for debugging
        logger.info(f"Active sessions before lookup: {list(active_sessions.keys())}")
        
        if session_id not in active_sessions:
            logger.error(f"Session {session_id} not found in active sessions")
            return jsonify({
                'final_stats': {
                    'questions_asked': 0,
                    'average_performance': 0,
                    'final_difficulty': 5.0
                }
            })
            
        session = active_sessions[session_id]
        
        # Get final statistics
        total_questions = max(1, session['questions_asked'])
        stats = {
            'questions_asked': session['questions_asked'],
            'average_performance': session['total_performance'] / total_questions,
            'final_difficulty': session['current_difficulty']
        }
        
        # Clean up session
        del active_sessions[session_id]
        logger.info(f"Session {session_id} ended and removed. Remaining sessions: {list(active_sessions.keys())}")
        
        return jsonify({
            'message': 'Interview session ended',
            'final_stats': stats
        })
        
    except Exception as e:
        logger.error(f"Error in end_interview: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000) 