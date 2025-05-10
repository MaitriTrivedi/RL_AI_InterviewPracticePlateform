from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import uuid
import logging
from werkzeug.utils import secure_filename
import google.generativeai as genai
from dotenv import load_dotenv
import json
import random
import numpy as np
import time
import sys
from datetime import datetime

# Add parent directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_agent.PPO_RL_AGENT.interview_agent import InterviewAgent
from rl_agent.PPO_RL_AGENT.ppo_agent import PPOAgent
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
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True,
        "expose_headers": ["Content-Type"],
        "max_age": 600
    }
})

# Constants for directory paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
UPLOAD_FOLDER = os.path.join(DATA_DIR, 'uploads')
MODELS_DIR = os.path.join(DATA_DIR, 'models')
INTERVIEW_HISTORY_DIR = os.path.join(DATA_DIR, 'interviewHistory')
INTERVIEW_SCORE_HISTORY_DIR = os.path.join(DATA_DIR, 'interviewScoreHistory')
INTERVIEW_SESSIONS_DIR = os.path.join(DATA_DIR, 'interview-sessions')
MONITOR_DIR = os.path.join(DATA_DIR, 'monitor')

# Create all required directories
for directory in [UPLOAD_FOLDER, MODELS_DIR, INTERVIEW_HISTORY_DIR, INTERVIEW_SCORE_HISTORY_DIR, INTERVIEW_SESSIONS_DIR, MONITOR_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

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
    try:
        model_version = INTERVIEW_CONFIG['model']['version']
        if not model_version:
            logger.warning("No model version specified in config, using default initialization")
            return InterviewAgent(state_dim=9)
            
        logger.info(f"Creating interview agent with model version: {model_version}")
        agent = InterviewAgent(
            state_dim=9,
            model_version=model_version,
            models_dir=MODELS_DIR
        )
        return agent
        
    except Exception as e:
        logger.error(f"Error creating interview agent: {e}")
        # Return a new agent without loading a model as fallback
        return InterviewAgent(state_dim=9)

def create_fallback_question(subtopic: str, difficulty: float) -> dict:
    """Create a fallback question when Gemini fails."""
    # Get topic category
    topic_category = None
    for topic, subtopics in sde_topics.items():
        if subtopic in subtopics:
            topic_category = topic
            break
            
    return {
        "id": str(uuid.uuid4()),
        "topic": topic_category or "general",  # Use main topic or fallback to general
        "subtopic": subtopic,
        "difficulty": difficulty,
        "content": f"Please explain the concept of {subtopic} and its practical applications.",
        "follow_up_questions": [
            f"What are the main use cases of {subtopic}?",
            f"What are common challenges when working with {subtopic}?",
            f"How would you optimize {subtopic} implementations?"
        ],
        "evaluation_points": [
            "Basic understanding of the concept",
            "Practical applications",
            "Common challenges and solutions"
        ],
        "expected_time_minutes": max(10, int(difficulty * 2))  # Scale with difficulty
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
        
        if not response or not response.text:
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
        "feedback": "Your answer has been recorded. Due to technical limitations, a detailed evaluation couldn't be generated.",
        "strengths": [
            "Submitted an answer",
            "Participated in the interview process"
        ],
        "areas_for_improvement": [
            "Consider providing more detailed explanations",
            "Include specific examples when possible"
        ],
        "follow_up_suggestions": [
            "Review the topic documentation",
            "Practice with more examples"
        ]
    }

def generate_question(subtopic: str, difficulty: float, topic: str = None) -> dict:
    """Generate a question using Gemini based on subtopic and difficulty."""
    try:
        # Determine difficulty level category with more granular boundaries
        if difficulty <= 2.5:
            difficulty_category = "easy"
        elif difficulty <= 6.5:
            difficulty_category = "medium"
        else:
            difficulty_category = "hard"

        # Get topic category (ds, algo, dbms, etc.)
        topic_category = topic
        if not topic_category:
            for t, subtopics in sde_topics.items():
                if subtopic in subtopics:
                    topic_category = t
                break
        
        # Get relevant templates
        templates = []
        if topic_category and topic_category in question_templates:
            templates = question_templates[topic_category][difficulty_category]
        
        # Construct prompt for Gemini with templates and focused instructions
        template_suggestions = "\n".join([f"- {t.format(subtopic=subtopic)}" for t in templates]) if templates else ""
        
        # Enhanced difficulty-specific focus areas
        difficulty_focus = {
            "easy": "basic concepts, definitions, and simple examples",
            "medium": "practical applications, common use cases, and problem-solving",
            "hard": "advanced concepts, optimizations, edge cases, and system design considerations"
        }[difficulty_category]
        
        prompt = f"""Generate a technical interview question about {subtopic}.
Target difficulty: {difficulty}/10 (where 1 is easiest and 10 is hardest)

Here are some suggested question patterns for this topic and difficulty:
{template_suggestions}

The question should be:
1. Clear and specific to {subtopic}
2. Appropriate for the difficulty level {difficulty}/10
3. Include practical aspects and real-world scenarios when possible
4. Follow the pattern of the suggested templates but with your own specific details
5. For difficulty {difficulty}/10, focus on {difficulty_focus}

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
            "topic": topic_category or "general",  # Use the main topic category or fallback to general
            "subtopic": subtopic,     # Keep the specific subtopic
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
    """Start a new interview session."""
    try:
        # Create new session ID
        session_id = str(uuid.uuid4())
        
        # Create and initialize agent
        agent = create_agent()
        if not agent:
            return jsonify({
                'error': 'Failed to initialize interview agent'
            }), 500
        
        # Reset agent state
        agent.reset_interview_state()
        
        # Get first topic based on agent's policy
        topic_weights = [1.0 / (1.0 + agent.question_history[t]) for t in agent.topics]
        topic = np.random.choice(agent.topics, p=np.array(topic_weights)/sum(topic_weights))
        
        # Get subtopic using agent's policy
        subtopic = agent.select_subtopic(topic)
        
        # Get difficulty using agent's policy
        action_tuple = agent.get_next_question(topic)
        difficulty = float(action_tuple[0]) if isinstance(action_tuple, (tuple, list)) else 5.0  # Use first element as difficulty
        
        # Generate question
        question = generate_question(subtopic, difficulty, topic)
        if not question:
            question = create_fallback_question(subtopic, difficulty)
        
        # Store session information
        session_data = {
            'agent': agent,
            'current_question': question,
            'questions_asked': 0,
            'total_score': 0.0,
            'topic_performances': {},
            'last_activity': time.time(),
            'interview_complete': False
        }
        active_sessions[session_id] = session_data
        
        return jsonify({
            'message': 'Interview session started',
            'session_id': session_id,
            'initial_difficulty': difficulty,
            'session_stats': {
                'questions_asked': 0,
                'average_performance': 0.0,
                'topics': agent.topics,
                'current_topic': topic,
                'current_subtopic': subtopic
            },
            'first_question': {
                'id': question['id'],
                'topic': topic,
                'subtopic': subtopic,
                'difficulty': difficulty,
                'content': question['content'],
                'follow_up_questions': question.get('follow_up_questions', []),
                'evaluation_points': question.get('evaluation_points', []),
                'expected_time': question.get('expected_time_minutes', 5 + difficulty)
            }
        })
    
    except Exception as e:
        logger.error(f"Error in start_interview: {str(e)}")
        return jsonify({
            'error': 'Failed to start interview session'
        }), 500

def get_next_interview_number():
    """Get the next interview number by checking existing files."""
    existing_files = os.listdir(INTERVIEW_HISTORY_DIR)
    numbers = [int(f.split('_')[0]) for f in existing_files if f.endswith('.json') and f.split('_')[0].isdigit()]
    return max(numbers, default=0) + 1

def save_interview_history(session_id, current_data):
    """Save interview interaction to JSON file."""
    try:
        session = active_sessions.get(session_id)
        if not session:
            return
            
        # Get the next interview number
        interview_number = get_next_interview_number()
        
        # Prepare interview data
        interview_data = {
            'interview_id': session_id,
            'timestamp': datetime.now().isoformat(),
            'total_questions': session['questions_asked'],
            'total_score': session['total_score'],
            'current_interaction': {
                'question': session['current_question'],
                'answer': current_data.get('answer', ''),
                'evaluation': current_data.get('evaluation', {}),
                'time_taken': current_data.get('time_taken', 0)
            },
            'agent_state': {
                'question_history': session['agent'].question_history,
                'topic_performances': session['agent'].topic_performances,
                'current_difficulty': session['agent'].current_difficulty
            }
        }
        
        # Save to file
        filename = f"{interview_number}_interview_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(INTERVIEW_HISTORY_DIR, filename)
        
        with open(filepath, 'w') as f:
            json.dump(interview_data, f, indent=2)
            
        logger.info(f"Saved interview history to {filepath}")
        
    except Exception as e:
        logger.error(f"Error saving interview history: {str(e)}")

def save_final_interview_summary(session_id):
    """Save final interview summary with comprehensive statistics."""
    try:
        session = active_sessions.get(session_id)
        if not session:
            return
            
        # Get the interview number from the last interaction
        interview_number = get_next_interview_number() - 1  # Use the same number as interactions
        
        # Get agent stats
        agent = session['agent']
        stats = agent.get_interview_stats()
        
        # Calculate topic-wise statistics
        topic_stats = {}
        for topic in agent.topics:
            scores = agent.topic_performances.get(topic, [])
            if scores:
                topic_stats[topic] = {
                    'average_score': float(sum(scores) / len(scores)),
                    'max_score': float(max(scores)),
                    'min_score': float(min(scores)),
                    'questions_attempted': len(scores),
                    'difficulty_progression': [float(d) for d in agent.topic_difficulty_history.get(topic, [])],
                    'time_history': [float(t) for t in agent.topic_time_history.get(topic, [])]
                }
        
        # Prepare final summary with enhanced metrics
        final_summary = {
            'interview_id': session_id,
            'completion_time': datetime.now().isoformat(),
            'duration_stats': {
                'total_questions_answered': session['questions_asked'],
                'total_time': sum(sum(times) for times in agent.topic_time_history.values()),
                'average_time_per_question': float(sum(sum(times) for times in agent.topic_time_history.values()) / max(1, session['questions_asked']))
            },
            'performance_stats': {
                'final_score': session['total_score'],
                'average_score': session['total_score'] / max(1, session['questions_asked']),
                'topic_wise_stats': topic_stats
            },
            'learning_metrics': {
                'topic_coverage': len([t for t, score in agent.question_history.items() if score > 0]),
                'learning_progress': stats['learning_progress'],
                'difficulty_progression': {
                    'final': float(agent.current_difficulty),
                    'progression': [float(d) for topic in agent.topics for d in agent.topic_difficulty_history.get(topic, [])]
                }
            },
            'efficiency_metrics': {
                'time_efficiency': float(stats['time_efficiency']),
                'topic_transitions': agent.topic_transition_count,
                'warmup_phase_questions': agent.warmup_questions
            }
        }
        
        # Add strengths and areas for improvement
        final_summary['recommendations'] = {
            'strengths': [],
            'areas_for_improvement': [],
            'next_steps': []
        }
        
        for topic, tstats in topic_stats.items():
            if tstats['average_score'] >= 0.7:
                final_summary['recommendations']['strengths'].append(
                    f"Strong performance in {topic} with {tstats['average_score']*100:.1f}% average score"
                )
            elif tstats['average_score'] < 0.5:
                final_summary['recommendations']['areas_for_improvement'].append(
                    f"Need improvement in {topic} (current average: {tstats['average_score']*100:.1f}%)"
                )
                final_summary['recommendations']['next_steps'].append(
                    f"Focus on foundational concepts in {topic} starting with difficulty level {max(1.0, tstats['min_score'] * 5):.1f}"
                )
        
        # Save to file
        filename = f"{interview_number}_interview_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(INTERVIEW_HISTORY_DIR, filename)
        
        with open(filepath, 'w') as f:
            json.dump(final_summary, f, indent=2)
            
        logger.info(f"Saved enhanced final interview summary to {filepath}")
        
    except Exception as e:
        logger.error(f"Error saving final interview summary: {str(e)}")

@app.route('/api/interview/<interview_id>/submit-answer', methods=['POST', 'OPTIONS'])
def submit_answer(interview_id):
    """Submit an answer and get the next question."""
    def add_cors_headers(response):
        response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
        response.headers.add('Access-Control-Allow-Credentials', 'true')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        return response

    if request.method == 'OPTIONS':
        # Handle CORS preflight request
        response = jsonify({})
        response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        response.headers.add('Access-Control-Allow-Credentials', 'true')
        return response
        
    try:
        # Get session
        session = get_session(interview_id)
        if not session:
            return add_cors_headers(jsonify({'error': 'Invalid or expired session'})), 404
        
        # Get request data
        data = request.get_json()
        if not data:
            return add_cors_headers(jsonify({'error': 'No data provided'})), 400
            
        answer = data.get('answer', '')
        time_taken = float(data.get('time_taken', 0))  # Time taken in seconds
        
        # Get current question
        current_question = session.get('current_question')
        if not current_question:
            return add_cors_headers(jsonify({'error': 'No current question found'})), 400
        
        # Evaluate answer using Gemini
        try:
            evaluation = evaluate_answer(answer, current_question)
        except Exception as e:
            logger.error(f"Error during answer evaluation: {str(e)}")
            evaluation = create_fallback_evaluation(answer)
        
        # Save current interaction to history
        current_data = {
            'answer': answer,
            'time_taken': time_taken,
            'evaluation': evaluation
        }
        save_interview_history(interview_id, current_data)
        
        # Ensure we have a valid evaluation
        if not evaluation or not isinstance(evaluation, dict):
            logger.warning("Invalid evaluation result, using fallback")
            evaluation = create_fallback_evaluation(answer)
        
        # Calculate performance score (0.0 to 1.0)
        try:
            performance_score = float(evaluation.get('score', 5.0)) / 10.0  # Convert from 0-10 to 0-1 scale
        except (TypeError, ValueError) as e:
            logger.error(f"Error converting score: {str(e)}")
            performance_score = 0.5  # Default to middle score
        
        # Update agent with performance
        agent = session.get('agent')
        if not agent:
            return add_cors_headers(jsonify({'error': 'Invalid session state'})), 500
            
        try:
            # Update agent with performance and get next difficulty
            update_success = agent.update_performance(
                topic=current_question['topic'],
                subtopic=current_question['subtopic'],
                performance_score=performance_score,
                time_taken=time_taken
            )
            
            if not update_success:
                logger.warning("Failed to update agent performance")
                
        except Exception as e:
            logger.error(f"Error updating agent performance: {str(e)}")
            # Continue with the interview even if agent update fails
        
        # Update session statistics
        session['questions_asked'] += 1
        session['total_score'] += performance_score
        
        # Get next topic based on agent's policy
        topic_weights = [1.0 / (1.0 + agent.question_history[t]) for t in agent.topics]
        topic = np.random.choice(agent.topics, p=np.array(topic_weights)/sum(topic_weights))
        
        # Get next question parameters from agent
        action_tuple = agent.get_next_question(topic)
        difficulty, subtopic = action_tuple
        
        # Generate next question
        next_question = generate_question(subtopic, difficulty, topic)
        if not next_question:
            next_question = create_fallback_question(subtopic, difficulty)
        
        # Update session with new question
        session['current_question'] = next_question
        session['last_activity'] = time.time()
        
        # Get updated stats
        stats = agent.get_interview_stats()
        
        # Prepare response in the format frontend expects
        response_data = {
            'evaluation': {
                'score': evaluation.get('score', 5.0),
                'feedback': evaluation.get('feedback', 'Answer evaluated.'),
                'strengths': evaluation.get('strengths', []),
                'improvements': evaluation.get('areas_for_improvement', [])
            },
            'current_state': {
                'current_question': current_question,
                'session_stats': {
                    'questions_asked': session['questions_asked'],
                    'average_performance': stats['average_score'],
                    'current_topic': current_question['topic'],
                    'current_subtopic': current_question['subtopic'],
                    'current_difficulty': stats['difficulty_level'],
                    'topic_performances': stats['topic_performances'],
                    'learning_progress': stats['learning_progress']
                }
            },
            'next_state': {
                'next_question': next_question,
                'next_difficulty': difficulty,
                'next_topic': topic,
                'next_subtopic': subtopic,
                'interview_complete': False
            }
        }
        
        return add_cors_headers(jsonify(response_data))
        
    except Exception as e:
        logger.error(f"Error in submit_answer: {str(e)}")
        return add_cors_headers(jsonify({
            'error': f'Failed to process answer: {str(e)}'
        })), 500

@app.route('/api/interview/end', methods=['POST'])
def end_interview():
    """End the interview session with detailed performance analysis."""
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
                'error': 'Session not found',
                'status': 'error'
            }), 404
            
        session = active_sessions[session_id]
        agent = session.get('agent')
        
        if not agent:
            return jsonify({
                'error': 'Agent not found in session',
                'status': 'error'
            }), 500
        
        # Get comprehensive statistics
        stats = agent.get_interview_stats()
        
        # Calculate topic-wise performance
        topic_stats = {}
        for topic in agent.topics:
            topic_scores = agent.topic_performances.get(topic, [])
            if topic_scores:
                avg_score = sum(topic_scores) / len(topic_scores)
                max_score = max(topic_scores)
                improvement = 0
                if len(topic_scores) >= 2:
                    first_half = topic_scores[:len(topic_scores)//2]
                    second_half = topic_scores[len(topic_scores)//2:]
                    improvement = sum(second_half)/len(second_half) - sum(first_half)/len(first_half)
                
                topic_stats[topic] = {
                    'average_score': float(avg_score),
                    'max_score': float(max_score),
                    'questions_attempted': len(topic_scores),
                    'improvement': float(improvement),
                    'current_difficulty': float(agent.topic_current_difficulty.get(topic, 1.0))
                }
        
        # Calculate overall statistics
        total_questions = max(1, session['questions_asked'])
        average_score = session['total_score'] / total_questions
        
        # Prepare final statistics
        final_stats = {
            'overall_performance': {
                'total_questions': session['questions_asked'],
                'average_score': float(average_score),
                'final_difficulty': float(agent.current_difficulty),
                'time_efficiency': float(stats['time_efficiency']),
                'topic_coverage': float(stats['topic_coverage'])
            },
            'topic_wise_performance': topic_stats,
            'learning_progress': stats['learning_progress'],
            'strengths': [],
            'areas_for_improvement': []
        }
        
        # Identify strengths and areas for improvement
        for topic, perf in topic_stats.items():
            if perf['average_score'] >= 0.7:  # 70% or better
                final_stats['strengths'].append(f"Strong performance in {topic} with {perf['average_score']*100:.1f}% average score")
            elif perf['average_score'] < 0.5 and perf['questions_attempted'] > 0:  # Below 50%
                final_stats['areas_for_improvement'].append(f"Consider focusing more on {topic} (current average: {perf['average_score']*100:.1f}%)")
        
        # Save final interview summary
        save_final_interview_summary(session_id)
        
        # Clean up session
        del active_sessions[session_id]
        logger.info(f"Session {session_id} ended and removed. Remaining sessions: {list(active_sessions.keys())}")
        
        return jsonify({
            'message': 'Interview session ended successfully',
            'status': 'success',
            'final_stats': final_stats
        })
        
    except Exception as e:
        logger.error(f"Error in end_interview: {str(e)}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/api/interview/score-history', methods=['POST'])
def save_score_history():
    try:
        data = request.json
        
        # Generate filename with interview ID only
        filename = f"interview_{data['interviewId']}.json"
        filepath = os.path.join(INTERVIEW_SCORE_HISTORY_DIR, filename)
        
        # Load existing data if file exists
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                interview_data = json.load(f)
        else:
            interview_data = {
                'interviewId': data['interviewId'],
                'startTime': data['timestamp'],
                'questions': []
            }
        
        # Add new question data
        interview_data['questions'].append({
            'questionNumber': data['questionNumber'],
            'difficulty': data['difficulty'],
            'score': data['score'],
            'topic': data['topic'],
            'timeTaken': data['timeTaken'],
            'timestamp': data['timestamp']
        })
        
        # Update summary statistics
        interview_data['totalQuestions'] = len(interview_data['questions'])
        interview_data['averageScore'] = sum(q['score'] for q in interview_data['questions']) / len(interview_data['questions'])
        interview_data['averageDifficulty'] = sum(q['difficulty'] for q in interview_data['questions']) / len(interview_data['questions'])
        interview_data['totalTime'] = sum(q['timeTaken'] for q in interview_data['questions'])
        interview_data['lastUpdated'] = data['timestamp']
        
        # Save updated data
        with open(filepath, 'w') as f:
            json.dump(interview_data, f, indent=2)
            
        return jsonify({'status': 'success', 'message': 'Score history saved successfully'})
        
    except Exception as e:
        app.logger.error(f"Error saving score history: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000) 