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
from rl_agent.model_inference import InterviewAgent
import time
from PPO_RL_AGENT.ppo_agent import PPOAgent
from rl_agent.interview_dataset import InterviewQuestionBank, InterviewSimulator
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
        "origins": ["http://localhost:3000"],  # Add your frontend URL
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define allowed extensions
ALLOWED_EXTENSIONS = {'pdf'}

# Initialize question bank and simulator
question_bank = InterviewQuestionBank()
interview_simulator = InterviewSimulator(question_bank)

# Store active sessions
active_sessions = {}

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

def create_agent():
    """Create a new PPO agent instance with expanded state space."""
    # State space: [difficulty, topic_idx, subtopic_idx, avg_performance]
    state_dim = 4
    # Action space: [difficulty_adjustment, topic_decision]
    action_dim = 2
    return PPOAgent(state_dim=state_dim, action_dim=action_dim)

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
        user_id = data.get('user_id')
        topic = data.get('topic', INTERVIEW_CONFIG['DEFAULT_TOPIC'])
        difficulty = float(data.get('difficulty', INTERVIEW_CONFIG['DEFAULT_DIFFICULTY']))
        max_questions = int(data.get('maxQuestions', INTERVIEW_CONFIG['MAX_QUESTIONS']))
        
        if not user_id:
            return jsonify({'error': 'user_id is required'}), 400
            
        # Get list of topics
        topics = list(sde_topics.keys())
        
        # Create new agent for this session
        agent = create_agent()
        
        # Initialize session state
        active_sessions[user_id] = {
            'agent': agent,
            'current_difficulty': difficulty,
            'questions_asked': 0,
            'total_performance': 0.0,
            'topics': topics,
            'current_topic_index': topics.index(topic) if topic in topics else 0,
            'max_questions': max_questions,
            'history': [],
            'current_subtopic_indices': {t: 0 for t in topics}
        }
        
        session = active_sessions[user_id]
        
        try:
            # Get current topic and subtopic
            current_topic = topics[session['current_topic_index']]
            current_subtopic = sde_topics[current_topic][session['current_subtopic_indices'][current_topic]]
            
            # Generate first question using Gemini
            question = generate_question(current_subtopic, difficulty)
            session['current_question'] = question
            
            # Format response
            response = {
                'message': 'Interview session started',
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
            
            return jsonify(response)
            
        except ValueError as e:
            return jsonify({'error': str(e)}), 500
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/interview/<interview_id>/next-question', methods=['GET', 'OPTIONS'])
def get_next_question(interview_id):
    """Get the next question's difficulty based on performance."""
    try:
        if request.method == 'OPTIONS':
            # Handle CORS preflight request
            response = jsonify({})
            response.headers.add('Access-Control-Allow-Origin', '*')
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
            response.headers.add('Access-Control-Allow-Methods', 'GET')
            return response
            
        session = active_sessions.get(interview_id)
        if not session:
            return jsonify({'error': 'No active interview session found'}), 404
            
        # Check if we have a current question
        if not session.get('current_question'):
            return jsonify({'error': 'No current question found'}), 404
            
        # Get next difficulty from agent
        state = np.array([
            session['current_difficulty'] / 10.0,
            session['current_topic_index'] / len(session['topics']),
            session['total_performance'] / max(1, session['questions_asked'])
        ])
        
        # Get action from agent
        action, value, _ = session['agent'].select_action(state)
        
        # Update difficulty
        new_difficulty = float(np.clip(action[0] * 10.0, 1.0, 10.0))
        session['current_difficulty'] = new_difficulty
        
        # Move to next topic
        session['current_topic_index'] = (session['current_topic_index'] + 1) % len(session['topics'])
        
        # Generate next question
        try:
            next_topic = session['topics'][session['current_topic_index']]
            next_subtopic = sde_topics[next_topic][session['current_subtopic_indices'][next_topic]]
            next_question = generate_question(next_subtopic, new_difficulty)
            session['current_question'] = next_question
            
            return jsonify({
                'question': {
                    'id': next_question['id'],
                    'topic': next_topic,
                    'subtopic': next_subtopic,
                    'difficulty': next_question['difficulty'],
                    'content': next_question['content'],
                    'follow_up_questions': next_question['follow_up_questions'],
                    'evaluation_points': next_question['evaluation_points']
                },
                'stats': {
                    'questions_asked': session['questions_asked'],
                    'average_performance': session['total_performance'] / max(1, session['questions_asked'])
                }
            })
            
        except Exception as e:
            logger.error(f"Error generating next question: {str(e)}")
            return jsonify({'error': f"Failed to generate next question: {str(e)}"}), 500
            
    except Exception as e:
        logger.error(f"Error in get_next_question: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/interview/<interview_id>/submit-answer', methods=['POST', 'OPTIONS'])
def submit_answer(interview_id):
    """Submit answer performance and update agent."""
    print("submit_answer called")
    
    try:
        if request.method == 'OPTIONS':
            # Handle CORS preflight request
            response = jsonify({})
            response.headers.add('Access-Control-Allow-Origin', '*')
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
            response.headers.add('Access-Control-Allow-Methods', 'POST')
            return response
            
        data = request.get_json()
        answer = data.get('answer')
        question_id = data.get('question_id')

        if not answer:
            return jsonify({'error': 'answer is required'}), 400

        session = active_sessions.get(interview_id)
        if not session:
            return jsonify({'error': 'No active interview session found'}), 404

        # Get current topic and subtopic
        current_topic = session['topics'][session['current_topic_index']]
        current_subtopic = sde_topics[current_topic][session['current_subtopic_indices'][current_topic]]

        # Evaluate answer using Gemini
        try:
            prompt = f"""You are an expert technical interviewer. Evaluate the following answer for a technical interview question.
            
                Question: {session['current_question']['content']}
                Answer: {answer}

                Evaluation points to consider:
                {session['current_question']['evaluation_points']}

                Provide evaluation in the following JSON format:
                {{
                    "score": <score from 0-10>,
                    "feedback": "detailed feedback explaining the score",
                    "strengths": ["list of strong points"],
                    "improvements": ["list of areas for improvement"]
                }}
                
                Important: Ensure the response is ONLY the JSON object, with no additional text, markdown formatting, or code blocks."""
            
            response = model.generate_content(prompt)
            
            # Clean the response text
            response_text = response.text.strip()
            # Remove any potential markdown code block formatting
            response_text = response_text.replace('```json', '').replace('```', '').strip()
            
            try:
                evaluation = json.loads(response_text)
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error: {str(e)}")
                logger.error(f"Raw response text: {response_text}")
                # Provide a fallback evaluation
                evaluation = {
                    "score": 5,
                    "feedback": "Unable to process evaluation. Default score assigned.",
                    "strengths": ["Answer was submitted successfully"],
                    "improvements": ["System was unable to provide detailed feedback"]
                }
            
            # Normalize score to 0-1 range for the RL agent
            normalized_score = float(evaluation['score']) / 10.0
            
            # Update session statistics
            session['questions_asked'] += 1
            session['total_performance'] += normalized_score
            
            # Calculate average performance
            avg_performance = session['total_performance'] / session['questions_asked']
            
            # State with 3 dimensions
            state = np.array([
                session['current_difficulty'] / 10.0,  # Normalize difficulty
                session['current_topic_index'] / len(session['topics']),  # Normalize topic index
                avg_performance  # Average performance
            ])
            
            # Calculate reward (simplified)
            reward = normalized_score
            
            # Store transition in agent's buffer
            done = session['questions_asked'] >= session['max_questions']
            session['agent'].store_transition(
                state=state,
                action=np.array([session['current_difficulty'] / 10.0]),  # Normalize action
                reward=reward,
                value=0.0,  # Will be computed in update
                log_prob=0.0,  # Will be computed in update
                done=done
            )
            
            # Get next difficulty from agent
            next_state = np.array([
                session['current_difficulty'] / 10.0,
                session['current_topic_index'] / len(session['topics']),
                avg_performance
            ])
            
            next_action, value, _ = session['agent'].select_action(next_state)
            next_difficulty = float(np.clip(next_action[0] * 10.0, 1.0, 10.0))
            
            # Update session state
            session['current_difficulty'] = next_difficulty
            session['current_topic_index'] = (session['current_topic_index'] + 1) % len(session['topics'])
            
            # Get next topic and subtopic
            next_topic = session['topics'][session['current_topic_index']]
            next_subtopic = sde_topics[next_topic][session['current_subtopic_indices'][next_topic]]
            
            # Generate next question with the new difficulty
            if not done:
                next_question = generate_question(next_subtopic, next_difficulty)
                session['current_question'] = next_question
            
            # Prepare enhanced response with current state information
            return jsonify({
                'evaluation': evaluation,
                'current_state': {
                    'current_question': {
                        'id': session['current_question']['id'],
                        'topic': current_topic,
                        'subtopic': current_subtopic,
                        'difficulty': session['current_difficulty'],
                        'content': session['current_question']['content'],
                        'follow_up_questions': session['current_question']['follow_up_questions'],
                        'evaluation_points': session['current_question']['evaluation_points']
                    },
                    'session_stats': {
                        'questions_asked': session['questions_asked'],
                        'total_questions': session['max_questions'],
                        'current_difficulty': session['current_difficulty'],
                        'average_performance': avg_performance * 10,  # Convert back to 0-10 scale
                        'current_topic': current_topic,
                        'current_subtopic': current_subtopic
                    }
                },
                'next_state': {
                    'next_difficulty': next_difficulty,
                    'next_topic': next_topic,
                    'next_subtopic': next_subtopic,
                    'interview_complete': done
                }
            })
            
        except Exception as e:
            logger.error(f"Error evaluating answer: {str(e)}")
            return jsonify({
                'error': f"Failed to evaluate answer: {str(e)}",
                'evaluation': {
                    'score': 5,
                    'feedback': "Error occurred during evaluation. Default score assigned.",
                    'strengths': ["Answer was submitted"],
                    'improvements': ["System encountered an error during evaluation"]
                },
                'current_state': {
                    'current_question': session['current_question'],
                    'session_stats': {
                        'questions_asked': session['questions_asked'],
                        'total_questions': session['max_questions'],
                        'current_difficulty': session['current_difficulty'],
                        'current_topic': current_topic,
                        'current_subtopic': current_subtopic
                    }
                }
            }), 200  # Return 200 with fallback evaluation instead of 500
            
    except Exception as e:
        logger.error(f"Error in submit_answer: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/interview/end', methods=['POST'])
def end_interview():
    """End the interview session."""
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        
        if not user_id:
            return jsonify({'error': 'user_id is required'}), 400
            
        session = active_sessions.get(user_id)
        if not session:
            return jsonify({'error': 'No active interview session found'}), 404
            
        # Get final statistics
        stats = {
            'questions_asked': session['questions_asked'],
            'average_performance': session['total_performance'] / max(1, session['questions_asked']),
            'final_difficulty': session['current_difficulty']
        }
        
        # Clean up session
        del active_sessions[user_id]
        
        return jsonify({
            'message': 'Interview session ended',
            'final_stats': stats
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/next-question', methods=['POST'])
def legacy_next_question():
    """Legacy endpoint for getting the next question's difficulty."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'error': 'Request body is required',
                'expected_format': {
                    'user_id': 'string',
                    'question': {
                        'topic_index': 'number (optional, default: 0)',
                        'time_allocated': 'number (optional, default: 15)'
                    }
                }
            }), 400

        user_id = data.get('user_id')
        if not user_id:
            return jsonify({'error': 'user_id is required'}), 400
            
        # Create a new session if one doesn't exist
        if user_id not in active_sessions:
            active_sessions[user_id] = {
                'agent': create_agent(),
                'current_difficulty': 5.0,
                'questions_asked': 0,
                'total_performance': 0.0,
                'history': [],
                'current_topic_index': 0
            }
        
        session = active_sessions[user_id]
        
        # Get current topic based on sequence
        topic = interview_simulator.topic_sequence[session['current_topic_index']]
        
        # Get question details with defaults
        question_data = data.get('question', {})
        if not isinstance(question_data, dict):
            question_data = {}
            
        time_allocated = question_data.get('time_allocated', 15)
        
        # Get question from question bank
        try:
            question = question_bank.get_question(topic, int(session['current_difficulty']))
            
            # Format response
            response = {
                'id': question['id'],
                'topic': topic,
                'difficulty': session['current_difficulty'],
                'content': question['question'],
                'expected_time': question['expected_time_minutes'],
                'follow_up_questions': question['follow_up_questions'],
                'subtopic': question['subtopic'],
                'session_stats': {
                    'questions_asked': session['questions_asked'],
                    'average_performance': session['total_performance'] / max(1, session['questions_asked']),
                    'current_topic': topic
                }
            }
            
            return jsonify(response)
            
        except ValueError as e:
            return jsonify({
                'error': str(e),
                'help': 'Failed to get question. The topic might not be available.'
            }), 404
            
    except Exception as e:
        return jsonify({
            'error': str(e),
            'help': 'Make sure to send a POST request with user_id and optionally question details'
        }), 500

# Update the progression logic to handle subtopics
def update_topic_and_subtopic(session, performance):
    """Update topic and subtopic based on performance."""
    current_topic = session['topics'][session['current_topic_index']]
    current_subtopic_idx = session['current_subtopic_indices'][current_topic]
    
    # If performance is good (> threshold), move to next subtopic
    if performance > INTERVIEW_CONFIG['PERFORMANCE_THRESHOLDS']['GOOD']:
        # If we have more subtopics in current topic
        if current_subtopic_idx < len(sde_topics[current_topic]) - 1:
            session['current_subtopic_indices'][current_topic] += 1
        # If we've completed all subtopics in current topic, move to next topic
        else:
            session['current_topic_index'] = (session['current_topic_index'] + 1) % len(session['topics'])
            # Reset subtopic index for new topic
            session['current_subtopic_indices'][current_topic] = 0
    # If performance is poor (< threshold), stay on same subtopic or move back
    elif performance < INTERVIEW_CONFIG['PERFORMANCE_THRESHOLDS']['POOR']:
        # If we're not at the first subtopic, move back one
        if current_subtopic_idx > 0:
            session['current_subtopic_indices'][current_topic] -= 1
    # For medium performance, continue to next subtopic
    else:
        # Move to next subtopic if available, otherwise next topic
        if current_subtopic_idx < len(sde_topics[current_topic]) - 1:
            session['current_subtopic_indices'][current_topic] += 1
        else:
            session['current_topic_index'] = (session['current_topic_index'] + 1) % len(session['topics'])
            session['current_subtopic_indices'][current_topic] = 0

    return session

# Update the PPO agent integration
def get_next_state(session, performance):
    """Get the next state for the RL agent."""
    current_topic = session['topics'][session['current_topic_index']]
    current_subtopic_idx = session['current_subtopic_indices'][current_topic]
    
    state = np.array([
        session['current_difficulty'] / 10.0,  # Normalize difficulty
        session['current_topic_index'] / len(session['topics']),  # Normalize topic index
        current_subtopic_idx / len(sde_topics[current_topic]),  # Normalize subtopic index
        session['total_performance'] / max(1, session['questions_asked'])  # Average performance
    ])
    return state

if __name__ == '__main__':
    app.run(debug=True, port=5000) 