import json
from typing import Dict, List
import random
import asyncio
from vertexai.language_models import TextGenerationModel

class InterviewQuestionBank:
    def __init__(self):
        # Initialize Gemini model
        self.model = TextGenerationModel.from_pretrained("text-bison@001")
        # Define topics with their sequence
        self.sde_topics = {
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
        
        # Use these as main topics
        self.topics = list(self.sde_topics.keys())
        
        # Initialize question bank with difficulty levels 1-10 for each topic
        self.questions = {
            topic: {
                difficulty: [] for difficulty in range(1, 11)
            } for topic in self.topics
        }
        
        # Generate questions based on topic sequence
        self._generate_sequential_questions()
    
    def _generate_prompt(self, topic: str, subtopic: str, difficulty: int) -> str:
        """Generate prompt for Gemini based on difficulty level"""
        base_prompt = f"""Generate a technical interview question about {subtopic} (under {topic}) with difficulty level {difficulty}/10.

For difficulty {difficulty}:
- Levels 1-3: Focus on basic concepts, definitions, and simple examples
- Levels 4-6: Focus on implementation details, internal workings, and practical applications
- Levels 7-10: Focus on complex problems, optimizations, and system design considerations

The question should:
1. Be clear and unambiguous
2. Match the appropriate difficulty level
3. Include 2-3 relevant follow-up questions
4. For implementation questions, mention any constraints or requirements
5. For system design questions, include scale and performance considerations

Format the response as a JSON object with fields:
{
    "question": "main question text",
    "follow_up_questions": ["question1", "question2"],
    "expected_time_minutes": number,
    "constraints": ["constraint1", "constraint2"] // if applicable
}"""
        return base_prompt

    async def _generate_question_with_gemini(self, topic: str, subtopic: str, difficulty: int) -> Dict:
        """Generate a question using Gemini"""
        prompt = self._generate_prompt(topic, subtopic, difficulty)
        
        try:
            response = await self.model.generate_content_async(prompt)
            question_data = json.loads(response.text)
            
            return {
                "id": f"{topic}_{difficulty:02d}_{random.randint(1000, 9999)}",
                "topic": topic,
                "subtopic": subtopic,
                "difficulty": difficulty,
                "question": question_data["question"],
                "follow_up_questions": question_data["follow_up_questions"],
                "expected_time_minutes": question_data["expected_time_minutes"],
                "constraints": question_data.get("constraints", [])
            }
        except Exception as e:
            print(f"Error generating question: {e}")
            # Fallback to a basic question if Gemini fails
            return {
                "id": f"{topic}_{difficulty:02d}_{random.randint(1000, 9999)}",
                "topic": topic,
                "subtopic": subtopic,
                "difficulty": difficulty,
                "question": f"Explain {subtopic} and its practical applications.",
                "follow_up_questions": [
                    f"What are the common challenges in implementing {subtopic}?",
                    f"How would you optimize {subtopic} for better performance?"
                ],
                "expected_time_minutes": 5 + difficulty * 2
            }

    def _generate_sequential_questions(self):
        """Generate questions based on topic sequence where index determines difficulty"""
        loop = asyncio.get_event_loop()
        for topic, subtopics in self.sde_topics.items():
            for idx, subtopic in enumerate(subtopics):
                difficulty = max(1, min(10, (idx + 1)))
                question = loop.run_until_complete(
                    self._generate_question_with_gemini(topic, subtopic, difficulty)
                )
                self.questions[topic][difficulty].append(question)
    
    def add_question(self, question: Dict):
        """Add a question to the question bank"""
        topic = question["topic"]
        difficulty = question["difficulty"]
        if difficulty < 1:
            difficulty = 1
        elif difficulty > 10:
            difficulty = 10
        self.questions[topic][difficulty].append(question)
    
    def get_question(self, topic: str, difficulty: int) -> Dict:
        """Get a random question of specified topic and difficulty"""
        # Ensure difficulty is within bounds
        difficulty = max(1, min(10, difficulty))
        
        # Try to find a question at the specified difficulty
        if self.questions[topic][difficulty]:
            return random.choice(self.questions[topic][difficulty])
        
        # If no question at specified difficulty, search nearby difficulties
        lower = difficulty - 1
        higher = difficulty + 1
        
        while lower >= 1 or higher <= 10:
            # Check lower difficulty first
            if lower >= 1 and self.questions[topic][lower]:
                return random.choice(self.questions[topic][lower])
            # Then check higher difficulty
            if higher <= 10 and self.questions[topic][higher]:
                return random.choice(self.questions[topic][higher])
            lower -= 1
            higher += 1
        
        # If still no question found, return a medium difficulty question
        for d in [5, 4, 6, 3, 7, 2, 8, 1, 9, 10]:
            if self.questions[topic][d]:
                return random.choice(self.questions[topic][d])
        
        raise ValueError(f"No questions available for topic {topic}")

class InterviewSimulator:
    def __init__(self, question_bank: InterviewQuestionBank):
        self.question_bank = question_bank
        self.questions_per_interview = 10
        self.current_topic_index = 0
        self.topic_sequence = self._determine_topic_sequence()
        
    def _determine_topic_sequence(self) -> List[str]:
        """Determine logical sequence of topics for interview"""
        # Start with fundamentals, then move to more complex topics
        return [
            "ds",        # Start with data structures
            "algo",      # Then algorithms
            "oops",      # Object-oriented concepts
            "dbms",      # Database concepts
            "os",        # Operating system fundamentals
            "cn",        # Computer networks
            "system_design"  # End with system design
        ]
        
    def start_interview(self) -> Dict:
        """Start a new interview with medium difficulty question"""
        self.current_topic_index = 0
        topic = self.topic_sequence[self.current_topic_index]
        # Start with medium difficulty (5)
        return self.question_bank.get_question(topic, 5)
    
    def next_question(self, prev_performance: float, prev_topic: str) -> Dict:
        """Get next question based on previous performance"""
        # Adjust difficulty based on performance
        if prev_performance > 0.8:
            difficulty_change = 2  # Significant increase
        elif prev_performance > 0.6:
            difficulty_change = 1  # Slight increase
        elif prev_performance < 0.3:
            difficulty_change = -2  # Significant decrease
        elif prev_performance < 0.5:
            difficulty_change = -1  # Slight decrease
        else:
            difficulty_change = 0  # Keep same difficulty
            
        # Move to next topic in sequence
        self.current_topic_index = (self.current_topic_index + 1) % len(self.topic_sequence)
        next_topic = self.topic_sequence[self.current_topic_index]
        
        # Get current difficulty of previous question and adjust
        prev_question = self.question_bank.get_question(prev_topic, 5)  # Default to medium if not found
        new_difficulty = max(1, min(10, prev_question["difficulty"] + difficulty_change))
        
        return self.question_bank.get_question(next_topic, new_difficulty)

if __name__ == "__main__":
    # Generate dataset
    question_bank = InterviewQuestionBank()  # Now automatically generates questions
    
    # Create simulator
    simulator = InterviewSimulator(question_bank)
    
    # Example simulation
    print("Starting interview simulation...")
    question = simulator.start_interview()
    print(f"\nInitial Question (Medium Difficulty):")
    print(f"Topic: {question['topic']}")
    print(f"Subtopic: {question['subtopic']}")
    print(f"Difficulty: {question['difficulty']}")
    print(f"Question: {question['question']}")
    
    # Simulate a few questions with different performance levels
    performances = [0.9, 0.3, 0.6]  # Example performance scores
    
    for i, perf in enumerate(performances, 1):
        prev_topic = question['topic']
        question = simulator.next_question(perf, prev_topic)
        print(f"\nQuestion {i+1} (After performance: {perf}):")
        print(f"Topic: {question['topic']}")
        print(f"Subtopic: {question['subtopic']}")
        print(f"Difficulty: {question['difficulty']}")
        print(f"Question: {question['question']}") 