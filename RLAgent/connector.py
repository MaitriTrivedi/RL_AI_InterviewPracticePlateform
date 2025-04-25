"""
connector.py - Integration Module for Interview System Components

This module provides integration between the ResumeExtractor, 
QuestionGenerator, and RL agent components of the system.
"""

import os
import sys
import json
import subprocess
import tempfile
from typing import Dict, List, Any, Tuple, Optional
import shlex

# Add parent directory to path to import from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our Python QuestionGenerator
try:
    from QuestionGenerator import QuestionGenerator as PythonQuestionGenerator
    PYTHON_QG_AVAILABLE = True
except ImportError:
    PYTHON_QG_AVAILABLE = False

class ResumeExtractorConnector:
    """
    Connector for the ResumeExtractor component.
    Provides a Python interface to the resume parsing functionality.
    """
    
    def __init__(self, resume_extractor_path: str = None):
        """
        Initialize the ResumeExtractor connector.
        
        Args:
            resume_extractor_path: Path to the ResumeExtractor module
        """
        # Use default path if not provided
        self.resume_extractor_path = resume_extractor_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'ResumeExtractor'
        )
    
    def extract_resume_data(self, resume_path: str) -> Dict[str, Any]:
        """
        Extract structured data from a resume.
        
        Args:
            resume_path: Path to the resume file
            
        Returns:
            Structured resume data as a dictionary
        """
        # Build the command
        main_script = os.path.join(self.resume_extractor_path, 'main.py')
        
        try:
            # Run the resume extractor as a subprocess
            cmd = [sys.executable, main_script, resume_path]
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True,
                check=True
            )
            
            # Parse the JSON output
            try:
                resume_data = json.loads(result.stdout)
                return resume_data
            except json.JSONDecodeError:
                # If JSON parsing fails, look for the output file
                output_file = f"{resume_path.split('.')[0]}_parsed.json"
                if os.path.exists(output_file):
                    with open(output_file, 'r') as f:
                        return json.load(f)
                else:
                    raise ValueError(f"Failed to parse resume data and no output file found at {output_file}")
        
        except subprocess.CalledProcessError as e:
            print(f"Error running ResumeExtractor: {e}")
            print(f"Stdout: {e.stdout}")
            print(f"Stderr: {e.stderr}")
            raise ValueError(f"ResumeExtractor failed with exit code {e.returncode}")


class QuestionGeneratorConnector:
    """
    Connector for the QuestionGenerator component.
    
    This connector uses the Python QuestionGenerator API if available,
    falling back to the JavaScript implementation if needed.
    """
    
    def __init__(self, question_generator_path: str = None):
        """
        Initialize the QuestionGenerator connector.
        
        Args:
            question_generator_path: Path to the QuestionGenerator module
        """
        # Use default path if not provided
        self.question_generator_path = question_generator_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'QuestionGenerator'
        )
        
        # Try to use the Python implementation
        self.python_generator = None
        if PYTHON_QG_AVAILABLE:
            try:
                self.python_generator = PythonQuestionGenerator()
                print("Using Python QuestionGenerator API")
            except Exception as e:
                print(f"Error initializing Python QuestionGenerator: {e}")
                print("Will fall back to JavaScript implementation")
                self.python_generator = None
    
    def generate_questions(
        self, 
        topic: str, 
        difficulty: float,
        num_questions: int = 5,
        custom_prompt: str = None
    ) -> List[Dict[str, Any]]:
        """
        Generate interview questions for a given topic and difficulty.
        
        Args:
            topic: Topic for questions (e.g., "Python", "Machine Learning")
            difficulty: Target difficulty level (0-10)
            num_questions: Number of questions to generate
            custom_prompt: Optional custom prompt for the LLM
            
        Returns:
            List of question dictionaries
        """
        # If Python implementation is available, use it
        if self.python_generator:
            try:
                return self.python_generator.generate_questions(
                    topic=topic,
                    difficulty=difficulty,
                    num_questions=num_questions,
                    custom_prompt=custom_prompt
                )
            except Exception as e:
                print(f"Error using Python QuestionGenerator: {e}")
                print("Falling back to JavaScript implementation")
        
        # Fall back to JavaScript implementation if Python version fails or is not available
        # Create a temporary input file with the parameters
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            input_file = f.name
            json.dump({
                'topic': topic,
                'difficulty': difficulty,
                'num_questions': num_questions,
                'custom_prompt': custom_prompt
            }, f)
        
        try:
            # Build the command to run the NodeJS script
            # Assumes there's a questions.js script in the QuestionGenerator directory
            script_path = os.path.join(self.question_generator_path, 'questions.js')
            
            # Check if we need to create this script
            if not os.path.exists(script_path):
                self._create_questions_script(script_path)
            
            # Run the generator
            cmd = ['node', script_path, input_file]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Parse the output
            try:
                questions = json.loads(result.stdout)
                return questions
            except json.JSONDecodeError:
                print(f"Warning: Failed to parse JSON from stdout: {result.stdout}")
                # Try to read from output file if stdout parsing fails
                output_file = f"{input_file}_output.json"
                if os.path.exists(output_file):
                    with open(output_file, 'r') as f:
                        return json.load(f)
                else:
                    raise ValueError("Failed to get questions from generator")
        
        except subprocess.CalledProcessError as e:
            print(f"Error running QuestionGenerator: {e}")
            print(f"Stdout: {e.stdout}")
            print(f"Stderr: {e.stderr}")
            raise ValueError(f"QuestionGenerator failed with exit code {e.returncode}")
        
        finally:
            # Clean up temp file
            if os.path.exists(input_file):
                os.unlink(input_file)
    
    def _create_questions_script(self, script_path: str):
        """
        Create a simple script to generate questions if one doesn't exist.
        
        This is a temporary solution until the main QuestionGenerator 
        is updated to work as a module/API.
        
        Args:
            script_path: Path to create the script
        """
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(script_path), exist_ok=True)
        
        script_content = """
// questions.js - Question Generation Script for the RL Agent
// This script provides a programmatic interface to the question generator.

const fs = require('fs');
const path = require('path');
require('dotenv').config();

// Load Google Generative AI library if available
let GoogleGenerativeAI;
try {
  GoogleGenerativeAI = require('@google/generative-ai');
} catch (error) {
  console.error('Error: @google/generative-ai package not found.');
  console.error('Please install it with: npm install @google/generative-ai');
  process.exit(1);
}

// Check for API key
const apiKey = process.env.GEMINI_API_KEY;
if (!apiKey) {
  console.error('Error: GEMINI_API_KEY environment variable not set.');
  process.exit(1);
}

// Function to generate questions
async function generateQuestions(params) {
  const {
    topic,
    difficulty,
    num_questions = 5,
    custom_prompt = null
  } = params;

  // Configure the model
  const genAI = new GoogleGenerativeAI(apiKey);
  const model = genAI.getGenerativeModel({ model: 'gemini-1.5-flash-latest' });
  const chatSession = model.startChat();

  // Create a prompt for generating questions
  const difficultyStr = difficulty.toFixed(1);
  
  let prompt;
  if (custom_prompt) {
    prompt = custom_prompt;
  } else {
    prompt = `
    Generate exactly ${num_questions} new, unique technical questions about the topic '${topic}'.
    Target difficulty score: Aim for questions around ${difficultyStr} on a scale of 0.0 (very easy) to 10.0 (expert).

    For each generated question:
    1. Provide the specific question text.
    2. Estimate *its* specific difficulty score on the 0.0 to 10.0 scale, formatted to one decimal place.
    3. Assign a sequential question number for this batch, starting from 1.

    Format the output STRICTLY as a JSON list of objects, like this example:
    [
        {"question": "What is RAM?", "difficulty": "1.5", "questionNo": "1"},
        {"question": "Explain memory virtualization.", "difficulty": "7.8", "questionNo": "2"},
        {"question": "...", "difficulty": "...", "questionNo": "3"}
    ]
    The output MUST be ONLY the valid JSON list and nothing else.
    `;
  }

  try {
    // Send the prompt to the model
    const result = await chatSession.sendMessage(prompt);
    const responseText = result.response.text();
    
    // Extract JSON from the response
    const jsonMatch = responseText.match(/\\[\\s*\\{.*\\}\\s*\\]/s);
    if (jsonMatch) {
      const jsonText = jsonMatch[0];
      return JSON.parse(jsonText);
    }
    
    // Fallback: try to parse the entire response as JSON
    try {
      return JSON.parse(responseText);
    } catch (error) {
      console.error("Failed to extract valid JSON from response.");
      console.error("Raw response:", responseText);
      return [];
    }
  } catch (error) {
    console.error(`Error generating questions: ${error.message}`);
    return [];
  }
}

// Main function
async function main() {
  // Read input parameters from command line argument (JSON file path)
  if (process.argv.length < 3) {
    console.error('Usage: node questions.js <input_file.json>');
    process.exit(1);
  }

  const inputFile = process.argv[2];
  
  try {
    // Read and parse input parameters
    const params = JSON.parse(fs.readFileSync(inputFile, 'utf8'));
    
    // Generate questions
    const questions = await generateQuestions(params);
    
    // Output questions as JSON
    console.log(JSON.stringify(questions));
    
    // Also write to output file
    fs.writeFileSync(`${inputFile}_output.json`, JSON.stringify(questions, null, 2));
  } catch (error) {
    console.error(`Error: ${error.message}`);
    process.exit(1);
  }
}

// Run the main function
main();
"""
        
        # Write the script
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        print(f"Created questions.js script at {script_path}")
        
        # Also create a package.json file if it doesn't exist
        package_path = os.path.join(os.path.dirname(script_path), 'package.json')
        if not os.path.exists(package_path):
            package_content = '''{
  "name": "question-generator",
  "version": "1.0.0",
  "description": "Question generator for RL interview system",
  "main": "questions.js",
  "dependencies": {
    "@google/generative-ai": "^0.1.3",
    "dotenv": "^16.3.1"
  }
}'''
            with open(package_path, 'w') as f:
                f.write(package_content)
            print(f"Created package.json at {package_path}")
    
    def evaluate_answer(
        self, 
        question: str, 
        answer: str, 
        topic: str
    ) -> float:
        """
        Evaluate an answer to a question.
        
        Args:
            question: The question text
            answer: The user's answer text
            topic: The topic of the question
            
        Returns:
            Score for the answer (0-10)
        """
        # If Python implementation is available, use it
        if self.python_generator:
            try:
                return self.python_generator.evaluate_answer(
                    topic=topic,
                    question=question,
                    answer=answer
                )
            except Exception as e:
                print(f"Error using Python QuestionGenerator for evaluation: {e}")
                print("Falling back to JavaScript implementation")
        
        # Fall back to JavaScript implementation if Python version fails or is not available
        # Create a temporary input file with the parameters
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            input_file = f.name
            json.dump({
                'question': question,
                'answer': answer,
                'topic': topic
            }, f)
        
        try:
            # Build the command to run the evaluation script
            # Assumes there's an evaluate.js script in the QuestionGenerator directory
            script_path = os.path.join(self.question_generator_path, 'evaluate.js')
            
            # Check if we need to create this script
            if not os.path.exists(script_path):
                self._create_evaluation_script(script_path)
            
            # Run the evaluator
            cmd = ['node', script_path, input_file]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Parse the output
            try:
                evaluation = json.loads(result.stdout)
                return float(evaluation.get('score', 0))
            except (json.JSONDecodeError, ValueError):
                print(f"Warning: Failed to parse score from stdout: {result.stdout}")
                # Try to read from output file if stdout parsing fails
                output_file = f"{input_file}_output.json"
                if os.path.exists(output_file):
                    with open(output_file, 'r') as f:
                        evaluation = json.load(f)
                        return float(evaluation.get('score', 0))
                else:
                    print("Failed to get evaluation score from generator")
                    return 5.0  # Default middle score
        
        except subprocess.CalledProcessError as e:
            print(f"Error running answer evaluation: {e}")
            print(f"Stdout: {e.stdout}")
            print(f"Stderr: {e.stderr}")
            return 5.0  # Default middle score
        
        finally:
            # Clean up temp file
            if os.path.exists(input_file):
                os.unlink(input_file)
    
    def _create_evaluation_script(self, script_path: str):
        """
        Create a simple script to evaluate answers if one doesn't exist.
        
        Args:
            script_path: Path to create the script
        """
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(script_path), exist_ok=True)
        
        script_content = """
// evaluate.js - Answer Evaluation Script for the RL Agent
// This script provides a programmatic interface to evaluate answers.

const fs = require('fs');
const path = require('path');
require('dotenv').config();

// Load Google Generative AI library if available
let GoogleGenerativeAI;
try {
  GoogleGenerativeAI = require('@google/generative-ai');
} catch (error) {
  console.error('Error: @google/generative-ai package not found.');
  console.error('Please install it with: npm install @google/generative-ai');
  process.exit(1);
}

// Check for API key
const apiKey = process.env.GEMINI_API_KEY;
if (!apiKey) {
  console.error('Error: GEMINI_API_KEY environment variable not set.');
  process.exit(1);
}

// Function to evaluate an answer
async function evaluateAnswer(params) {
  const { question, answer, topic } = params;

  // Configure the model
  const genAI = new GoogleGenerativeAI(apiKey);
  const model = genAI.getGenerativeModel({ model: 'gemini-1.5-flash-latest' });
  const chatSession = model.startChat();

  // Create a prompt for evaluation
  const prompt = `
  You are an expert technical evaluator for the topic '${topic}'.

  The specific question was:
  "${question}"

  The provided answer is:
  "${answer}"

  **Evaluation Task:**
  1. Identify key technical elements based on the question.
  2. Analyze the answer for accuracy and use of key technical elements.
  3. Score based on technical accuracy between 0.0 and 10.0, allowing decimal values.
     * High (8.0-10.0): Strong understanding, accurate terms/concepts.
     * Medium (5.0-7.9): Partial understanding, some correct elements but gaps.
     * Low (0.0-4.9): Misses main technical points, incorrect/irrelevant.

  **Output Format:**
  Respond ONLY with a valid JSON object containing the score, like this:
  {"score": "8.5"}
  Do not include any other text, just the JSON object.
  `;

  try {
    // Send the prompt to the model
    const result = await chatSession.sendMessage(prompt);
    const responseText = result.response.text();
    
    // Extract JSON from the response
    const jsonMatch = responseText.match(/\\{.*\\}/s);
    if (jsonMatch) {
      const jsonText = jsonMatch[0];
      return JSON.parse(jsonText);
    }
    
    // Fallback: try to parse the entire response as JSON
    try {
      return JSON.parse(responseText);
    } catch (error) {
      console.error("Failed to extract valid JSON from response.");
      console.error("Raw response:", responseText);
      return { score: "5.0" };
    }
  } catch (error) {
    console.error(`Error evaluating answer: ${error.message}`);
    return { score: "5.0" };
  }
}

// Main function
async function main() {
  // Read input parameters from command line argument (JSON file path)
  if (process.argv.length < 3) {
    console.error('Usage: node evaluate.js <input_file.json>');
    process.exit(1);
  }

  const inputFile = process.argv[2];
  
  try {
    // Read and parse input parameters
    const params = JSON.parse(fs.readFileSync(inputFile, 'utf8'));
    
    // Validate required parameters
    if (!params.question || !params.answer) {
      throw new Error('Missing required parameters: question and answer');
    }
    
    // Default topic if not provided
    if (!params.topic) {
      params.topic = 'technical interview';
    }
    
    // Evaluate answer
    const evaluation = await evaluateAnswer(params);
    
    // Output evaluation as JSON
    console.log(JSON.stringify(evaluation));
    
    // Also write to output file
    fs.writeFileSync(`${inputFile}_output.json`, JSON.stringify(evaluation, null, 2));
  } catch (error) {
    console.error(`Error: ${error.message}`);
    process.exit(1);
  }
}

// Run the main function
main();
"""
        
        # Write the script
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        print(f"Created evaluate.js script at {script_path}")


class SystemConnector:
    """
    Main connector that combines all system components.
    """
    
    def __init__(self):
        """Initialize the system connector with all component connectors."""
        self.resume_connector = ResumeExtractorConnector()
        self.question_connector = QuestionGeneratorConnector()
    
    def parse_resume(self, resume_path: str) -> Dict[str, Any]:
        """
        Parse a resume and return structured data.
        
        Args:
            resume_path: Path to the resume file
            
        Returns:
            Structured resume data
        """
        return self.resume_connector.extract_resume_data(resume_path)
    
    def generate_topic_from_resume(self, resume_data: Dict[str, Any]) -> str:
        """
        Generate a relevant topic based on resume data.
        
        Args:
            resume_data: Structured resume data
            
        Returns:
            A relevant technical topic
        """
        # Extract technologies from resume
        technologies = []
        
        # Extract from projects
        for project in resume_data.get('projects', []):
            if 'technologies' in project and isinstance(project['technologies'], list):
                technologies.extend(project['technologies'])
        
        # Look in education degrees
        for edu in resume_data.get('education', []):
            degree = edu.get('degree', '').lower()
            if 'computer science' in degree:
                technologies.append('computer science')
            elif 'data science' in degree:
                technologies.append('data science')
            elif 'machine learning' in degree:
                technologies.append('machine learning')
        
        # Find most common technology or return a default
        if technologies:
            # Simple frequency count
            tech_count = {}
            for tech in technologies:
                tech_count[tech] = tech_count.get(tech, 0) + 1
            
            # Return the most common technology
            return max(tech_count.items(), key=lambda x: x[1])[0]
        
        # Default topic if no technologies found
        return "programming"
    
    def generate_questions(
        self, 
        resume_data: Dict[str, Any],
        topic: str = None,
        difficulty: float = 5.0,
        num_questions: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Generate relevant questions based on resume and parameters.
        
        Args:
            resume_data: Structured resume data
            topic: Optional topic override
            difficulty: Target difficulty level (0-10)
            num_questions: Number of questions to generate
            
        Returns:
            List of question dictionaries
        """
        # If topic not provided, derive from resume
        if topic is None:
            topic = self.generate_topic_from_resume(resume_data)
        
        # Generate questions using custom prompt that incorporates resume details
        technologies = []
        for project in resume_data.get('projects', []):
            if 'technologies' in project and isinstance(project['technologies'], list):
                technologies.extend(project['technologies'])
        
        experience = []
        for exp in resume_data.get('work_experience', []):
            if 'description' in exp:
                experience.append(exp['description'])
        
        # Create a custom prompt that incorporates resume information
        custom_prompt = f"""
        Generate exactly {num_questions} technical questions about '{topic}' with difficulty {difficulty:.1f}/10.
        
        Consider this candidate's background:
        - Technologies: {', '.join(technologies) if technologies else 'Not specified'}
        - Experience: {' '.join(experience[:2]) if experience else 'Not specified'}
        
        For each question:
        1. Make it relevant to their background but still challenging
        2. Provide the question text
        3. Estimate its difficulty (0.0-10.0)
        4. Assign a sequential number
        
        Format as a JSON list:
        [
            {{"question": "...", "difficulty": "...", "questionNo": "1"}},
            {{"question": "...", "difficulty": "...", "questionNo": "2"}},
            ...
        ]
        Return ONLY the JSON list.
        """
        
        return self.question_connector.generate_questions(
            topic, 
            difficulty,
            num_questions,
            custom_prompt
        )
    
    def evaluate_answer(
        self, 
        question: str,
        answer: str,
        topic: str = None,
        resume_data: Dict[str, Any] = None
    ) -> float:
        """
        Evaluate an answer to a question.
        
        Args:
            question: The question text
            answer: The user's answer
            topic: The topic (optional)
            resume_data: Structured resume data (optional)
            
        Returns:
            Score for the answer (0-10)
        """
        # If topic not provided, derive from resume if available
        if topic is None and resume_data is not None:
            topic = self.generate_topic_from_resume(resume_data)
        elif topic is None:
            topic = "technical interview"
        
        return self.question_connector.evaluate_answer(question, answer, topic) 