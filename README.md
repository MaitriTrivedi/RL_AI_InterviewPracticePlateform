# RL_AI_InterviewPracticePlateform

An AI-powered interview practice platform using Reinforcement Learning to adaptively select questions based on your resume and performance.

## Overview

This project implements an intelligent interview practice system that:

1. Extracts skills and experience from your resume
2. Generates relevant technical questions
3. Uses Reinforcement Learning to adapt question difficulty
4. Evaluates your answers and provides scores
5. Improves over time through training

## Components

- **ResumeExtractor**: Python module to parse resumes and extract structured data
- **QuestionGenerator**: Module to generate and evaluate technical questions (available in both JavaScript and Python)
- **RLAgent**: Reinforcement learning system to select optimal questions

## Requirements

### Python Requirements
- Python 3.8+
- NumPy
- PyPDF2
- Ollama (for local LLM)
- Google's Generative AI library (`google-generativeai`)
- python-dotenv

### JavaScript Requirements (Optional)
- Node.js 14+
- npm packages: `@google/generative-ai`, `dotenv`

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/RL_AI_InterviewPracticePlateform.git
   cd RL_AI_InterviewPracticePlateform
   ```

2. Install Python dependencies:
   ```
   pip install numpy pypdf2 ollama google-generativeai python-dotenv
   ```

3. Create `.env` file with your Gemini API key:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

## Usage

### QuestionGenerator

You can use the QuestionGenerator in two ways:

#### 1. Interactive Mode

Run the interactive Q&A session:

```
python -m QuestionGenerator.generator
```

This will:
- Prompt you for a topic and difficulty level
- Generate technical questions
- Let you select and answer a question
- Evaluate your answer with a score

#### 2. Programmatic API

Use the QuestionGenerator in your Python code:

```python
from QuestionGenerator import QuestionGenerator

# Initialize the generator
generator = QuestionGenerator()

# Generate questions
questions = generator.generate_questions(
    topic="Python",
    difficulty=7.5,
    num_questions=3
)

# Evaluate an answer
score = generator.evaluate_answer(
    topic="Python",
    question="What is a decorator in Python?",
    answer="Your answer here"
)
```

### Training the RL Agent

To train the RL agent with your resume:

```
python -m RLAgent.main your_resume.pdf --mode train --episodes 100
```

This will:
1. Extract information from your resume
2. Run simulated interviews to train the agent
3. Save the trained agent to `models/agent.pkl`

### Practice Interviews

To practice interviewing with a trained agent:

```
python -m RLAgent.main your_resume.pdf
```

Additional options:
- `--topic "Python"` - Override the default topic
- `--difficulty 7.0` - Set initial difficulty (0-10)
- `--max-questions 5` - Set number of questions
- `--agent-file "my_agent.pkl"` - Use a specific agent file

## How It Works

### The RL Approach

The system uses Q-Learning, a model-free reinforcement learning algorithm:

1. **State**: Combines resume details and performance on previous questions
2. **Action**: Selects question difficulty and topics
3. **Reward**: Based on answer scores and appropriate challenge level
4. **Learning**: Updates Q-values to improve question selection

### Interview Flow

1. Your resume is parsed to extract skills, education, and experience
2. The RL agent selects appropriate question parameters
3. Multiple questions are generated and the best one is selected
4. You provide an answer, which is evaluated
5. The agent learns from your performance to select better questions
6. This repeats until the interview is complete

## Future Improvements

- DQN (Deep Q-Network) implementation for better generalization
- More sophisticated resume parsing with better NLP techniques
- Expanded question diversity and improved evaluation
- Web interface for easier interaction

## License

This project is licensed under the MIT License - see the LICENSE file for details.