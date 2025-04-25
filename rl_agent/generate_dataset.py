import os
import json
import random
import google.generativeai as genai
from typing import List, Dict
import time
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Please set GEMINI_API_KEY environment variable")

logging.info("Configuring Gemini API...")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash-latest')
logging.info("Gemini API configured successfully")

# Topics and their weights for question distribution
TOPICS = {
    "algorithms": ["sorting", "searching", "dynamic programming", "greedy algorithms"],
    "data_structures": ["arrays", "linked lists", "trees", "graphs", "hash tables"],
    "system_design": ["scalability", "databases", "microservices", "distributed systems"],
    "databases": ["SQL", "indexing", "transactions", "optimization"],
    "operating_systems": ["processes", "threads", "memory management", "file systems"]
}

class QuestionGenerator:
    def __init__(self):
        self.generated_questions = set()  # Track generated questions to avoid duplicates
        self.output_file = "interview_questions.json"
        self.total_questions = 0
        self.successful_generations = 0
        self.failed_generations = 0
        
    def generate_system_prompt(self, topic: str, difficulty: int) -> str:
        """Generate a detailed prompt for Gemini to create interview questions."""
        return f"""You are an expert technical interviewer at a top tech company. Generate a realistic technical interview question with the following specifications:

Topic: {topic}
Difficulty Level: {difficulty}/10 (where 1 is entry-level and 10 is extremely challenging)

Important Guidelines:
1. Create a question that feels natural and conversational, like a real interviewer would ask
2. The difficulty should be appropriate for the level specified
3. Include practical, real-world scenarios when possible
4. Make the question clear and unambiguous
5. Include:
   - The main question/problem statement
   - Expected time to solve
   - Key concepts being tested
   - Sample input/output if applicable
   - 2-3 follow-up questions that build upon the main question
   - Any constraints or assumptions that should be considered

Format your response in a clear, structured way but avoid using rigid templates. Make it feel like a natural conversation while ensuring all necessary information is included.

Remember:
- For lower difficulties (1-3): Focus on fundamental concepts and straightforward applications
- For medium difficulties (4-7): Introduce more complex scenarios and edge cases
- For high difficulties (8-10): Include system design considerations, scalability, and advanced optimizations

The question should be challenging but fair for the specified difficulty level."""

    async def generate_question(self, topic: str, difficulty: int) -> Dict:
        """Generate a single interview question using Gemini."""
        self.total_questions += 1
        logging.info(f"Generating question {self.total_questions} - Topic: {topic}, Difficulty: {difficulty}")
        
        prompt = self.generate_system_prompt(topic, difficulty)
        
        try:
            start_time = time.time()
            response = await model.generate_content_async(prompt)
            generation_time = time.time() - start_time
            
            # Generate a unique ID for the question
            question_id = f"{topic.lower().replace(' ', '_')}_{difficulty:02d}_{random.randint(100, 999)}"
            
            # Parse and structure the response
            question_data = {
                "id": question_id,
                "topic": topic,
                "difficulty": difficulty,
                "content": response.text,
                "metadata": {
                    "generated_timestamp": time.time(),
                    "model": "gemini-pro",
                    "generation_time": generation_time
                }
            }
            
            # Check if this question is too similar to existing ones
            question_hash = hash(response.text)
            if question_hash in self.generated_questions:
                logging.warning(f"Duplicate question detected for {topic} (difficulty {difficulty})")
                self.failed_generations += 1
                return None
            
            self.generated_questions.add(question_hash)
            self.successful_generations += 1
            logging.info(f"Successfully generated question {self.successful_generations} in {generation_time:.2f} seconds")
            return question_data
            
        except Exception as e:
            self.failed_generations += 1
            logging.error(f"Error generating question for {topic} (difficulty {difficulty}): {str(e)}")
            return None

    async def generate_dataset(self, questions_per_topic_per_difficulty: int = 2) -> None:
        """Generate the complete dataset of interview questions."""
        dataset = []
        start_time = time.time()
        
        # Calculate total expected questions
        total_topics = sum(len(subtopics) for subtopics in TOPICS.values())
        total_expected = total_topics * 10 * questions_per_topic_per_difficulty
        logging.info(f"Starting dataset generation. Expected questions: {total_expected}")
        
        for topic_category, subtopics in TOPICS.items():
            logging.info(f"\nProcessing topic category: {topic_category}")
            for subtopic in subtopics:
                logging.info(f"Processing subtopic: {subtopic}")
                for difficulty in range(1, 11):
                    logging.info(f"Generating {questions_per_topic_per_difficulty} questions for difficulty {difficulty}")
                    for _ in range(questions_per_topic_per_difficulty):
                        question = await self.generate_question(subtopic, difficulty)
                        if question:
                            dataset.append(question)
                        # Add delay to respect API rate limits
                        await asyncio.sleep(1)
                        
                    # Log progress after each difficulty level
                    progress = (len(dataset) / total_expected) * 100
                    logging.info(f"Progress: {progress:.1f}% ({len(dataset)}/{total_expected} questions)")
        
        # Save the dataset
        with open(self.output_file, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        total_time = time.time() - start_time
        logging.info("\nGeneration Summary:")
        logging.info(f"Total questions attempted: {self.total_questions}")
        logging.info(f"Successfully generated: {self.successful_generations}")
        logging.info(f"Failed generations: {self.failed_generations}")
        logging.info(f"Total time taken: {total_time:.2f} seconds")
        logging.info(f"Average time per question: {total_time/len(dataset):.2f} seconds")
        logging.info(f"Dataset saved to {self.output_file}")

if __name__ == "__main__":
    import asyncio
    
    logging.info("Starting question generation script...")
    generator = QuestionGenerator()
    asyncio.run(generator.generate_dataset()) 