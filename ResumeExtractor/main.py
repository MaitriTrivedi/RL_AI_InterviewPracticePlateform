import re
import json
import sys
import ollama
import PyPDF2
from typing import List, Dict, Any
import logging

class ResumeParser:
    def __init__(self, model_name='mistral'):
        self.model_name = model_name
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(levelname)s: %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _extract_text_from_pdf(self, file_path: str) -> str:
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ''
                for page in reader.pages:
                    text += page.extract_text()
                
                # Clean text
                text = re.sub(r'\s+', ' ', text).strip()
                return text
        except Exception as e:
            self.logger.error(f"PDF Text Extraction Error: {e}")
            raise ValueError(f"Could not extract text from PDF: {e}")
    
    def _identify_sections(self, text: str) -> Dict[str, int]:
        section_patterns = {
            'education': [
                r'education', 
                r'academic\s*background', 
                r'academic\s*qualifications'
            ],
            'work_experience': [
                r'work\s*experience', 
                r'professional\s*experience', 
                r'employment\s*history'
            ],
            'projects': [
                r'projects', 
                r'project\s*experience', 
                r'academic\s*projects'
            ]
        }
        
        sections = {}
        for section_name, patterns in section_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    sections[section_name] = match.start()
                    break
        
        return sections
    
    def _extract_section_content(self, text: str, sections: Dict[str, int]) -> Dict[str, str]:
        section_contents = {}
        sorted_sections = sorted(sections.items(), key=lambda x: x[1])
        
        for i, (section, start_index) in enumerate(sorted_sections):
            end_index = sorted_sections[i+1][1] if i+1 < len(sorted_sections) else len(text)
            section_contents[section] = text[start_index:end_index].strip()
        
        return section_contents
    
    def _llm_extract(self, section_text: str, section_type: str) -> List[Dict[str, Any]]:
        # Check if section text is empty or contains only whitespace
        if not section_text or section_text.isspace():
            self.logger.info(f"No {section_type} section found. Skipping extraction.")
            return []
        
        print(f"\n🔍 Extracting {section_type.upper()} Section\n{'='*40}\n{section_text}\n{'='*40}\n")

        # Output templates for different section types
        output_templates = {
            'education': '''[
    {
        "institution": "Full name of educational institution",
        "degree": "Degree name and major",
        "year": "Duration of study",
        "gpa": "GPA or percentage (if available)"
    }
]''',
            'projects': '''[
    {
        "name": "Project name",
        "description": "Detailed project description",
        "technologies": ["Technology 1", "Technology 2"]
    }
]''',
            'work_experience': '''[
    {
        "company": "Company name",
        "year": "Duration of employment",
        "description": "Job responsibilities and key achievements"
    }
]'''
        }

        prompt = f'''
You are a resume parsing expert. ONLY extract structured information from the following {section_type} section.
Return a valid JSON array with no explanation.
Strictly use only the information provided.
Use double quotes for all keys and string values.


Text: 
"""
{section_text}
"""

Return JSON Format:
{output_templates.get(section_type, '[]')}
Only return JSON, no extra text.
'''
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt}]
            )
            content = response['message']['content']

            # Clean and extract JSON array
            content = re.sub(r'```(?:json)?|```', '', content).strip()
            json_pattern = re.search(r'(\[.*\])', content, re.DOTALL)
            if json_pattern:
                content = json_pattern.group(1)

            content = re.sub(r'"|"', '"', content)  # Replace curly quotes
            content = re.sub(r',\s*([}\]])', r'\1', content)  # Remove trailing commas

            return json.loads(content)
        except Exception as e:
            self.logger.error(f"LLM Extraction Error for {section_type}: {e}")
            return []
    
    def parse_resume(self, file_path: str) -> Dict[str, List[Dict[str, Any]]]:
        # Extract text from PDF
        resume_text = self._extract_text_from_pdf(file_path)
        
        # Identify sections
        sections = self._identify_sections(resume_text)
        section_contents = self._extract_section_content(resume_text, sections)
        
        # Extract structured information
        parsed_resume = {
            'education': self._llm_extract(section_contents.get('education', ''), 'education'),
            'projects': self._llm_extract(section_contents.get('projects', ''), 'projects'),
            'work_experience': self._llm_extract(section_contents.get('work_experience', ''), 'work_experience')
        }
        
        # Save parsed resume
        output_file = f"{file_path.split('.')[0]}_parsed.json"
        with open(output_file, 'w') as f:
            json.dump(parsed_resume, f, indent=4)
        
        self.logger.info(f"Resume parsed and saved to {output_file}")
        return parsed_resume

def main():
    if len(sys.argv) < 2:
        print("Usage: python resume_parser.py <path_to_resume.pdf>")
        sys.exit(1)
    
    resume_path = sys.argv[1]
    parser = ResumeParser()
    
    try:
        parsed_resume = parser.parse_resume(resume_path)
        print(json.dumps(parsed_resume, indent=2))
    except Exception as e:
        print(f"Error parsing resume: {e}")

if __name__ == "__main__":
    main()