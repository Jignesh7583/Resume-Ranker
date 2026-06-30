import re
from typing import Dict, Any
from models import JobDescription

class JDExtractor:
    def __init__(self):
        pass

    def extract(self, jd_text: str) -> JobDescription:
        """
        A heuristic-based extractor for parsing a raw job description string.
        In a production system without an LLM, this would use named entity recognition
        and dependency parsing (e.g., via spaCy).
        For this prototype, we'll use regex and keyword matching.
        """
        # Parse sections based on hackathon JD structure
        lines = [line.strip() for line in jd_text.split('\n') if line.strip()]
        title = "Senior AI Engineer"
        for line in lines:
            if line.startswith("Job Description:"):
                title = line.split(":", 1)[1].strip()
                break
                
        required_skills = []
        preferred_skills = []
        in_required = False
        for line in lines:
            line_lower = line.lower()
            if "things you absolutely need" in line_lower:
                in_required = True
                continue
            elif "things we'd like you to have" in line_lower or "things we explicitly do not want" in line_lower:
                in_required = False
                continue
                
            if in_required and len(line) > 15:  # Valid requirement sentence
                required_skills.append(line)
                
        # Parse experience years
        min_exp = 5.0
        max_exp = 9.0
        exp_match = re.search(r'(\d+)[-–](\d+)\s*years?', jd_text.lower())
        if exp_match:
            min_exp = float(exp_match.group(1))
            max_exp = float(exp_match.group(2))
                
        return JobDescription(
            title=title,
            required_skills=required_skills,
            preferred_skills=preferred_skills,
            min_experience_years=min_exp,
            max_experience_years=max_exp,
            raw_text=jd_text
        )

if __name__ == '__main__':
    sample_jd = """
    Role: Senior AI Engineer
    Required:
    * Machine Learning
    * LLMs
    * Embeddings
    * Python
    
    Experience:
    5-9 years
    """
    extractor = JDExtractor()
    print(extractor.extract(sample_jd))
