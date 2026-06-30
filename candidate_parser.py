import json
import os
from typing import List, Dict, Any
from models import CandidateProfile, RedrobSignals

class CandidateParser:
    def __init__(self):
        pass

    def parse_candidates(self, file_path: str) -> List[CandidateProfile]:
        """
        Loads candidates from either a JSON array or a JSONL file.
        """
        candidates = []
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data_list = json.load(f)
                for data in data_list:
                    candidates.append(self._parse_single(data))
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    candidates.append(self._parse_single(data))
                    
        return candidates

    def _parse_single(self, data: Dict[str, Any]) -> CandidateProfile:
        signals_data = data.get("redrob_signals", {})
        signals = RedrobSignals(**signals_data)
        
        profile_data = data.get("profile", {})
        
        current_role = profile_data.get("current_title") or profile_data.get("current_role")
        company = profile_data.get("current_company") or profile_data.get("company")
        industry = profile_data.get("current_industry") or profile_data.get("industry")
        
        # Support both 'years_of_experience' and 'experience_years'
        experience_years = profile_data.get("years_of_experience")
        if experience_years is None:
            experience_years = profile_data.get("experience_years", 0.0)
            
        raw_skills = data.get("skills", [])
        skills = [s.get("name", str(s)) if isinstance(s, dict) else str(s) for s in raw_skills]
        
        raw_langs = data.get("languages", [])
        languages = [l.get("language", str(l)) if isinstance(l, dict) else str(l) for l in raw_langs]
        
        raw_certs = data.get("certifications", [])
        certifications = [c.get("name", str(c)) if isinstance(c, dict) else str(c) for c in raw_certs]

        return CandidateProfile(
            candidate_id=data.get("candidate_id", ""),
            current_role=current_role,
            company=company,
            industry=industry,
            experience_years=float(experience_years),
            location=profile_data.get("location"),
            skills=skills,
            career_history=data.get("career_history", []),
            education=data.get("education", []),
            certifications=certifications,
            languages=languages,
            redrob_signals=signals
        )

if __name__ == '__main__':
    # Simple test case
    pass
