from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class JobDescription(BaseModel):
    title: str
    required_skills: List[str]
    preferred_skills: List[str] = []
    min_experience_years: float = 0.0
    max_experience_years: float = 99.0
    responsibilities: List[str] = []
    location: Optional[str] = None
    raw_text: Optional[str] = None

class RedrobSignals(BaseModel):
    profile_completeness: float = 0.0 # 0.0 to 1.0
    last_active_days_ago: int = 999
    open_to_work: bool = False
    recruiter_response_rate: float = 0.0 # 0.0 to 1.0
    response_speed_hours: float = 999.0
    assessment_scores: Dict[str, float] = {}
    recruiter_saves: int = 0
    interview_completion_rate: float = 0.0 # 0.0 to 1.0
    offer_acceptance_rate: float = 0.0 # 0.0 to 1.0
    verification_status: bool = False
    github_activity_score: float = 0.0 # 0.0 to 1.0
    linkedin_connections: int = 0
    relocation_willingness: bool = False

class CandidateProfile(BaseModel):
    candidate_id: str
    current_role: Optional[str] = None
    company: Optional[str] = None
    industry: Optional[str] = None
    experience_years: float = 0.0
    location: Optional[str] = None
    skills: List[str] = []
    career_history: List[Dict[str, Any]] = [] # [{role, company, duration, description}]
    education: List[Dict[str, Any]] = []
    certifications: List[str] = []
    languages: List[str] = []
    redrob_signals: RedrobSignals = Field(default_factory=RedrobSignals)

class RankedCandidate(BaseModel):
    candidate_id: str
    rank: int
    score: float
    reasoning: str
