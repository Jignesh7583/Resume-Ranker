from typing import Tuple, List
from models import JobDescription, CandidateProfile, RankedCandidate
from embedding_engine import EmbeddingEngine

class ScoringEngine:
    def __init__(self, embedding_engine: EmbeddingEngine):
        self.embedding_engine = embedding_engine

    def _score_technical(self, jd: JobDescription, candidate: CandidateProfile) -> Tuple[float, str]:
        """
        Calculates Technical Capability Fit (50%).

        Improvements applied:
        - Improvement 1: Recency Weighting — recent jobs in career_history contribute more.
        - Improvement 2: Coverage Threshold — candidate must semantically clear a minimum bar
                         for each required skill, not just have one vague match.
        - Improvement 3: Role Title Matching — candidate's current job title is compared to
                         the JD title and blended into the final technical score.
        - Improvement 4: Skill Frequency Bonus — if a skill appears in multiple past jobs,
                         a small bonus is awarded to reflect genuine depth.
        """
        # --- Core Advanced Semantic Score (Improvements 1, 2, 4) ---
        adv_score, coverage_pct, freq_bonus = self.embedding_engine.advanced_semantic_match_score(
            required_skills=jd.required_skills,
            career_history=candidate.career_history,
            fallback_skills=candidate.skills
        )

        # --- Improvement 3: Role Title Matching (20% blend into technical score) ---
        title_score = self.embedding_engine.title_match_score(jd.title, candidate.current_role)

        # Blend: 80% semantic capability + 20% title relevance
        final_tech_score = (adv_score * 0.80) + (title_score * 0.20)

        # Build a transparent reason string
        coverage_label = f"{int(coverage_pct * 100)}% skill coverage"
        bonus_note = " [Depth bonus applied]" if freq_bonus else ""
        reason = (
            f"Semantic Capability Match is {final_tech_score:.2f} "
            f"({coverage_label}, title match: {title_score:.2f}){bonus_note}."
        )

        return final_tech_score, reason

    def _score_experience(self, jd: JobDescription, candidate: CandidateProfile) -> Tuple[float, str]:
        """
        Calculates Experience Fit (20%)
        """
        exp = candidate.experience_years
        
        if exp >= jd.min_experience_years and exp <= jd.max_experience_years:
            score = 1.0
            reason = f"{exp} years of experience fits the requested {jd.min_experience_years}-{jd.max_experience_years} years."
        elif exp < jd.min_experience_years:
            # Penalize linearly based on how short they are
            shortfall = jd.min_experience_years - exp
            score = max(0.0, 1.0 - (shortfall / jd.min_experience_years))
            reason = f"{exp} years of experience is slightly below the minimum {jd.min_experience_years} years."
        else:
            # Overqualified
            score = 0.8 
            reason = f"{exp} years of experience is above the max {jd.max_experience_years} years (overqualified)."
            
        return score, reason

    def _score_behavioral(self, candidate: CandidateProfile) -> Tuple[float, str]:
        """
        Calculates Behavioral Fit (25%) using Redrob signals.
        """
        signals = candidate.redrob_signals
        score = 0.0
        reasons = []
        
        if signals.open_to_work:
            score += 0.3
            reasons.append("Open to work")
        
        # Last active (recent is better)
        if signals.last_active_days_ago < 7:
            score += 0.2
            reasons.append("Highly active recently")
        elif signals.last_active_days_ago < 30:
            score += 0.1
            
        # Response rate
        if signals.recruiter_response_rate > 0.8:
            score += 0.3
            reasons.append("High response rate")
            
        # Add completeness and other signals
        score += (signals.profile_completeness * 0.2)
        
        final_score = min(1.0, score)
        return final_score, "Behavioral signals: " + ", ".join(reasons) if reasons else "Average behavioral signals."

    def _score_location(self, jd: JobDescription, candidate: CandidateProfile) -> Tuple[float, str]:
        """
        Calculates Location/Preference Fit (5%)
        """
        if not jd.location:
            return 1.0, "No location constraint."
            
        if candidate.location and jd.location.lower() in candidate.location.lower():
            return 1.0, "Location matches."
            
        if candidate.redrob_signals.relocation_willingness:
            return 0.8, "Location differs but willing to relocate."
            
        return 0.0, "Location mismatch and no relocation willingness."

    def score_candidate(self, jd: JobDescription, candidate: CandidateProfile) -> RankedCandidate:
        tech_score, tech_reason = self._score_technical(jd, candidate)
        exp_score, exp_reason = self._score_experience(jd, candidate)
        beh_score, beh_reason = self._score_behavioral(candidate)
        loc_score, loc_reason = self._score_location(jd, candidate)
        
        # Weighted Final Score
        final_score = (tech_score * 0.50) + (exp_score * 0.20) + (beh_score * 0.25) + (loc_score * 0.05)
        
        # Generate summary reasoning
        summary_reasoning = f"{candidate.current_role or 'Candidate'} with {candidate.experience_years} years exp. "
        summary_reasoning += tech_reason + " " + beh_reason
        
        # Apply strict filters/penalties as per JD rules
        career_text = " ".join([c.get('description', '') for c in candidate.career_history]).lower()
        company_text = " ".join([c.get('company', '') for c in candidate.career_history]).lower()
        
        # Pure Research penalty
        if 'research' in career_text and 'production' not in career_text and 'deploy' not in career_text:
            final_score *= 0.5
            summary_reasoning += " [Penalty: Research-only profile]"
            
        # Services Only penalty
        services_firms = ["tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini", "mindtree"]
        if any(firm in company_text for firm in services_firms):
            # Check if there are non-services companies
            non_services = False
            for c in candidate.career_history:
                comp = c.get('company', '').lower()
                if not any(f in comp for f in services_firms):
                    non_services = True
                    break
            if not non_services:
                final_score *= 0.4
                summary_reasoning += " [Penalty: Services-only profile]"
                
        # Title Chaser penalty (Average tenure < 18 months)
        if candidate.career_history and len(candidate.career_history) > 2:
            total_months = 0
            for c in candidate.career_history:
                duration = c.get('duration_months', 0)
                if duration:
                    total_months += duration
            avg_tenure = total_months / len(candidate.career_history)
            if 0 < avg_tenure < 18:
                final_score *= 0.6
                summary_reasoning += " [Penalty: Title chaser / Job hopper]"
                
        # Honeypot heuristic (e.g. lots of skills but 0 experience)
        if candidate.experience_years <= 0.5 and len(candidate.skills) > 10:
            final_score *= 0.1
            summary_reasoning += " [Penalty: Potential Honeypot]"

        return RankedCandidate(
            candidate_id=candidate.candidate_id,
            rank=0, # Will be set later
            score=round(final_score, 4),
            reasoning=summary_reasoning
        )
