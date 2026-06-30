import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class EmbeddingEngine:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initializes the local embedding model. 
        'all-MiniLM-L6-v2' is small, fast, and suitable for CPU.
        """
        print(f"Loading embedding model {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.cache = {}
        
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Returns embeddings for a list of strings with caching to speed up repeated queries.
        """
        if not texts:
            return np.array([])
            
        uncached_texts = []
        for t in texts:
            if t not in self.cache:
                uncached_texts.append(t)
                
        if uncached_texts:
            new_embeddings = self.model.encode(uncached_texts, batch_size=128, show_progress_bar=True)
            for t, emb in zip(uncached_texts, new_embeddings):
                self.cache[t] = emb
                
        return np.array([self.cache[t] for t in texts])
        
    def compute_similarity(self, query_texts: List[str], target_texts: List[str]) -> np.ndarray:
        """
        Computes cosine similarity between queries and targets.
        Returns a 2D matrix of shape (len(query_texts), len(target_texts))
        """
        if not query_texts or not target_texts:
            return np.zeros((len(query_texts), len(target_texts)))
            
        query_embeddings = self.get_embeddings(query_texts)
        target_embeddings = self.get_embeddings(target_texts)
        
        return cosine_similarity(query_embeddings, target_embeddings)

    def semantic_match_score(self, required_skills: List[str], candidate_skills: List[str]) -> float:
        """
        [LEGACY] Basic semantic match — kept for compatibility.
        Prefer using advanced_semantic_match_score for scoring.
        """
        if not required_skills:
            return 1.0
        if not candidate_skills:
            return 0.0
            
        similarity_matrix = self.compute_similarity(required_skills, candidate_skills)
        best_matches = np.max(similarity_matrix, axis=1)
        
        return float(np.mean(best_matches))

    # ===========================================================================
    # IMPROVEMENT 1: Recency-Weighted Career Description Scoring
    # ===========================================================================
    def get_recency_weighted_descriptions(self, career_history: List[Dict[str, Any]]) -> List[tuple]:
        """
        Returns a list of (description, weight) tuples.
        Jobs listed earlier in career_history are assumed more recent and get higher weight.
        Weight decays as: 1.0 -> 0.7 -> 0.5 -> 0.4 for older jobs.
        """
        recency_weights = [1.0, 0.7, 0.5, 0.4]
        weighted_descriptions = []
        
        for i, job in enumerate(career_history):
            desc = job.get('description', '').strip()
            if not desc:
                continue
            weight = recency_weights[min(i, len(recency_weights) - 1)]
            weighted_descriptions.append((desc, weight))
            
        return weighted_descriptions

    # ===========================================================================
    # IMPROVEMENT 2: Coverage Threshold Scoring
    # IMPROVEMENT 4: Skill Frequency Bonus
    # ===========================================================================
    def advanced_semantic_match_score(
        self,
        required_skills: List[str],
        career_history: List[Dict[str, Any]],
        fallback_skills: List[str],
        coverage_threshold: float = 0.38
    ) -> tuple:
        """
        Advanced scoring that combines:
        - Improvement 1: Recency weighting (recent jobs score higher)
        - Improvement 2: Coverage threshold (candidate must meet a minimum bar per skill)
        - Improvement 4: Skill frequency bonus (skills appearing in multiple jobs get boosted)

        Returns: (final_score: float, coverage_pct: float, bonus_applied: bool)
        """
        if not required_skills:
            return 1.0, 1.0, False

        # Get weighted descriptions from career history
        weighted_descs = self.get_recency_weighted_descriptions(career_history)

        if not weighted_descs:
            # Fallback to flat skills list
            if not fallback_skills:
                return 0.0, 0.0, False
            score = self.semantic_match_score(required_skills, fallback_skills)
            return score, 0.5, False

        descriptions = [d for d, _ in weighted_descs]
        weights = np.array([w for _, w in weighted_descs])

        # Compute similarity: shape (num_required_skills, num_jobs)
        similarity_matrix = self.compute_similarity(required_skills, descriptions)

        # --- Improvement 1: Recency Weighting ---
        # Multiply each job column by its recency weight, then take max across jobs
        weighted_similarity = similarity_matrix * weights  # broadcast weights across rows
        # For each required skill, pick the highest weighted-similarity match
        best_weighted_scores = np.max(weighted_similarity, axis=1)

        # --- Improvement 2: Coverage Threshold ---
        # A skill is "covered" only if the candidate scores above the threshold for it
        raw_best_scores = np.max(similarity_matrix, axis=1)  # unweighted for threshold check
        covered_mask = raw_best_scores >= coverage_threshold
        coverage_pct = float(np.mean(covered_mask))

        # Base score: mean of weighted best-match scores
        base_score = float(np.mean(best_weighted_scores))

        # Apply a coverage penalty: if coverage is low, scale down the score
        # e.g., 50% coverage → score gets multiplied by ~0.75
        coverage_multiplier = 0.5 + (0.5 * coverage_pct)
        score_after_coverage = base_score * coverage_multiplier

        # --- Improvement 4: Skill Frequency Bonus ---
        # Count how many job descriptions each required skill appears in (above threshold)
        skill_frequency_bonus = False
        frequency_multiplier = 1.0
        if len(descriptions) > 1:
            # Count how many jobs exceed threshold for each skill
            jobs_per_skill = np.sum(similarity_matrix >= coverage_threshold, axis=1)
            # If ANY required skill shows up in 2+ jobs, apply a small bonus
            if np.any(jobs_per_skill >= 2):
                frequency_multiplier = 1.08  # 8% bonus for demonstrated depth
                skill_frequency_bonus = True

        final_score = score_after_coverage * frequency_multiplier
        return min(1.0, final_score), coverage_pct, skill_frequency_bonus

    # ===========================================================================
    # IMPROVEMENT 3: Role Title Semantic Matching
    # ===========================================================================
    def title_match_score(self, jd_title: str, candidate_role: Optional[str]) -> float:
        """
        Compares the JD job title to the candidate's current role title using cosine similarity.
        Returns a score between 0.0 and 1.0.
        """
        if not candidate_role or not jd_title:
            return 0.5  # Neutral score if data is missing
            
        similarity_matrix = self.compute_similarity([jd_title], [candidate_role])
        return float(similarity_matrix[0][0])
