import argparse
import pandas as pd
from tqdm import tqdm

from jd_extractor import JDExtractor
from candidate_parser import CandidateParser
from embedding_engine import EmbeddingEngine
from scoring_engine import ScoringEngine

def main():
    parser = argparse.ArgumentParser(description="Intelligent Candidate Discovery & Ranking System")
    parser.add_argument("--jd", type=str, required=True, help="Path to Job Description txt file")
    parser.add_argument("--candidates", type=str, required=True, help="Path to Candidates JSON/JSONL file")
    parser.add_argument("--output", type=str, default="team_scoutiq.csv", help="Output CSV path")
    parser.add_argument("--top_k", type=int, default=100, help="Number of top candidates to output")
    
    args = parser.parse_args()
    
    print(f"Reading JD from {args.jd}...")
    with open(args.jd, 'r', encoding='utf-8') as f:
        jd_text = f.read()
        
    jd_extractor = JDExtractor()
    jd = jd_extractor.extract(jd_text)
    
    print(f"Parsed JD: Role = {jd.title}, Required Skills = {jd.required_skills}, Exp = {jd.min_experience_years}-{jd.max_experience_years}")
    
    print(f"Loading candidates from {args.candidates}...")
    candidate_parser = CandidateParser()
    candidates = candidate_parser.parse_candidates(args.candidates)
    print(f"Loaded {len(candidates)} candidates.")
    
    print("Initializing AI models (Embedding Engine)...")
    embedding_engine = EmbeddingEngine()
    scoring_engine = ScoringEngine(embedding_engine)
    
    print("Pre-computing embeddings for all candidates (Batch Mode)...")
    all_texts_to_embed = set()
    all_texts_to_embed.update(jd.required_skills)
    
    for cand in candidates:
        career_descriptions = [c.get('description', '') for c in cand.career_history if c.get('description')]
        target_texts = career_descriptions if career_descriptions else cand.skills
        all_texts_to_embed.update(target_texts)
        
    # This will hit the cache in the embedding engine for all future calls
    embedding_engine.get_embeddings(list(all_texts_to_embed))
    print(f"Pre-computation complete for {len(all_texts_to_embed)} unique text blocks.")
    
    print("Scoring candidates...")
    ranked_candidates = []
    for cand in tqdm(candidates):
        ranked_cand = scoring_engine.score_candidate(jd, cand)
        ranked_candidates.append(ranked_cand)
        
    # Sort by score descending, then candidate_id ascending (tie break)
    ranked_candidates.sort(key=lambda x: (-x.score, x.candidate_id))
    
    # Assign ranks
    for i, rc in enumerate(ranked_candidates):
        rc.rank = i + 1
        
    # Output top K
    top_candidates = ranked_candidates[:args.top_k]
    
    # Export to CSV
    output_data = [
        {
            "candidate_id": rc.candidate_id,
            "rank": rc.rank,
            "score": rc.score,
            "reasoning": rc.reasoning
        }
        for rc in top_candidates
    ]
    
    df = pd.DataFrame(output_data)
    df.to_csv(args.output, index=False)
    print(f"Ranking complete! Top {len(top_candidates)} candidates exported to {args.output}")

if __name__ == '__main__':
    main()
