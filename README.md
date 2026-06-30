# Intelligent Candidate Ranking System (Team ScoutIQ)

This repository contains the source code for our intelligent semantic ranking engine, built for the Redrob Hackathon.

## Approach

Our ranking system uses a **Pure Semantic Capability Matching** approach rather than keyword counting. It evaluates four core aspects of a candidate:

1. **Technical Capability (50%)**: The most advanced component, combining four improvements:
   - **Recency Weighting**: The most recent jobs in a candidate's career contribute more to the score. A skill used last year counts more than one used 6 years ago.
   - **Coverage Threshold**: Each required JD skill must pass a minimum semantic similarity bar (0.38). This prevents a candidate from scoring high just because they vaguely match one or two skills.
   - **Role Title Matching**: The candidate's current job title is semantically compared to the JD title and blended in (20% weight), as a title like "Senior ML Engineer" is a strong relevance signal.
   - **Skill Frequency Bonus**: If a required skill appears across **multiple past jobs**, an 8% score bonus is awarded to reflect genuine, demonstrated depth of expertise.

2. **Experience (20%)**: Distance metric to the target 5-9 years range.

3. **Behavioral Signals (25%)**: Scoring based on `recruiter_response_rate`, `last_active_days_ago`, and `open_to_work` status.

4. **Location (5%)**: Checks candidate location vs job requirements.

It also employs strict business-logic penalties for Job Hoppers, "IT Services Only" traps, and Honeypot candidates by analyzing specific `company` fields and `duration` metadata mathematically.

## Setup Instructions
Ensure you are using a CPU machine with at least 16GB of RAM.

1. Clone this repository.
2. Install the requirements:
```bash
pip install -r requirements.txt
```
*(Note: If you encounter an unauthenticated HuggingFace warning during model load, it is safe to ignore as the `all-MiniLM-L6-v2` model is public).*

## How to Run

To reproduce the submission exactly from the candidates file, run the following single command:

```bash
python ranking_pipeline.py --jd indiarun/indiarun/job_description.txt --candidates candidates.jsonl
```

### Pre-computation Note:
To fit within the 5-minute compute budget on a 100K candidate dataset, our pipeline utilizes **Batch Pre-computation**. When you run the command above, it will first gather all unique career descriptions and batch-encode them using the sentence-transformer model. You will see a progress bar for this step. Once cached, the scoring engine calculates the final scores and generates `team_scoutiq.csv` well within the time limit.
