"""
Prompt template for STEP 3: Fit Score Generation.

Scoring weights:
  - Skill Match      : 50%
  - Experience Match : 30%
  - Tools Match      : 20%

Score range: 0 – 100
"""

from langchain_core.prompts import PromptTemplate

# ──────────────────────────────────────────────────────────────
# Few-shot scoring examples
# ──────────────────────────────────────────────────────────────

FEW_SHOT_EXAMPLES = """
=== EXAMPLE (CORRECT) ===
Match Data:
  matched_skills   : ["Python", "REST APIs"]
  missing_skills   : ["FastAPI", "PostgreSQL"]
  matched_tools    : ["AWS EC2"]
  missing_tools    : ["Docker"]
  match_percentage : 50

Extracted Experience : "3 years"
Required Experience  : "5+ years"

Scoring Calculation:
  skill_score      = (2 matched / 4 required) * 50 = 25
  tool_score       = (1 matched / 2 required) * 20 = 10
  experience_score = candidate has 3 yrs, needs 5+ yrs => partial = 18
  TOTAL            = 25 + 10 + 18 = 53

Output:
{{
  "score": 53,
  "skill_score": 25,
  "tool_score": 10,
  "experience_score": 18
}}
"""

# ──────────────────────────────────────────────────────────────
# Scoring Prompt Template
# ──────────────────────────────────────────────────────────────

SCORE_TEMPLATE = """
You are a precise hiring scorecard evaluator.

Scoring weights are FIXED:
  - Skill Match      : 50 points maximum
  - Experience Match : 30 points maximum
  - Tools Match      : 20 points maximum
  TOTAL POSSIBLE     : 100 points

{few_shot_examples}

=== MATCH DATA ===
Matched Skills    : {matched_skills}
Missing Skills    : {missing_skills}
Matched Tools     : {matched_tools}
Missing Tools     : {missing_tools}
Match Percentage  : {match_percentage}%

=== EXPERIENCE ===
Candidate Experience : {candidate_experience}
Required Experience  : {required_experience}

=== SCORING INSTRUCTIONS ===
1. skill_score      = (len(matched_skills) / (len(matched_skills) + len(missing_skills))) * 50
                      Round to nearest integer. If no skills required, give full 50.
2. tool_score       = (len(matched_tools) / (len(matched_tools) + len(missing_tools))) * 20
                      Round to nearest integer. If no tools required, give full 20.
3. experience_score = Evaluate how well candidate experience matches required experience.
                      Full match = 30, partial = proportional, no match = 0–5.
4. score            = skill_score + tool_score + experience_score  (max 100)

STRICT RULE: Only use the provided data. No external assumptions.

=== OUTPUT FORMAT (strict JSON only) ===
{{
  "score": <integer 0-100>,
  "skill_score": <integer 0-50>,
  "tool_score": <integer 0-20>,
  "experience_score": <integer 0-30>
}}

JSON Output:
"""

def get_score_prompt() -> PromptTemplate:
    """Return the scoring prompt with few-shot examples pre-filled."""
    return PromptTemplate(
        input_variables=[
            "matched_skills",
            "missing_skills",
            "matched_tools",
            "missing_tools",
            "match_percentage",
            "candidate_experience",
            "required_experience",
        ],
        template=SCORE_TEMPLATE.replace("{few_shot_examples}", FEW_SHOT_EXAMPLES),
    )
