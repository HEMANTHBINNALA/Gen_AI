"""
Prompt template for STEP 2: Skill Matching.

Compares extracted resume skills/tools against
skills/tools required in the job description.
"""

from langchain_core.prompts import PromptTemplate

# ──────────────────────────────────────────────────────────────
# Few-shot examples for matching
# ──────────────────────────────────────────────────────────────

FEW_SHOT_EXAMPLES = """
=== EXAMPLE (CORRECT) ===
Candidate Skills: ["Python", "Django", "REST APIs"]
Candidate Tools:  ["AWS EC2"]
Required Skills:  ["Python", "FastAPI", "REST APIs", "PostgreSQL"]
Required Tools:   ["AWS EC2", "Docker"]

Output:
{{
  "matched_skills": ["Python", "REST APIs"],
  "missing_skills": ["FastAPI", "PostgreSQL"],
  "matched_tools":  ["AWS EC2"],
  "missing_tools":  ["Docker"],
  "match_percentage": 50
}}

RULE: match_percentage = (matched_skills + matched_tools) /
                         (required_skills + required_tools) * 100
      Rounded to nearest integer.
"""

# ──────────────────────────────────────────────────────────────
# Matching Prompt Template
# ──────────────────────────────────────────────────────────────

MATCH_TEMPLATE = """
You are an expert technical recruiter performing skill gap analysis.

{few_shot_examples}

=== CANDIDATE PROFILE ===
Extracted Skills : {candidate_skills}
Extracted Tools  : {candidate_tools}

=== JOB REQUIREMENTS ===
Required Skills  : {required_skills}
Required Tools   : {required_tools}

=== TASK ===
1. Compare candidate skills/tools against the job requirements.
2. Identify EXACT matches (case-insensitive comparison is fine).
3. List what is MISSING from the candidate profile.
4. Calculate match_percentage as:
   (number of matched items) / (total required items) * 100

STRICT RULE: Only match what was provided. Do NOT invent matches.

=== OUTPUT FORMAT (strict JSON only) ===
{{
  "matched_skills": ["skill1", "skill2"],
  "missing_skills": ["skill3", "skill4"],
  "matched_tools":  ["tool1"],
  "missing_tools":  ["tool2"],
  "match_percentage": <integer 0-100>
}}

JSON Output:
"""

def get_match_prompt() -> PromptTemplate:
    """Return the matching prompt with few-shot examples pre-filled."""
    return PromptTemplate(
        input_variables=[
            "candidate_skills",
            "candidate_tools",
            "required_skills",
            "required_tools",
        ],
        template=MATCH_TEMPLATE.replace("{few_shot_examples}", FEW_SHOT_EXAMPLES),
    )
