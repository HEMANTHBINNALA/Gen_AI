"""
Prompt template for STEP 4: Score Explanation.

Generates a human-readable explanation of the score.
STRICT RULE: Use ONLY extracted + matched data. No hallucination.
"""

from langchain_core.prompts import PromptTemplate

# ──────────────────────────────────────────────────────────────
# Explanation Prompt Template
# ──────────────────────────────────────────────────────────────

EXPLAIN_TEMPLATE = """
You are an expert HR analyst writing a candidate evaluation report.

STRICT RULE: Base your explanation SOLELY on the data provided below.
Do NOT invent, assume, or hallucinate any skills, experience, or qualifications.
If a field is empty, state that clearly — do not fill it with assumptions.

=== CANDIDATE DATA ===
Extracted Skills      : {candidate_skills}
Extracted Tools       : {candidate_tools}
Candidate Experience  : {candidate_experience}

=== JOB REQUIREMENTS ===
Required Skills       : {required_skills}
Required Tools        : {required_tools}
Required Experience   : {required_experience}

=== MATCH RESULTS ===
Matched Skills        : {matched_skills}
Missing Skills        : {missing_skills}
Matched Tools         : {matched_tools}
Missing Tools         : {missing_tools}
Match Percentage      : {match_percentage}%

=== SCORE BREAKDOWN ===
Total Fit Score       : {score} / 100
  - Skill Score       : {skill_score} / 50
  - Tool Score        : {tool_score} / 20
  - Experience Score  : {experience_score} / 30

=== TASK ===
Write a structured evaluation report with these exact sections:

1. OVERALL ASSESSMENT  – 2 sentences summarising the candidate's fit.
2. STRENGTHS           – Bullet list of what the candidate does well (based on matched data ONLY).
3. WEAKNESSES / GAPS   – Bullet list of missing skills/tools/experience.
4. RECOMMENDATION      – One of: "Strong Hire", "Consider with Reservations", or "Not Recommended".
                          Justify the recommendation using the score and gaps.

Keep the language professional, factual, and concise.
Do NOT pad with generic phrases. Do NOT add skills not present in the data.

=== OUTPUT ===
"""

def get_explain_prompt() -> PromptTemplate:
    """Return the explanation prompt template."""
    return PromptTemplate(
        input_variables=[
            "candidate_skills",
            "candidate_tools",
            "candidate_experience",
            "required_skills",
            "required_tools",
            "required_experience",
            "matched_skills",
            "missing_skills",
            "matched_tools",
            "missing_tools",
            "match_percentage",
            "score",
            "skill_score",
            "tool_score",
            "experience_score",
        ],
        template=EXPLAIN_TEMPLATE,
    )
