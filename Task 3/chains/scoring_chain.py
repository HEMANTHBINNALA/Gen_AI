"""
Chain for STEP 3: Fit Score Calculation.

Generates a 0–100 score using weighted criteria:
  - Skill Match      : 50%
  - Experience Match : 30%
  - Tools Match      : 20%

Tagged with "scoring" for LangSmith tracing.
"""

import json
import re
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableLambda
from prompts.score_prompt import get_score_prompt


# ──────────────────────────────────────────────────────────────
# Helper: safely parse JSON from LLM text response
# ──────────────────────────────────────────────────────────────

def parse_json_response(text: str) -> dict:
    """Extract and parse JSON from LLM output."""
    cleaned = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {
        "score": 0,
        "skill_score": 0,
        "tool_score": 0,
        "experience_score": 0,
        "raw_output": text,
    }


# ──────────────────────────────────────────────────────────────
# Build the Scoring Chain
# ──────────────────────────────────────────────────────────────

def build_scoring_chain(llm: ChatGroq):
    """
    Build and return the scoring LCEL chain.

    Pipeline:
      PromptTemplate → ChatGroq LLM → JSON Parser

    Args:
        llm: Initialised ChatGroq instance.

    Returns:
        A Runnable chain accepting match results + experience inputs.
    """
    prompt = get_score_prompt()

    chain = (
        prompt
        | llm
        | RunnableLambda(lambda response: parse_json_response(response.content))
    )

    return chain


def run_scoring(
    llm: ChatGroq,
    matched_skills: list,
    missing_skills: list,
    matched_tools: list,
    missing_tools: list,
    match_percentage: int,
    candidate_experience: str,
    required_experience: str,
) -> dict:
    """
    Execute the scoring chain.

    Args:
        llm                  : Initialised ChatGroq LLM.
        matched_skills       : Skills that match the JD.
        missing_skills       : Skills the candidate lacks.
        matched_tools        : Tools that match the JD.
        missing_tools        : Tools the candidate lacks.
        match_percentage     : Overall match % from matching step.
        candidate_experience : Experience extracted from resume.
        required_experience  : Experience required by JD.

    Returns:
        dict with keys: score, skill_score, tool_score, experience_score.
    """
    chain = build_scoring_chain(llm)

    result = chain.invoke(
        {
            "matched_skills": str(matched_skills),
            "missing_skills": str(missing_skills),
            "matched_tools": str(matched_tools),
            "missing_tools": str(missing_tools),
            "match_percentage": str(match_percentage),
            "candidate_experience": candidate_experience,
            "required_experience": required_experience,
        },
        config={"tags": ["scoring"]},
    )
    return result
