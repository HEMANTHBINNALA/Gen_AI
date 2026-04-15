"""
Chain for STEP 2: Skill Matching.

Compares candidate's extracted profile against job requirements.
Tagged with "matching" for LangSmith tracing.
"""

import json
import re
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableLambda
from prompts.match_prompt import get_match_prompt


# ──────────────────────────────────────────────────────────────
# Helper: safely parse JSON from LLM text response
# ──────────────────────────────────────────────────────────────

def parse_json_response(text: str) -> dict:
    """Extract and parse JSON from LLM output, handles code fences."""
    cleaned = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {
        "matched_skills": [],
        "missing_skills": [],
        "matched_tools": [],
        "missing_tools": [],
        "match_percentage": 0,
        "raw_output": text,
    }


# ──────────────────────────────────────────────────────────────
# Build the Matching Chain
# ──────────────────────────────────────────────────────────────

def build_matching_chain(llm: ChatGroq):
    """
    Build and return the matching LCEL chain.

    Pipeline:
      PromptTemplate → ChatGroq LLM → JSON Parser

    Args:
        llm: Initialised ChatGroq instance.

    Returns:
        A Runnable chain accepting candidate + job requirement inputs.
    """
    prompt = get_match_prompt()

    chain = (
        prompt
        | llm
        | RunnableLambda(lambda response: parse_json_response(response.content))
    )

    return chain


def run_matching(
    llm: ChatGroq,
    candidate_skills: list,
    candidate_tools: list,
    required_skills: list,
    required_tools: list,
) -> dict:
    """
    Execute the matching chain.

    Args:
        llm               : Initialised ChatGroq LLM.
        candidate_skills  : Skills extracted from resume.
        candidate_tools   : Tools extracted from resume.
        required_skills   : Skills listed in job description.
        required_tools    : Tools listed in job description.

    Returns:
        dict with keys: matched_skills, missing_skills,
                        matched_tools, missing_tools, match_percentage.
    """
    chain = build_matching_chain(llm)

    result = chain.invoke(
        {
            "candidate_skills": str(candidate_skills),
            "candidate_tools": str(candidate_tools),
            "required_skills": str(required_skills),
            "required_tools": str(required_tools),
        },
        config={"tags": ["matching"]},
    )
    return result
