"""
Chain for STEP 4: Score Explanation.

Generates a detailed, factual HR evaluation report.
STRICT RULE: No hallucination – only uses provided data.
Tagged with "explanation" for LangSmith tracing.
"""

from langchain_groq import ChatGroq
from prompts.explain_prompt import get_explain_prompt


# ──────────────────────────────────────────────────────────────
# Build the Explanation Chain
# ──────────────────────────────────────────────────────────────

def build_explanation_chain(llm: ChatGroq):
    """
    Build and return the explanation LCEL chain.

    Pipeline:
      PromptTemplate → ChatGroq LLM → Raw text response

    Args:
        llm: Initialised ChatGroq instance.

    Returns:
        A Runnable chain that produces a text explanation.
    """
    prompt = get_explain_prompt()

    # Output is plain text (HR report), not JSON
    chain = prompt | llm

    return chain


def run_explanation(
    llm: ChatGroq,
    candidate_skills: list,
    candidate_tools: list,
    candidate_experience: str,
    required_skills: list,
    required_tools: list,
    required_experience: str,
    matched_skills: list,
    missing_skills: list,
    matched_tools: list,
    missing_tools: list,
    match_percentage: int,
    score: int,
    skill_score: int,
    tool_score: int,
    experience_score: int,
) -> str:
    """
    Execute the explanation chain and return a text evaluation report.

    All arguments are required to ensure no hallucination.
    The LLM can only use what is explicitly passed in.

    Returns:
        str: Human-readable HR evaluation report.
    """
    chain = build_explanation_chain(llm)

    response = chain.invoke(
        {
            "candidate_skills": str(candidate_skills),
            "candidate_tools": str(candidate_tools),
            "candidate_experience": candidate_experience,
            "required_skills": str(required_skills),
            "required_tools": str(required_tools),
            "required_experience": required_experience,
            "matched_skills": str(matched_skills),
            "missing_skills": str(missing_skills),
            "matched_tools": str(matched_tools),
            "missing_tools": str(missing_tools),
            "match_percentage": str(match_percentage),
            "score": str(score),
            "skill_score": str(skill_score),
            "tool_score": str(tool_score),
            "experience_score": str(experience_score),
        },
        config={"tags": ["explanation"]},
    )

    # response is an AIMessage; extract text content
    return response.content
