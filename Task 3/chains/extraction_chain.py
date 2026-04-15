"""
Chain for STEP 1: Skill Extraction.

Uses LCEL (LangChain Expression Language) pipe operator.
Calls Groq LLM → parses JSON output.
Tagged with "extraction" for LangSmith tracing.
"""

import json
import re
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableLambda
from prompts.extract_prompt import get_extract_prompt


# ──────────────────────────────────────────────────────────────
# Helper: safely parse JSON from LLM text response
# ──────────────────────────────────────────────────────────────

def parse_json_response(text: str) -> dict:
    """
    Extract and parse the first valid JSON object from LLM output.
    Handles code fences (```json ... ```) and raw JSON strings.
    """
    # Strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()

    # Attempt to find JSON block
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Fallback: return error dict
    return {
        "skills": [],
        "tools": [],
        "experience": "Parse error – LLM output was not valid JSON",
        "raw_output": text,
    }


# ──────────────────────────────────────────────────────────────
# Build the Extraction Chain
# ──────────────────────────────────────────────────────────────

def build_extraction_chain(llm: ChatGroq):
    """
    Build and return the extraction LCEL chain.

    Pipeline:
      PromptTemplate → ChatGroq LLM → JSON Parser

    Args:
        llm: Initialised ChatGroq instance.

    Returns:
        A Runnable chain that accepts {"resume_text": str}
        and returns a parsed dict with skills/tools/experience.
    """
    prompt = get_extract_prompt()

    # LCEL chain using the | (pipe) operator
    chain = (
        prompt
        | llm
        | RunnableLambda(lambda response: parse_json_response(response.content))
    )

    return chain


def run_extraction(llm: ChatGroq, resume_text: str) -> dict:
    """
    Execute the extraction chain for a given resume.

    Args:
        llm         : Initialised ChatGroq LLM.
        resume_text : Raw resume text string.

    Returns:
        dict with keys: skills, tools, experience.
    """
    chain = build_extraction_chain(llm)

    # .invoke() with LangSmith tag for tracing
    result = chain.invoke(
        {"resume_text": resume_text},
        config={"tags": ["extraction"]},
    )
    return result
