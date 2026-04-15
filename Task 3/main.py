"""
AI Resume Screening System
===========================
Author  : AI Resume Screener
Stack   : Python 3.10+ | LangChain | Groq (Mixtral) | LangSmith

Pipeline per resume:
  Resume Text
    → [1] Skill Extraction   (extraction_chain)
    → [2] Skill Matching     (matching_chain)
    → [3] Fit Score          (scoring_chain)
    → [4] Explanation        (explanation_chain)
    → Final Report

LangSmith traces every run automatically via env variables.
"""

import os
import re
import json
from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# ── Import chain runners ───────────────────────────────────────
from chains.extraction_chain import run_extraction
from chains.matching_chain import run_matching
from chains.scoring_chain import run_scoring
from chains.explanation_chain import run_explanation

# ─────────────────────────────────────────────────────────────
# 1. LOAD ENVIRONMENT VARIABLES
# ─────────────────────────────────────────────────────────────

load_dotenv()

# Verify essential keys are present
REQUIRED_ENV = ["GROQ_API_KEY", "LANGCHAIN_API_KEY"]
for key in REQUIRED_ENV:
    if not os.getenv(key):
        raise EnvironmentError(f"Missing required environment variable: {key}")


# ─────────────────────────────────────────────────────────────
# 2. INITIALISE LLM (Groq via LangChain)
# ─────────────────────────────────────────────────────────────

llm = ChatGroq(
    model="llama-3.3-70b-versatile",  # LLaMA 3.3 70B – Groq's recommended model
    temperature=0,                     # Deterministic output (no randomness)
    api_key=os.getenv("GROQ_API_KEY"),
)


# ─────────────────────────────────────────────────────────────
# 3. LOAD DATA FILES
# ─────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent / "data"

def load_text(filename: str) -> str:
    """Load and return text content from a data file."""
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def parse_job_description(jd_text: str) -> dict:
    """
    Parse the job description to extract required skills, tools,
    and experience using simple rule-based extraction.

    Returns a dict with keys: required_skills, required_tools, required_experience.
    """
    required_skills = []
    required_tools = []
    required_experience = "Not specified"

    lines = jd_text.splitlines()
    section = None

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Detect sections
        if "Required Skills" in stripped:
            section = "skills"
            continue
        elif "Required Tools" in stripped or "Required Technologies" in stripped:
            section = "tools"
            continue
        elif "Required Experience" in stripped:
            section = "experience"
            continue
        elif stripped.startswith("Responsibilities") or stripped.startswith("Nice"):
            section = None
            continue

        # Collect items in current section
        if section == "skills" and stripped.startswith("-"):
            skill = stripped.lstrip("- ").strip()
            if skill:
                required_skills.append(skill)

        elif section == "tools" and stripped.startswith("-"):
            tool = stripped.lstrip("- ").strip()
            if tool:
                required_tools.append(tool)

        elif section == "experience":
            # Grab the first non-empty line as experience requirement
            if required_experience == "Not specified" and stripped:
                required_experience = stripped

    return {
        "required_skills": required_skills,
        "required_tools": required_tools,
        "required_experience": required_experience,
    }


# ─────────────────────────────────────────────────────────────
# 4. PRINT HELPERS
# ─────────────────────────────────────────────────────────────

DIVIDER = "=" * 70

def print_section(title: str):
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)

def print_dict(label: str, data: dict):
    """Pretty-print a dictionary."""
    print(f"\n[INFO] {label}:")
    for k, v in data.items():
        if k == "raw_output":
            continue  # Skip raw fallback output
        print(f"   {k:22s}: {v}")

def print_score_bar(score: int):
    """Print a visual score bar."""
    filled = int(score / 5)   # 20 blocks max
    bar = "#" * filled + "-" * (20 - filled)
    label = (
        "[STRONG]"   if score >= 75 else
        "[AVERAGE]"  if score >= 45 else
        "[WEAK]"
    )
    print(f"\n   Score: [{bar}] {score}/100  {label}")


# ─────────────────────────────────────────────────────────────
# 5. CORE PIPELINE FUNCTION
# ─────────────────────────────────────────────────────────────

def run_pipeline(
    candidate_label: str,
    resume_text: str,
    jd_data: dict,
) -> dict:
    """
    Run the full 4-step screening pipeline for one resume.

    Steps:
      1. Extraction  – extract skills/tools/experience from resume
      2. Matching    – compare against JD requirements
      3. Scoring     – compute weighted 0-100 score
      4. Explanation – generate HR evaluation report

    Args:
        candidate_label : Human-readable label (e.g. "Strong Candidate")
        resume_text     : Raw text content of the resume
        jd_data         : Parsed job description dict

    Returns:
        Full result dict with all pipeline outputs.
    """
    print_section(f"CANDIDATE: {candidate_label}")

    # ── STEP 1: EXTRACTION ─────────────────────────────────
    print("\n[1] Step 1: Extracting skills from resume...")
    extracted = run_extraction(llm, resume_text)
    print_dict("Extracted Data", extracted)

    # ── STEP 2: MATCHING ───────────────────────────────────
    print("\n[2] Step 2: Matching skills against Job Description...")
    matched = run_matching(
        llm,
        candidate_skills=extracted.get("skills", []),
        candidate_tools=extracted.get("tools", []),
        required_skills=jd_data["required_skills"],
        required_tools=jd_data["required_tools"],
    )
    print_dict("Match Results", matched)

    # ── STEP 3: SCORING ────────────────────────────────────
    print("\n[3] Step 3: Calculating fit score...")
    scored = run_scoring(
        llm,
        matched_skills=matched.get("matched_skills", []),
        missing_skills=matched.get("missing_skills", []),
        matched_tools=matched.get("matched_tools", []),
        missing_tools=matched.get("missing_tools", []),
        match_percentage=matched.get("match_percentage", 0),
        candidate_experience=extracted.get("experience", "Not specified"),
        required_experience=jd_data["required_experience"],
    )
    print_score_bar(scored.get("score", 0))
    print(f"   Breakdown -> Skill: {scored.get('skill_score',0)}/50 | "
          f"Tools: {scored.get('tool_score',0)}/20 | "
          f"Experience: {scored.get('experience_score',0)}/30")

    # ── STEP 4: EXPLANATION ────────────────────────────────
    print("\n[4] Step 4: Generating explanation report...")
    explanation = run_explanation(
        llm,
        candidate_skills=extracted.get("skills", []),
        candidate_tools=extracted.get("tools", []),
        candidate_experience=extracted.get("experience", "Not specified"),
        required_skills=jd_data["required_skills"],
        required_tools=jd_data["required_tools"],
        required_experience=jd_data["required_experience"],
        matched_skills=matched.get("matched_skills", []),
        missing_skills=matched.get("missing_skills", []),
        matched_tools=matched.get("matched_tools", []),
        missing_tools=matched.get("missing_tools", []),
        match_percentage=matched.get("match_percentage", 0),
        score=scored.get("score", 0),
        skill_score=scored.get("skill_score", 0),
        tool_score=scored.get("tool_score", 0),
        experience_score=scored.get("experience_score", 0),
    )

    print(f"\n[REPORT] Evaluation Report:\n")
    print(explanation)

    return {
        "candidate": candidate_label,
        "extracted": extracted,
        "matched": matched,
        "scored": scored,
        "explanation": explanation,
    }


# ─────────────────────────────────────────────────────────────
# 6. MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────

def main():
    print(DIVIDER)
    print("  [AI]  AI RESUME SCREENING SYSTEM")
    print("  Powered by: LangChain + Groq (Mixtral) + LangSmith")
    print(DIVIDER)

    # Load and parse job description once
    print("\n[*] Loading Job Description...")
    jd_text = load_text("job_description.txt")
    jd_data = parse_job_description(jd_text)

    print(f"   Required Skills   : {jd_data['required_skills']}")
    print(f"   Required Tools    : {jd_data['required_tools']}")
    print(f"   Required Exp.     : {jd_data['required_experience']}")

    # Define the three candidate resumes to evaluate
    candidates = [
        ("[STRONG] Candidate  (Ayesha Khan)", "strong_resume.txt"),
        ("[AVERAGE] Candidate (Bilal Ahmed)", "average_resume.txt"),
        ("[WEAK] Candidate    (Zara Malik)",  "weak_resume.txt"),
    ]

    results = []

    for label, filename in candidates:
        resume_text = load_text(filename)
        result = run_pipeline(label, resume_text, jd_data)
        results.append(result)

    # ── FINAL SUMMARY TABLE ────────────────────────────────
    print(f"\n\n{DIVIDER}")
    print("  [SUMMARY]  FINAL SUMMARY - ALL CANDIDATES")
    print(DIVIDER)
    print(f"  {'Candidate':<40} {'Score':>6}  {'Skill':>6}  {'Tools':>6}  {'Exp':>6}  {'Match%':>7}")
    print(f"  {'-'*40} {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*7}")

    for r in results:
        s = r["scored"]
        m = r["matched"]
        print(
            f"  {r['candidate']:<40} "
            f"{s.get('score', 0):>6}  "
            f"{s.get('skill_score', 0):>6}  "
            f"{s.get('tool_score', 0):>6}  "
            f"{s.get('experience_score', 0):>6}  "
            f"{m.get('match_percentage', 0):>7}%"
        )

    print(f"\n{DIVIDER}")
    print("  [DONE]  Screening Complete! Check LangSmith for traces.")
    print(f"          Project: {os.getenv('LANGCHAIN_PROJECT', 'resume-screening')}")
    print(DIVIDER)


if __name__ == "__main__":
    main()
