"""
Prompt template for STEP 1: Skill Extraction from Resume.

STRICT RULE: Extract ONLY what is explicitly stated.
Do NOT infer, assume, or hallucinate any information.
"""

from langchain_core.prompts import PromptTemplate

# ──────────────────────────────────────────────────────────────
# Few-shot examples to guide accurate extraction
# ──────────────────────────────────────────────────────────────

FEW_SHOT_EXAMPLES = """
=== EXAMPLE 1 (CORRECT) ===
Resume Text:
"I have 3 years of experience in Python, worked with Django and REST APIs.
Familiar with AWS EC2 and S3."

Output:
{{
  "skills": ["Python", "Django", "REST APIs"],
  "tools": ["AWS EC2", "AWS S3"],
  "experience": "3 years"
}}

=== EXAMPLE 2 (INCORRECT - Do NOT do this) ===
Resume Text:
"I worked with Python and Django."

WRONG Output (hallucination):
{{
  "skills": ["Python", "Django", "Flask", "FastAPI"],
  "tools": ["Docker", "Kubernetes"],
  "experience": "5 years"
}}

WHY WRONG: Flask, FastAPI, Docker, Kubernetes, and "5 years" were never mentioned.

=== EXAMPLE 2 (CORRECTED) ===
{{
  "skills": ["Python", "Django"],
  "tools": [],
  "experience": "Not specified"
}}
"""

# ──────────────────────────────────────────────────────────────
# Extraction Prompt Template
# ──────────────────────────────────────────────────────────────

EXTRACT_TEMPLATE = """
You are a strict and precise resume parser.

Your ONLY job is to extract information EXPLICITLY mentioned in the resume.
Do NOT infer, guess, or assume anything beyond what is written.

{few_shot_examples}

=== RESUME TEXT ===
{resume_text}

=== TASK ===
Extract the following from the resume above:
1. skills     - Programming languages, frameworks, methodologies (only explicitly mentioned)
2. tools      - Software, platforms, cloud services, DevOps tools (only explicitly mentioned)
3. experience - Total years of experience (state "Not specified" if not explicitly mentioned)

=== OUTPUT FORMAT (strict JSON only, no extra text) ===
{{
  "skills": ["skill1", "skill2"],
  "tools": ["tool1", "tool2"],
  "experience": "X years or Not specified"
}}

JSON Output:
"""

# Build the PromptTemplate
extract_prompt = PromptTemplate(
    input_variables=["resume_text", "few_shot_examples"],
    template=EXTRACT_TEMPLATE,
)

def get_extract_prompt() -> PromptTemplate:
    """Return the extraction prompt with few-shot examples pre-filled."""
    return PromptTemplate(
        input_variables=["resume_text"],
        template=EXTRACT_TEMPLATE.replace("{few_shot_examples}", FEW_SHOT_EXAMPLES),
    )
