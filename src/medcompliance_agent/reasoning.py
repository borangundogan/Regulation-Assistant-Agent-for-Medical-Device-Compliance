# src/medcompliance_agent/reasoning.py

from .llm_client import call_llama


def analyze_requirements(device: str, context_text: str) -> str:
    """
    Ask model: What requirements exist? What is missing?
    """
    messages = [
        {"role": "system", "content": "You analyze regulatory requirements."},
        {"role": "user", "content": f"""
Device: {device}

Text from regulations:
{context_text}

Task:
1. List all requirements explicitly mentioned.
2. List requirements that are missing or unclear.
3. DO NOT invent new requirements.
4. Answer in two sections:
    - present_requirements:
    - missing_information:
"""},
    ]
    return call_llama(messages, model="llama3.1")


def make_checklist(device: str, present_reqs: str) -> str:
    """
    Create structured checklist ONLY from present requirements.
    """
    messages = [
        {"role": "system", "content": "You generate compliance checklists."},
        {"role": "user", "content": f"""
Device: {device}

Explicit Requirements:
{present_reqs}

Task:
- Create a numbered compliance checklist.
- Only include requirements listed above.
- NO hallucinations.
- Format strictly as:

Compliance checklist for <device>:
1. ...
2. ...
3. ...
"""},
    ]
    return call_llama(messages, model="llama3.1")


def refine_checklist(device: str, checklist: str, context_text: str) -> str:
    messages = [
        {"role": "system", "content": "You refine compliance checklists. You are NOT allowed to add any requirement that is not explicitly stated in the provided regulatory text."},
        {"role": "user", "content": f"""
Device: {device}

Checklist draft:
{checklist}

Regulatory context (you MUST NOT exceed this information):
{context_text}

Your tasks:
1. Improve wording and clarity.
2. Remove redundancy.
3. DO NOT add any new requirements.
4. DO NOT assume general medical device rules.
5. DO NOT expand content beyond what exists in the regulatory context.
6. Maintain the SAME checklist structure:
   Compliance checklist for <device>:
   1. ...
   2. ...
   3. ...
7. Output ONLY the refined checklist.
8. DO NOT output explanations, reasoning, notes, or bullet points describing changes.
9. DO NOT output any text except the final checklist.

"""},
    ]
    return call_llama(messages, model="llama3.1")
