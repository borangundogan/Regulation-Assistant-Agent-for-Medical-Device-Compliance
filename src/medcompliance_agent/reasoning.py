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
    """
    LLM self-evaluates and improves the checklist.
    """
    messages = [
        {"role": "system", "content": "You evaluate and refine compliance documents."},
        {"role": "user", "content": f"""
Device: {device}

Checklist draft:
{checklist}

Regulatory context:
{context_text}

Task:
1. Check if checklist matches the regulations.
2. Fix missing items.
3. Remove hallucinations.
4. Improve clarity.
5. Output only the improved checklist.
"""},
    ]
    return call_llama(messages, model="llama3.1")
