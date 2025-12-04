# src/medcompliance_agent/evaluation.py

from typing import List, Dict, Tuple
import re
import json


# Utility scoring functions
def compute_hallucination_score(checklist: str, retrieved_text: str) -> Tuple[int, List[str]]:
    """
    Detect hallucinations:
    - A checklist item is a hallucination if it does not appear in retrieved text.
    - We do simple string containment for now (safe + deterministic).
    """

    hallucinations = []
    items = extract_checklist_items(checklist)

    for item in items:
        if item.lower() not in retrieved_text.lower():
            hallucinations.append(item)

    score = len(hallucinations)
    return score, hallucinations



def compute_redundancy_score(checklist: str) -> float:
    """
    Detect whether any checklist items are too similar/duplicated.
    Simple heuristic:
    - Convert to lowercase
    - Remove punctuation
    - Check duplicates
    """
    items = extract_checklist_items(checklist)
    normalized = [re.sub(r"[^a-z0-9 ]", "", i.lower()) for i in items]
    unique = set(normalized)
    if len(items) == 0:
        return 0.0
    return 1.0 - (len(unique) / len(items))  # redundancy ratio



def compute_coverage_score(present_requirements: List[str], checklist: str) -> float:
    """
    How many known requirements appear in the checklist?
    present_requirements: extracted by the LLM in reasoning stage
    """
    if not present_requirements:
        return 1.0  # nothing missing

    included = 0
    for req in present_requirements:
        if req.lower() in checklist.lower():
            included += 1

    return included / len(present_requirements)



def format_validity(checklist: str) -> bool:
    """
    Check if output is in the correct format:
    Compliance checklist for <device>:
    1. ...
    """
    if "Compliance checklist for" not in checklist:
        return False
    if not re.search(r"\n1\.", checklist):
        return False
    return True



# Checklist item extractor
def extract_checklist_items(checklist: str) -> List[str]:
    """
    Extract numbered items from checklist.
    """
    return re.findall(r"\d+\.\s+(.*)", checklist)



# Main evaluation function
def evaluate_checklist(checklist: str, retrieved_chunks: List[Tuple[Dict, float]]):
    """
    Full evaluation of checklist quality.
    Returns:
        - human_readable (str)
        - json_result (dict)
    """

    # Combine retrieved text for scoring
    retrieved_text = "\n".join(chunk["text"] for chunk, _ in retrieved_chunks)

    # Compute metrics
    halluc_score, halluc_items = compute_hallucination_score(checklist, retrieved_text)
    redundancy = compute_redundancy_score(checklist)
    validity = format_validity(checklist)

    # For coverage we need a list of "present requirements"
    # The agent's reasoning module can extract this later
    coverage = None  # placeholder for now

    # Build JSON output
    result_json = {
        "hallucinations": halluc_score,
        "hallucinated_items": halluc_items,
        "redundancy_score": redundancy,
        "format_valid": validity,
        "coverage_score": coverage,
    }

    # Human-readable summary
    result_text = f"""
Evaluation Summary:
-------------------
- Hallucinations: {halluc_score}
- Redundancy score: {redundancy:.2f}
- Format valid: {"YES" if validity else "NO"}
- Coverage score: {coverage if coverage is not None else "N/A"}

Hallucinated Items:
{halluc_items if halluc_items else "None"}

JSON output:
{json.dumps(result_json, indent=2)}
"""

    return result_text, result_json
