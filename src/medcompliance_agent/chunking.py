# src/medcompliance_agent/chunking.py

from typing import List, Dict, Any
import re
from .config import CHUNK_SIZE_CHARS, CHUNK_OVERLAP_CHARS


# 1) Legacy function: static MDR chunking
def chunk_text(text: str) -> List[Dict]:
    """
    Simple character-based chunking for static regulation text.

    Returns a list of dicts:
    {
        "id": int,
        "text": str,
        "start": int,
        "end": int,
    }
    """
    chunks: List[Dict] = []
    start = 0
    chunk_id = 0
    length = len(text)

    while start < length:
        end = min(start + CHUNK_SIZE_CHARS, length)
        chunk_text_str = text[start:end].strip()

        if chunk_text_str:
            chunks.append(
                {
                    "id": chunk_id,
                    "text": chunk_text_str,
                    "start": start,
                    "end": end,
                }
            )
            chunk_id += 1

        # Apply overlap
        start = end - CHUNK_OVERLAP_CHARS
        if start < 0:
            start = 0

        if end == length:
            break

    return chunks


# 2) Section-based segmentation: Article, Annex, Chapter detection
def segment_regulation_text(raw_text: str) -> List[Dict[str, str]]:
    """
    Splits regulation text into logical sections based on markdown-like headings
    such as 'Article 10', 'ANNEX I', 'CHAPTER II'.

    Returns:
    [
        {
            "title": "Article 10 – General obligations",
            "body": "... section text ...",
            "type": "RISK_MANAGEMENT" | "TECHNICAL_DOCUMENTATION" | ...
        }
    ]
    """

    lines = [line.rstrip() for line in raw_text.splitlines()]
    sections: List[Dict[str, str]] = []

    current_title = "PREFACE"
    current_lines: List[str] = []

    # Heading detector
    def is_heading(line: str) -> bool:
        line = line.strip()
        if re.match(r"^(Article|ARTICLE)\s+\d+", line):
            return True
        if re.match(r"^(ANNEX|Annex)\b", line):
            return True
        if re.match(r"^(CHAPTER|Chapter)\b", line):
            return True
        return False

    # Guess coarse section type for filtering
    def guess_type(title: str, body: str) -> str:
        t = (title + " " + body[:300]).lower()

        if "technical documentation" in t or "annex ii" in t:
            return "TECHNICAL_DOCUMENTATION"
        if "risk management" in t or "14971" in t:
            return "RISK_MANAGEMENT"
        if "post-market" in t or "post market" in t or "pms" in t:
            return "POST_MARKET"
        if "clinical evaluation" in t or "pmcf" in t:
            return "CLINICAL_EVALUATION"
        if "classification" in t:
            return "CLASSIFICATION"
        if "general safety" in t or "performance requirements" in t:
            return "GENERAL_SAFETY"
        return "OTHER"

    for line in lines:
        if is_heading(line):
            # Flush previous section
            if current_lines:
                body = "\n".join(current_lines).strip()
                if body:
                    sections.append(
                        {
                            "title": current_title,
                            "body": body,
                            "type": guess_type(current_title, body),
                        }
                    )
            current_title = line.strip()
            current_lines = []
        else:
            if line.strip():
                current_lines.append(line)

    # Flush last section
    if current_lines:
        body = "\n".join(current_lines).strip()
        if body:
            sections.append(
                {
                    "title": current_title,
                    "body": body,
                    "type": guess_type(current_title, body),
                }
            )

    return sections


# 3) sliding-window chunking
def _simple_split_into_chunks(
    text: str,
    max_chars: int = 800,
    overlap: int = 200,
) -> List[str]:
    """
    Simple sliding-window chunking used inside regulation sections.
    """
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + max_chars, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)

    return chunks

# 4) PDF-specific chunking: segmentation + metadata injection
def chunk_pdf_with_sections(raw_text: str, source_name: str = "uploaded_pdf") -> List[Dict[str, Any]]:
    """
    Main pipeline for PDF-based regulation RAG.

    Steps:
    1. Detect regulatory sections (Article / Annex / Chapter)
    2. Chunk each section with overlap
    3. Add metadata: section title, section type, source pdf name

    Output format:
    [
        {
            "id": "pdfname_sec0_chunk0",
            "text": "...",
            "source": "pdfname.pdf",
            "section_title": "Article 10",
            "section_type": "TECHNICAL_DOCUMENTATION"
        },
        ...
    ]
    """

    sections = segment_regulation_text(raw_text)
    all_chunks: List[Dict[str, Any]] = []

    for sec_idx, sec in enumerate(sections):
        title = sec["title"]
        body = sec["body"]
        sec_type = sec["type"]
        base_id = f"{source_name}_sec{sec_idx}"

        # Split into overlapped chunks
        pieces = _simple_split_into_chunks(body, max_chars=800, overlap=200)

        for j, piece in enumerate(pieces):
            chunk_id = f"{base_id}_chunk{j}"
            all_chunks.append(
                {
                    "id": chunk_id,
                    "text": piece,
                    "source": source_name,
                    "section_title": title,
                    "section_type": sec_type,
                }
            )

    return all_chunks
