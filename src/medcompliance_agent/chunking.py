# src/chunking.py
from typing import List, Dict
from .config import CHUNK_SIZE_CHARS, CHUNK_OVERLAP_CHARS


def chunk_text(text: str) -> List[Dict]:
    """
    Simple character-based chunking.

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

        # overlap
        start = end - CHUNK_OVERLAP_CHARS
        if start < 0:
            start = 0

        if end == length:
            break

    return chunks
