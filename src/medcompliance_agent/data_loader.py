# src/data_loader.py
from pathlib import Path
from typing import List
from .config import REGULATIONS_DIR


def load_regulation_files() -> List[Path]:
    """
    Return a list of all regulation text files in the regulations directory.
    """
    if not REGULATIONS_DIR.exists():
        raise FileNotFoundError(f"Regulations directory not found: {REGULATIONS_DIR}")

    return sorted(REGULATIONS_DIR.glob("*.txt"))


def load_regulation_text() -> str:
    """
    Load and concatenate all regulation text files into a single string.
    """
    files = load_regulation_files()
    if not files:
        raise FileNotFoundError(f"No .txt files found in {REGULATIONS_DIR}")

    parts: List[str] = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        parts.append(text)

    return "\n\n".join(parts)
