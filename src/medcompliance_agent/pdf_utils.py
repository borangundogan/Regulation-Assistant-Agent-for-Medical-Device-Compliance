# src/medcompliance_agent/pdf_utils.py

import io
import pdfplumber


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """
    Extract plain text from a PDF given as raw bytes.
    Very simple for now; later we can add better cleaning.
    """
    text_parts = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)

    return "\n".join(text_parts)
