# ============================================================
# app/domain/documents/parser_txt.py — Plain text file parser
#
# WHY THIS FILE EXISTS:
# Provides a simple parser for .txt files that follows the same
# interface as parser_pdf.py and parser_docx.py so that
# DocumentService can dispatch to it without special casing.
#
# Returns (text, line_count) where line_count serves as a
# proxy for "page count" since plain text has no pages.
# ============================================================

from app.core.logging import get_logger

logger = get_logger(__name__)


def parse_txt(file_path: str) -> tuple[str, int]:
    """
    Extract text from a plain text (.txt) file.

    Args:
        file_path: Absolute or relative path to the .txt file.

    Returns:
        A tuple of (extracted_text, line_count).
        line_count is used as a proxy for page_count.

    Raises:
        ValueError: If the file cannot be read or is empty.
    """
    logger.debug(f"Parsing TXT file: {file_path}")

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    if not text.strip():
        raise ValueError(f"TXT file is empty or contains only whitespace: {file_path}")

    line_count = len(text.splitlines())
    logger.debug(f"Extracted {len(text)} characters, {line_count} lines from {file_path}")

    return text, line_count
