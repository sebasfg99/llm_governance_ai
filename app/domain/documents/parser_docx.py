# ============================================================
# app/domain/documents/parser_docx.py — DOCX text extraction
#
# WHY THIS FILE EXISTS:
# Same rationale as parser_pdf.py — isolates DOCX parsing
# logic so the service layer stays clean.
#
# CURRENT APPROACH (MVP):
# Uses python-docx to read .docx files.
# Extracts text from paragraphs and table cells.
#
# NOTE: .doc files (old Word format) are NOT supported.
# Users must upload .docx (Office Open XML format).
# TODO: Add doc-to-docx conversion using LibreOffice if needed
# ============================================================

import docx
from app.core.logging import get_logger
from app.core.errors import DocumentProcessingError

# Logger for this module
logger = get_logger(__name__)


def parse_docx(file_path: str) -> tuple[str, int]:
    """
    Extract all text from a DOCX (Word) file.

    Reads all paragraphs and table cells from the document.
    Returns the text as a single string and a paragraph count
    (used as a proxy for "page count" since DOCX has no page concept).

    Args:
        file_path: Absolute or relative path to the .docx file on disk.

    Returns:
        A tuple of (extracted_text, paragraph_count):
          - extracted_text: Full text content of the document
          - paragraph_count: Number of non-empty paragraphs

    Raises:
        DocumentProcessingError: If the file cannot be opened or parsed.

    Example:
        text, para_count = parse_docx("./uploads/acme/policy.docx")
        print(f"Extracted {len(text)} characters from {para_count} paragraphs")
    """
    logger.info(f"Parsing DOCX: {file_path}")

    try:
        # Open the DOCX file with python-docx
        # Document() parses the XML structure of the .docx file
        document = docx.Document(file_path)

    except FileNotFoundError:
        raise DocumentProcessingError(
            f"DOCX file not found: {file_path}",
        )
    except Exception as e:
        raise DocumentProcessingError(
            f"Could not open DOCX file (may be corrupted or not a valid .docx): {e}",
        )

    # Collect all text segments
    text_segments: list[str] = []
    paragraph_count = 0

    # --- Extract text from paragraphs ---
    # A paragraph in python-docx is a block of text (including headings,
    # body text, lists, etc.)
    for paragraph in document.paragraphs:
        text = paragraph.text.strip()

        if text:  # Skip empty paragraphs
            text_segments.append(text)
            paragraph_count += 1

    # --- Extract text from tables ---
    # Tables in DOCX contain important structured data (e.g., policy matrices).
    # We extract each cell's text and join with tabs/newlines to preserve structure.
    for table_index, table in enumerate(document.tables):
        for row in table.rows:
            # Join cells in a row with a tab separator
            row_text = "\t".join(cell.text.strip() for cell in row.cells if cell.text.strip())
            if row_text:
                text_segments.append(row_text)

    # Join all segments with double newlines
    full_text = "\n\n".join(text_segments)

    if not full_text.strip():
        raise DocumentProcessingError(
            "No text could be extracted from the DOCX file. "
            "The document may be empty or contain only images."
        )

    logger.info(
        f"DOCX parsed successfully: {paragraph_count} paragraphs, "
        f"{len(document.tables)} tables, "
        f"{len(full_text)} characters extracted"
    )

    # Return paragraph count as "page count" equivalent for DOCX
    # (DOCX files don't have a concept of pages in their raw XML structure)
    return full_text, paragraph_count
