# ============================================================
# app/domain/documents/parser_pdf.py — PDF text extraction
#
# WHY THIS FILE EXISTS:
# Separating PDF parsing into its own module means:
#   - It's easy to swap out the parsing library later
#   - The DocumentService stays clean and doesn't know
#     about PDF internals
#   - We can add OCR support here in the future without
#     touching the service layer
#
# CURRENT APPROACH (MVP):
# Uses pypdf — a pure-Python library with no native dependencies.
# It extracts embedded text from PDF pages.
#
# LIMITATION:
# pypdf can only extract TEXT that's embedded in the PDF.
# It CANNOT handle scanned PDFs (images of text).
# For scanned documents, you'd need OCR (e.g., Tesseract, AWS Textract).
# TODO: Add OCR fallback when extracted text is too short (< 50 chars/page)
# ============================================================

import pypdf
from app.core.logging import get_logger
from app.core.errors import DocumentProcessingError

# Logger for this module
logger = get_logger(__name__)


def parse_pdf(file_path: str) -> tuple[str, int]:
    """
    Extract all text from a PDF file.

    Reads each page of the PDF and concatenates the text into
    a single string. Pages are separated by double newlines.

    Args:
        file_path: Absolute or relative path to the .pdf file on disk.

    Returns:
        A tuple of (extracted_text, page_count):
          - extracted_text: The full text content of the PDF
          - page_count: The number of pages in the PDF

    Raises:
        DocumentProcessingError: If the file cannot be opened or parsed.

    Example:
        text, pages = parse_pdf("./uploads/acme/contract.pdf")
        print(f"Extracted {len(text)} characters from {pages} pages")
    """
    logger.info(f"Parsing PDF: {file_path}")

    try:
        # Open the PDF file using pypdf
        # PdfReader handles opening, parsing, and closing the file
        reader = pypdf.PdfReader(file_path)

    except FileNotFoundError:
        raise DocumentProcessingError(
            f"PDF file not found: {file_path}",
        )
    except pypdf.errors.PdfReadError as e:
        raise DocumentProcessingError(
            f"Could not parse PDF (file may be corrupted or encrypted): {e}",
        )
    except Exception as e:
        raise DocumentProcessingError(
            f"Unexpected error opening PDF: {e}",
        )

    # Count the number of pages
    page_count = len(reader.pages)

    if page_count == 0:
        raise DocumentProcessingError("PDF has no pages")

    logger.debug(f"PDF has {page_count} pages")

    # Extract text from each page
    page_texts: list[str] = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            # extract_text() returns the embedded text content of the page
            # Returns an empty string if the page has no text (e.g., image-only page)
            page_text = page.extract_text() or ""

            if page_text.strip():
                # Add a page marker for better chunk context
                page_texts.append(f"[Page {page_number}]\n{page_text.strip()}")
            else:
                # Log pages with no extractable text (likely scanned images)
                logger.warning(
                    f"Page {page_number} of {file_path} has no extractable text. "
                    "This may be a scanned image page. OCR is not supported in MVP."
                )

        except Exception as e:
            # Don't fail the whole document if one page fails — log and continue
            logger.warning(f"Could not extract text from page {page_number}: {e}")
            continue

    # Join all page texts with double newlines
    full_text = "\n\n".join(page_texts)

    if not full_text.strip():
        raise DocumentProcessingError(
            "No text could be extracted from the PDF. "
            "The file may contain only scanned images. "
            "OCR support is planned for a future version."
        )

    logger.info(
        f"PDF parsed successfully: {page_count} pages, "
        f"{len(full_text)} characters extracted"
    )

    return full_text, page_count
