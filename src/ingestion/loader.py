"""PDF loader: read a PDF from disk into per-page text dicts."""

import logging
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

logger = logging.getLogger(__name__)


def load_pdf(path: str) -> list[dict]:
    """Load a PDF and return one dict per non-empty page.

    Args:
        path: Filesystem path to a PDF file.

    Returns:
        List of dicts shaped {"page": int (1-indexed), "text": str, "source": str}.
        Pages with no extractable text are skipped (with a warning log).
        An empty PDF returns an empty list.

    Raises:
        FileNotFoundError: If `path` does not exist.
        ValueError: If `path` does not point to a readable PDF.
    """
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    if not p.is_file():
        raise ValueError(f"Path is not a file: {path}")

    if p.suffix.lower() != ".pdf":
        raise ValueError(f"Not a PDF (expected .pdf extension): {path}")

    try:
        reader = PdfReader(str(p))
    except PdfReadError as e:
        raise ValueError(f"Could not parse PDF: {path} ({e})") from e

    source = str(p)
    pages: list[dict] = []

    if len(reader.pages) == 0:
        logger.warning("PDF has zero pages: %s", source)
        return pages

    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as e:
            logger.warning("Failed to extract text from page %d of %s: %s", i, source, e)
            continue

        text = text.strip()
        if not text:
            logger.warning("Skipping page %d of %s: no extractable text", i, source)
            continue

        pages.append({"page": i, "text": text, "source": source})

    return pages


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if len(sys.argv) != 2:
        print("Usage: python -m src.ingestion.loader <path-to-pdf>")
        sys.exit(1)

    result = load_pdf(sys.argv[1])
    print(f"Loaded {len(result)} non-empty pages from {sys.argv[1]}")
    for entry in result[:3]:
        preview = entry["text"][:120].replace("\n", " ")
        print(f"  page {entry['page']}: {preview}...")
