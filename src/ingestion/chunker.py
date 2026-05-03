"""Token-based chunking for loaded PDF pages.

Splits each page's text into overlapping chunks measured in tokens (not
characters), so chunk sizes line up with embedding-model token limits.
Page and source metadata is preserved on every chunk for downstream
citation.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Match the embedding model used in src/ingestion/embedder.py.
# from_tiktoken_encoder uses this name to pick the right tokenizer.
ENCODING_MODEL = "text-embedding-3-small"


def chunk_pages(
    pages: list[dict],
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[dict]:
    """Split per-page text into token-sized chunks with source metadata.

    Args:
        pages: Output of `load_pdf` — list of
            {"page": int, "text": str, "source": str}.
        chunk_size: Target chunk size in tokens.
        overlap: Number of tokens shared between adjacent chunks.

    Returns:
        List of {"text": str, "metadata": {"page": int, "source": str,
        "chunk_index": int}}. `chunk_index` is global across the whole
        document so downstream code can preserve order even if pages are
        re-shuffled.

    Raises:
        ValueError: If `chunk_size <= 0` or `overlap < 0` or
            `overlap >= chunk_size`.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if overlap < 0:
        raise ValueError(f"overlap must be non-negative, got {overlap}")
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be smaller than chunk_size ({chunk_size})"
        )

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name=ENCODING_MODEL,
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )

    chunks: list[dict] = []
    chunk_index = 0

    for page in pages:
        for piece in splitter.split_text(page["text"]):
            chunks.append(
                {
                    "text": piece,
                    "metadata": {
                        "page": page["page"],
                        "source": page["source"],
                        "chunk_index": chunk_index,
                    },
                }
            )
            chunk_index += 1

    return chunks


if __name__ == "__main__":
    import sys

    from src.ingestion.loader import load_pdf

    if len(sys.argv) != 2:
        print("Usage: python -m src.ingestion.chunker <path-to-pdf>")
        sys.exit(1)

    pages = load_pdf(sys.argv[1])
    chunks = chunk_pages(pages, chunk_size=200, overlap=20)

    print(f"Loaded {len(pages)} pages -> {len(chunks)} chunks")
    for chunk in chunks[:3]:
        meta = chunk["metadata"]
        preview = chunk["text"][:100].replace("\n", " ")
        print(
            f"  chunk {meta['chunk_index']:3d} | page {meta['page']} | "
            f"{preview}..."
        )
