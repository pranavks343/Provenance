"""End-to-end RAG CLI: PDF + question -> streamed answer.

Usage:
    python -m scripts.cli <path-to-pdf> "<question>"

Re-runs against the same PDF reuse the persisted Chroma collection
(named deterministically from the absolute PDF path), so embedding cost
is paid once per document.
"""

import hashlib
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from src.generation.chain import build_chain
from src.ingestion.chunker import chunk_pages
from src.ingestion.embedder import embed
from src.ingestion.loader import load_pdf
from src.retrieval.vector_store import VectorStore

PERSIST_PATH = "./chroma_db"
RETRIEVAL_K = 4


@contextmanager
def stage(label: str):
    """Print '<label>: Xs' on exit. Yields a dict the caller can mutate."""
    info: dict = {}
    start = time.perf_counter()
    try:
        yield info
    finally:
        elapsed = time.perf_counter() - start
        suffix = f" ({info['detail']})" if "detail" in info else ""
        print(f"[stage] {label:<14} {elapsed:6.2f}s{suffix}", file=sys.stderr)


def collection_name_for(pdf_path: Path) -> str:
    """Deterministic collection name from the absolute PDF path.

    Same path -> same collection -> cache hit on re-runs.
    Chroma collection names must be alphanumeric/underscore/hyphen.
    """
    digest = hashlib.sha256(str(pdf_path.resolve()).encode("utf-8")).hexdigest()[:16]
    return f"pdf_{digest}"


def ingest(pdf_path: Path, store: VectorStore) -> None:
    """Load, chunk, embed, and store the PDF if the store is empty."""
    with stage("load_pdf") as info:
        pages = load_pdf(str(pdf_path))
        info["detail"] = f"{len(pages)} pages"

    with stage("chunk_pages") as info:
        chunks = chunk_pages(pages)
        info["detail"] = f"{len(chunks)} chunks"

    if not chunks:
        print("No extractable text in PDF — nothing to index.", file=sys.stderr)
        sys.exit(2)

    with stage("embed") as info:
        embeddings = embed([c["text"] for c in chunks])
        info["detail"] = f"{len(embeddings)} vectors"

    with stage("store.add") as info:
        store.add(
            texts=[c["text"] for c in chunks],
            embeddings=embeddings,
            metadatas=[c["metadata"] for c in chunks],
        )
        info["detail"] = f"total={store.count()}"


def main() -> None:
    if len(sys.argv) != 3:
        print('Usage: python -m scripts.cli <path-to-pdf> "<question>"', file=sys.stderr)
        sys.exit(1)

    pdf_path = Path(sys.argv[1])
    question = sys.argv[2]

    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    store = VectorStore(
        collection_name=collection_name_for(pdf_path),
        persist_path=PERSIST_PATH,
    )

    if store.count() == 0:
        ingest(pdf_path, store)
    else:
        print(
            f"[stage] cache hit       (reusing {store.count()} chunks in "
            f"'{store.collection_name}')",
            file=sys.stderr,
        )

    chain = build_chain(store, k=RETRIEVAL_K)

    print(f"\nQuestion: {question}\n", file=sys.stderr)
    print("Answer:")

    with stage("generate"):
        for token in chain.stream(question):
            print(token, end="", flush=True)
        print()


if __name__ == "__main__":
    main()
