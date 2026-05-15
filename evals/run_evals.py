"""Evaluation runner: run the gold set through the structured chain and save results.

Usage:
    python -m evals.run_evals

Progress, errors, and summary go to stderr.
Result JSONL goes to evals/results/run_{timestamp}.jsonl.
"""
# === IMPORTS ===
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from src.ingestion.loader import load_pdf
from src.ingestion.chunker import chunk_pages
from src.ingestion.embedder import embed
from src.retrieval.vector_store import VectorStore
from src.generation.chain import build_structured_chain
from src.generation.cache import enable_llm_cache


# === CONSTANTS ===
PDF_PATH = "data/sample.pdf"
GOLD_PATH = "evals/gold_set.jsonl"
RESULTS_DIR = "evals/results"
COLLECTION_NAME = "eval_run"
PERSIST_PATH = "./chroma_db"


# === HELPERS ===
def load_gold_set(path: str) -> list[dict]:
    """Load gold_set.jsonl into a list of dicts."""
    with open(path) as f:
        raw = f.read().strip()

    if not raw:
        return []

    if raw[0] == "[":
        loaded = json.loads(raw)
        if not isinstance(loaded, list):
            raise ValueError(f"Expected a JSON array in {path}")
        return loaded

    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def ingest_pdf(pdf_path: str, store) -> None:
    """Load -> chunk -> filter_new -> embed -> add. Mirrors cli.py's ingest().

    Chunks are dicts: {"text": str, "metadata": dict}.
    filter_new takes list[str], returns (new_texts, new_indices, new_ids).
    store.add takes (texts, embeddings, metadatas, ids).
    """
    pages = load_pdf(pdf_path)
    chunks = chunk_pages(pages)

    texts = [c["text"] for c in chunks]
    new_texts, new_indices, new_ids = store.filter_new(texts)

    if not new_texts:
        print(f"  No new chunks — store already has {pdf_path}", file=sys.stderr)
        return

    new_metadatas = [chunks[i]["metadata"] for i in new_indices]
    vectors = embed(new_texts)
    store.add(
        texts=new_texts,
        embeddings=vectors,
        metadatas=new_metadatas,
        ids=new_ids,
    )
    print(f"  Added {len(new_texts)} new chunks", file=sys.stderr)


def run_one_question(chain, gold_entry: dict) -> dict:
    """Run one question through the chain, return a result dict.

    Captures: actual answer, citations, latency, error if any.
    Never raises — errors are stored in the result dict.
    """
    t0 = time.perf_counter()
    actual_answer = None
    citations: list = []
    confidence = None
    used_context = None
    error = None

    try:
        answer = chain.invoke(gold_entry["question"])
        payload = answer.model_dump()
        actual_answer = payload.get("text")  # Answer schema field is `text`, not `answer`
        citations = payload.get("citations", [])
        confidence = payload.get("confidence")
        used_context = payload.get("used_context")
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        print(
            f"  ERROR on {gold_entry['id']} ({gold_entry['question'][:60]}...): {error}",
            file=sys.stderr,
        )

    latency_s = round(time.perf_counter() - t0, 3)

    return {
        "id": gold_entry["id"],
        "question": gold_entry["question"],
        "expected_answer": gold_entry.get("expected_answer"),
        "expected_pages": gold_entry.get("expected_pages", []),
        "actual_answer": actual_answer,
        "citations": citations,
        "confidence": confidence,
        "used_context": used_context,
        "latency_s": latency_s,
        "error": error,
    }


# === MAIN ===
def main() -> None:
    # --- Setup ---
    enable_llm_cache()
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

    gold = load_gold_set(GOLD_PATH)
    total = len(gold)
    print(f"Loaded {total} questions from {GOLD_PATH}", file=sys.stderr)

    store = VectorStore(collection_name=COLLECTION_NAME, persist_path=PERSIST_PATH)

    print(f"Ingesting {PDF_PATH} ...", file=sys.stderr)
    ingest_pdf(PDF_PATH, store)

    print("Building structured chain ...", file=sys.stderr)
    chain = build_structured_chain(store)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    out_path = Path(RESULTS_DIR) / f"run_{timestamp}.jsonl"

    # --- Loop ---
    successes = 0
    failed_ids: list[str] = []
    t_start = time.perf_counter()

    with out_path.open("w") as f:
        for i, entry in enumerate(gold):
            print(
                f"[{i+1}/{total}] {entry['id']}: {entry['question'][:60]}...",
                file=sys.stderr,
            )
            result = run_one_question(chain, entry)
            f.write(json.dumps(result) + "\n")
            f.flush()  # cheap insurance: each result hits disk before the next call

            if result["error"] is None:
                successes += 1
            else:
                failed_ids.append(entry["id"])

    # --- Summary ---
    elapsed = time.perf_counter() - t_start
    avg = elapsed / total if total else 0.0
    print(file=sys.stderr)
    print(f"Ran {total} questions in {elapsed:.1f}s (avg {avg:.1f}s/q)", file=sys.stderr)
    print(f"Successes: {successes}/{total}", file=sys.stderr)
    if failed_ids:
        print(f"Errors: {len(failed_ids)} ({', '.join(failed_ids)})", file=sys.stderr)
    else:
        print("Errors: 0", file=sys.stderr)
    print(f"Output: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()