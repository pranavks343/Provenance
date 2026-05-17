"""Score eval results with Ragas (faithfulness, answer relevancy, context precision).

Ragas uses an LLM as a judge to score answers, so each evaluation costs API tokens.
The LLM cache from Phase 3 dramatically reduces re-run cost.

Usage:
    python -m evals.ragas_score evals/results/run_<timestamp>.jsonl
"""

import json
import sys
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    )
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv
load_dotenv()


def load_results(path: str) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]


def to_ragas_dataset(results: list[dict]) -> Dataset:
    """Convert our result schema to Ragas's expected fields.

    Ragas expects:
        - question: the user's question
        - answer: the system's answer
        - contexts: list of retrieved chunks (strings)
        - ground_truth: the expected answer
    """
    rows = []
    for r in results:
        if r.get("error"):
            continue  # skip errored questions
        # Citations' "quote" field is our retrieved context
        contexts = [c["quote"] for c in r.get("citations", [])]
        if not contexts:
            contexts = ["No context retrieved."]  # Ragas requires non-empty
        rows.append({
            "question": r["question"],
            "answer": r["actual_answer"] or "",
            "contexts": contexts,
            "ground_truth": r["expected_answer"] or "",
        })
    return Dataset.from_list(rows)


def main(path: str) -> None:
    print(f"Loading results from {path}")
    results = load_results(path)
    print(f"  {len(results)} results loaded")

    print("Converting to Ragas dataset...")
    dataset = to_ragas_dataset(results)
    print(f"  {len(dataset)} questions to score (errored questions skipped)")

    print("\nRunning Ragas evaluation (this hits OpenAI — uses LLM cache when possible)...")
    scores = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        embeddings=OpenAIEmbeddings(model="text-embedding-3-small"),
    )

    print("\n=== Ragas Scores ===")
    print(scores)

    # Save detailed per-question scores for later analysis
    df = scores.to_pandas()
    out_path = Path(path).with_suffix(".ragas.csv")
    df.to_csv(out_path, index=False)
    print(f"\nDetailed scores saved to: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m evals.ragas_score <results.jsonl>")
        sys.exit(1)
    main(sys.argv[1])