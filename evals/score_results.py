"""Score an eval run and print a summary table.

Usage:
    python -m evals.score_results evals/results/run_<timestamp>.jsonl
"""

import json
import sys
from collections import defaultdict
from pathlib import Path


def retrieval_hit(result: dict) -> bool:
    """Did we retrieve a chunk from at least one expected page?"""
    expected = set(result["expected_pages"])
    if not expected:
        return True  # out_of_scope: pass by convention
    retrieved = {c["page"] for c in result.get("citations", [])}
    return bool(expected & retrieved)


def correctly_rejected(result: dict) -> bool:
    """For out_of_scope questions, did the system refuse to answer?"""
    text = (result.get("actual_answer") or "").lower()
    return any(p in text for p in [
        "don't know", "no context", "not in the document",
        "cannot find", "out of scope", "not contain",
        "does not contain", "does not mention",
    ])


def main(path: str) -> None:
    # Load this run's results
    results = [
        json.loads(line)
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]

    # Load gold set to map id → category
    gold: dict[str, dict] = {}
    for line in Path("evals/gold_set.jsonl").read_text().splitlines():
        if line.strip():
            entry = json.loads(line)
            gold[entry["id"]] = entry

    # Tally
    total = len(results)
    hits = sum(1 for r in results if retrieval_hit(r))
    errors = sum(1 for r in results if r.get("error"))

    # Category breakdown
    by_cat: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        cat = gold[r["id"]]["category"]
        by_cat[cat].append(retrieval_hit(r))

    # Out-of-scope rejection
    oos_results = [
        r for r in results if gold[r["id"]]["category"] == "out_of_scope"
    ]
    oos_rejected = sum(1 for r in oos_results if correctly_rejected(r))

    # Avg latency
    latencies = [r["latency_s"] for r in results if r.get("latency_s")]
    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0

    print(f"\n=== Eval Summary ===")
    print(f"Total questions:      {total}")
    print(f"Errors:               {errors}")
    print(f"Retrieval hit-rate:   {hits}/{total} ({100*hits/total:.1f}%)")
    print(f"OOS rejection rate:   {oos_rejected}/{len(oos_results)}")
    print(f"Avg latency:          {avg_lat:.2f}s")
    print(f"\nBy category:")
    for cat, hits_list in sorted(by_cat.items()):
        h = sum(hits_list)
        n = len(hits_list)
        pct = 100 * h / n
        print(f"  {cat:20s} {h}/{n}  ({pct:.0f}%)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m evals.score_results <results.jsonl>")
        sys.exit(1)
    main(sys.argv[1])