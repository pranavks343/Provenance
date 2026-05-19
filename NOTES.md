## Reranking experiment

- Added Cohere `rerank-v3.5` as a second-stage retriever (top-20 → top-4)
- Context precision 0.68 → 0.83, answer relevancy 0.82 → 0.87, faithfulness 0.47 → 0.49
- Retrieval hit-rate regressed: 80.0% (v1) / 86.7% (v2) → 73.3% (v3). Reranker optimizes for "useful context" by Cohere's judgment, which doesn't always coincide with the gold-page chunk.
- Trial-key rate limit (10/min) required `time.sleep(6.5)` workaround in the eval loop
- Diagnosed: faithfulness is generation-bound, not retrieval-bound
- Per-question CSV (`evals/results/run_2026-05-19_22-47.ragas.csv`) shows training-data leakage on q02, q11, q23. q23 answers "41.0 BLEU" (gold = 41.8) — number not in the retrieved chunk
