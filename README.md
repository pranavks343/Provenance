Create me a README file as per this content. 





# 📄 PDF Knowledge Assistant

A production-grade RAG system for answering questions over PDFs with grounded
citations. Built end-to-end across ingestion, retrieval, generation,
evaluation, and observability.

> **Why this exists:** Most "I built RAG" projects are demos. This one is
> measured. The system is evaluated on a 30-question gold set with Ragas
> metrics, and this README documents both successes *and* a real experimental
> regression — because real engineering includes failed experiments.

---

## ⚡ Quick demo

```bash
# CLI mode (streaming answer + citations)
python -m scripts.cli data/sample.pdf "What is multi-head attention?"

# Full stack
uvicorn src.api.main:app --reload --port 8000   # backend on :8000
streamlit run src/ui/app.py                     # UI on :8501





The Streamlit UI supports two modes: **streaming text** (SSE) and

**structured JSON** with Pydantic-validated citations.



<!-- TODO: insert a screenshot of the Streamlit UI here when you have one -->



---



## 🏗️ Architecture


PDF ─► loader ─► chunker ─► embedder ─► VectorStore (Chroma, HNSW, cosine) (pypdf) (tiktoken (OpenAI │ recursive) 3-small) │ ▼ Query ─► embed query ──────────────────► retrieval ──► LCEL chain │ ├─ ChatPromptTemplate ├─ ChatOpenAI(gpt-4o-mini) ├─ Streaming → SSE → UI └─ Structured → Pydantic Answer






Two backends, one retrieval pipeline. Caching at both the LLM and embedding

layers means repeat queries cost ~0.01× the original.



---



## 📊 Evaluation



System evaluated on a 30-question gold set covering 7 categories on

*Attention Is All You Need* (Vaswani et al., 2017).



### Baseline (v1 prompt)



| Metric                  | Score |

|-------------------------|-------|

| Retrieval hit-rate      | 80.0% (24/30) |

| Out-of-scope rejection  | 100% (2/2) |

| Ragas faithfulness      | 0.47 |

| Ragas answer relevancy  | 0.82 |

| Ragas context precision | 0.68 |

| Avg latency             | 3.9s |



### Experiment: v1 → v2 prompt



Hypothesized that stricter grounding language would improve faithfulness by

forcing the LLM to stop using its training-data knowledge of the Transformer

paper. v2 added:

- Numbered rules instead of prose

- Explicit "Never use outside knowledge"

- Fixed-string abstain ("I cannot answer this from the provided documents")

- "Do not paraphrase beyond what the context supports"



**Result: hypothesis falsified.**



| Metric                  | v1 (baseline) | v2 (stricter) | Δ |

|-------------------------|---------------|---------------|----|

| Retrieval hit-rate      | 80.0%         | // TODO       | // TODO |

| Faithfulness            | 0.47          | 0.42          | -0.05 ⚠️ |

| Answer relevancy        | 0.82          | 0.79          | -0.03 |

| Context precision       | 0.68          | 0.78          | +0.10 |

| Avg latency             | 3.9s          | 4.7s          | +0.8s |



**Finding:** the strict abstain instruction made the model hedge more often,

producing vaguer answers with fewer concrete claims — which Ragas scores as

less faithful. Context precision improved, but this metric depends on the

retriever (unchanged in v2), suggesting the gain is HNSW search variance

rather than a real signal.



**What I'd try next (out of time budget):**

- v3: keep the no-prior-knowledge rule, soften the abstain so the LLM can

  still partially answer when context covers some of the question

- Reduce `top_k` from 4 to 3 to see if smaller context blocks improve

  faithfulness via tighter relevance

- Add an "atomic claim extraction" pass before scoring to separate what the

  model said from how it said it



This is a deliberately documented failed experiment. The point isn't the

final score — it's that the evaluation harness caught a regression that a

"vibes-based" prompt change would have shipped silently.



### By question category (v1 baseline)



| Category          | Score        |

|-------------------|--------------|

| Core concept      | 6/6  (100%)  |

| Comparison        | 4/4  (100%)  |

| Out-of-scope      | 2/2  (100%)  |

| Technical detail  | 7/8  (88%)   |

| Multi-hop         | 3/4  (75%)   |

| Edge case         | 1/2  (50%)   |

| Definition        | 1/4  (25%)   |



---



## 🛠️ Setup



git clone <this-repo>
cd pdf-knowledge-bot

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Add your OPENAI_API_KEY to .env

# Smoke test
python -m scripts.cli data/sample.pdf "What is attention?"





---



## 🧱 Tech stack



- **Python 3.12**

- **OpenAI** — embeddings (`text-embedding-3-small`) + LLM (`gpt-4o-mini`)

- **ChromaDB** — persistent vector store, cosine distance, HNSW

- **LangChain (LCEL)** — declarative chain composition + streaming

- **FastAPI + uvicorn** — async backend with SSE streaming

- **Streamlit** — clickable frontend

- **Pydantic v2** — structured outputs

- **Ragas** — evaluation metrics (faithfulness, relevancy, precision)

- **tiktoken** — token counting + chunking

- **pypdf** — PDF text extraction



---



## 🔍 Production-style design choices



- **Two-layer caching:** SQLite-backed LLM response cache (`set_llm_cache`)

  + content-hash embedding dedup (`VectorStore.filter_new`). Repeat queries

  on the same PDF cost ~0.01× the original.

- **Per-process VectorStore cache:** the API caches `VectorStore` instances

  by `document_id` to avoid Chroma re-init (100–300ms) on every request.

  Trade-off: unbounded memory growth; production fix would be LRU eviction.

- **SSE for streaming:** picked over WebSockets (one-way flow only needs

  unidirectional) and long polling (one HTTP request per chunk = waste).

- **Adapter pattern over Chroma:** `VectorStore` class wraps the SDK. Swapping

  to Pinecone or pgvector = changing one file.

- **Failure isolation:** PDF pages with bad extraction are logged + skipped,

  not fatal. Eval errors stored per-row so one bad question doesn't kill the

  run.

- **Defensive validation:** token count checked before embedding API call;

  distance metric verified after Chroma collection creation.

- **Empty-context short-circuit:** if retrieval returns 0 chunks, the chain

  returns a fixed "no context" answer without spending an LLM call.



---



## 🧪 Reproducing the evals



# Run the 30-question gold set through the structured chain
python -m evals.run_evals

# Simple metrics (retrieval hit-rate + OOS rejection + per-category)
python -m evals.score_results evals/results/run_<timestamp>.jsonl

# Ragas metrics (faithfulness, answer relevancy, context precision)
# Costs ~$0.10–0.30 first run; subsequent runs hit the LLM cache
python -m evals.ragas_score evals/results/run_<timestamp>.jsonl





Gold set lives at `evals/gold_set.jsonl` — 30 questions, 7 categories.



---



## ⚠️ Known limitations / future work



These were scoped out under time pressure and documented honestly:



- **No Phase 5 (security/reliability):** no prompt-injection defenses, no PII

  redaction, no rate limiting, no fallback chains (OpenAI → Gemini), no

  tenacity retries.

- **No Phase 6 (deploy):** runs locally only. No Dockerfile, no GitHub

  Actions CI, no live URL.

- **VectorStore cache has no LRU eviction** — memory grows with document count.

- **`load_dotenv()` is CWD-dependent.** Should use a fixed path relative to

  `__file__` so the app behaves the same regardless of launch directory.

- **Citation filename in API mode** shows `{document_id}.pdf` (the hashed

  upload name) instead of the original filename. Fixed in CLI mode; pending

  in API.

- **No retrieval-quality logging** — Ragas catches it offline, but production

  systems would need per-query distance metrics in logs.

- **v2 prompt regressed faithfulness** — documented as an open experiment

  rather than reverted, because the eval harness is the artifact, not the

  prompt.



---



## 📂 Project structure


. ├── src/ │ ├── ingestion/ │ │ ├── loader.py # PDF → per-page text dicts │ │ ├── chunker.py # token-bounded recursive splitting │ │ └── embedder.py # OpenAI embeddings + token validation │ ├── retrieval/ │ │ └── vector_store.py # Chroma adapter, content-hash dedup │ ├── generation/ │ │ ├── chain.py # LCEL: streaming + structured chains │ │ └── cache.py # LLM response cache │ ├── api/ │ │ ├── main.py # FastAPI: /upload, /query (SSE), /structured-query │ │ └── schemas.py # Pydantic models │ └── ui/ │ └── app.py # Streamlit UI ├── scripts/ │ └── cli.py # end-to-end CLI ├── evals/ │ ├── gold_set.jsonl # 30-question gold set │ ├── run_evals.py # eval runner (writes JSONL results) │ ├── score_results.py # retrieval hit-rate + per-category │ └── ragas_score.py # faithfulness / relevancy / precision └── data/sample.pdf # Vaswani et al., 2017




---



## 📌 What this project taught me



- **Evals beat opinions every time.** v2 of my system prompt *felt* like an

  improvement and *measured* as a regression. Without Ragas, I would have

  shipped it.

- **Retrieval hit-rate and faithfulness measure different things.** My

  retrieval was 80% but faithfulness was 0.47 — the gap revealed that the

  generation step was pulling from training data, not the retrieved context.

- **Caching is leverage.** Two layers (LLM cache + embedding dedup) made

  experimentation cheap enough that I could iterate on prompts without

  worrying about API cost.

- **LCEL composition pays off.** Adding the `/structured-query` endpoint to

  return Pydantic objects took ~30 lines because the retrieval pipeline was

  already a Runnable; I just swapped the output parser.

- **Senior engineering is documenting failure modes, not hiding them.** This

  README has a "Known limitations" section longer than the "Tech stack"

  section. That's intentional.

```