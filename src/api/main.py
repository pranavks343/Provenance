"""FastAPI app: PDF upload + streaming query + structured query."""

import hashlib
import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from src.api.schemas import Answer, QueryRequest, UploadResponse
from src.generation.chain import build_chain, build_structured_chain

from src.ingestion.chunker import chunk_pages
from src.ingestion.embedder import embed
from src.ingestion.loader import load_pdf
from src.retrieval.vector_store import VectorStore

UPLOAD_DIR = Path("data/uploads")
PERSIST_PATH = "./chroma_db"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB

# Process-wide cache of VectorStore instances keyed by document_id.
# Avoids re-initializing the Chroma client on every request.
_stores: dict[str, VectorStore] = {}


def get_store(document_id: str) -> VectorStore:
    """Return a cached VectorStore, creating if needed."""
    if document_id not in _stores:
        _stores[document_id] = VectorStore(
            collection_name=f"pdf_{document_id}",
            persist_path=PERSIST_PATH,
        )
    return _stores[document_id]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown: ensure upload dir exists, clear cache on exit."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    yield
    _stores.clear()


app = FastAPI(title="PDF RAG Assistant", lifespan=lifespan)

# CORS: open for local dev. Restrict in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)) -> UploadResponse:
    """Accept a PDF, ingest if new, return its document_id."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "File must be a .pdf")

    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            400, f"File exceeds max size of {MAX_UPLOAD_BYTES} bytes"
        )

    # Content-hash as document_id → identical PDFs dedupe automatically.
    document_id = hashlib.sha256(contents).hexdigest()[:16]
    saved_path = UPLOAD_DIR / f"{document_id}.pdf"
    saved_path.write_bytes(contents)

    store = get_store(document_id)

    # Cache hit: skip ingestion entirely.
    if store.count() > 0:
        return UploadResponse(
            document_id=document_id, pages=-1, chunks=store.count()
        )

    pages = load_pdf(str(saved_path))
    if not pages:
        raise HTTPException(400, "PDF has no extractable text")

    chunks = chunk_pages(pages)
    embeddings = embed([c["text"] for c in chunks])
    store.add(
        texts=[c["text"] for c in chunks],
        embeddings=embeddings,
        metadatas=[c["metadata"] for c in chunks],
    )

    return UploadResponse(
        document_id=document_id, pages=len(pages), chunks=len(chunks)
    )


@app.post("/query")
async def query(request: QueryRequest) -> StreamingResponse:
    """Stream the answer as Server-Sent Events."""
    store = get_store(request.document_id)
    if store.count() == 0:
        raise HTTPException(404, f"Document {request.document_id} not found")

    chain = build_chain(store)

    async def event_stream():
        try:
            # LCEL chains expose .astream() for async streaming.
            async for token in chain.astream(request.question):
                # SSE format: 'data: <json>\n\n' — double newline is mandatory.
                payload = json.dumps({"token": token})
                yield f"data: {payload}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/structured-query", response_model=Answer)
async def structured_query(request: QueryRequest) -> Answer:
    """Return a structured Answer object (no streaming — JSON is atomic)."""
    store = get_store(request.document_id)
    if store.count() == 0:
        raise HTTPException(404, f"Document {request.document_id} not found")

    chain = build_structured_chain(store)
    try:
        return await chain.ainvoke(request.question)
    except Exception as e:
        raise HTTPException(
            502, f"LLM failed to produce structured output: {e}"
        )


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "cached_documents": len(_stores)}
