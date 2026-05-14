# Phase 2: Streamlit
"""Streamlit UI for the PDF RAG Assistant.

Run:
    streamlit run src/ui/app.py

Assumes the FastAPI backend is running on http://localhost:8000.
"""

import json

import httpx
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="PDF RAG Assistant", page_icon="📄")
st.title("📄 PDF Knowledge Assistant")

# Session state holds the active document so Streamlit reruns don't lose it.
if "document_id" not in st.session_state:
    st.session_state.document_id = None
    st.session_state.doc_info = None


# ---------- Sidebar: upload ----------
with st.sidebar:
    st.header("Document")
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded is not None and st.button("Ingest"):
        with st.spinner("Uploading and indexing..."):
            try:
                files = {"file": (uploaded.name, uploaded.getvalue(), "application/pdf")}
                r = httpx.post(f"{API_URL}/upload", files=files, timeout=120.0)
                r.raise_for_status()
                data = r.json()
                st.session_state.document_id = data["document_id"]
                st.session_state.doc_info = data
                pages_str = "cached" if data["pages"] == -1 else str(data["pages"])
                st.success(f"Indexed: {data['chunks']} chunks ({pages_str} pages)")
            except httpx.HTTPError as e:
                st.error(f"Upload failed: {e}")

    if st.session_state.doc_info:
        st.markdown("**Active document**")
        st.code(st.session_state.document_id, language=None)
        st.caption(f"{st.session_state.doc_info['chunks']} chunks")


# ---------- Main: query ----------
if not st.session_state.document_id:
    st.info("Upload a PDF in the sidebar to get started.")
    st.stop()

mode = st.radio("Output mode", ["Streaming text", "Structured JSON"], horizontal=True)
question = st.text_input("Ask a question about the document:")


def parse_sse_stream(response: httpx.Response):
    """Yield token strings from a Server-Sent Events response.

    The server sends 'data: {"token": "..."}\\n\\n' per event.
    Strip the 'data: ' prefix, parse the JSON, stop on {"done": True}.
    """
    for raw_line in response.iter_lines():
        if not raw_line or not raw_line.startswith("data: "):
            continue
        payload = json.loads(raw_line[len("data: "):])
        if payload.get("done"):
            return
        if "error" in payload:
            raise RuntimeError(payload["error"])
        if "token" in payload:
            yield payload["token"]


if st.button("Ask") and question:
    body = {"document_id": st.session_state.document_id, "question": question}

    if mode == "Streaming text":
        try:
            with httpx.stream("POST", f"{API_URL}/query", json=body, timeout=120.0) as r:
                r.raise_for_status()
                st.write_stream(parse_sse_stream(r))
        except Exception as e:
            st.error(f"Query failed: {e}")

    else:  # Structured JSON
        with st.spinner("Thinking..."):
            try:
                r = httpx.post(
                    f"{API_URL}/structured-query", json=body, timeout=120.0
                )
                r.raise_for_status()
                answer = r.json()
                st.markdown(f"**Answer:** {answer['text']}")
                st.markdown(f"**Confidence:** `{answer['confidence']}`")
                st.markdown(f"**Used context:** `{answer['used_context']}`")
                if answer["citations"]:
                    st.markdown("**Citations:**")
                    for c in answer["citations"]:
                        st.markdown(
                            f"- *{c['source']}*, p.{c['page']}: \"{c['quote']}\""
                        )
                else:
                    st.caption("No citations returned.")
                with st.expander("Raw JSON"):
                    st.json(answer)
            except Exception as e:
                st.error(f"Query failed: {e}")