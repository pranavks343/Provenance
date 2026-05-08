"""LCEL RAG chain: question -> retrieve -> format context -> LLM answer.

Chain shape::

    str (question)
      |  RunnableParallel
      v
    {context: str, question: str}
      |  RunnableBranch (short-circuit if context is empty)
      v
    str (answer)

The empty-context branch returns a fixed string without calling the LLM,
so a query that retrieves no chunks costs zero generation tokens.
"""

from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    Runnable,
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_openai import ChatOpenAI

from src.ingestion.embedder import embed
from src.retrieval.vector_store import VectorStore

LLM_MODEL = "gpt-4o-mini"
DEFAULT_K = 4
EMPTY_CONTEXT_MESSAGE = "No context found in the indexed documents."

SYSTEM_PROMPT = (
    "You are a careful research assistant. Answer ONLY using the provided "
    "context. After each claim, cite the source inline as "
    "[Source: <source>, Page <n>]. If the context does not contain the "
    "answer, say so explicitly — do not invent facts."
)

USER_PROMPT = "Context:\n{context}\n\nQuestion: {question}"


def format_context(docs: list[dict]) -> str:
    """Render retrieved chunks as a single string with source headers.

    Source paths are reduced to their basename so that absolute paths
    (which may contain usernames or other host-specific structure) are
    not echoed back to the user via the LLM. The full path stays in
    chunk metadata for downstream debugging.

    Empty input returns an empty string, which the chain treats as a
    signal to short-circuit and skip the LLM call.
    """
    if not docs:
        return ""
    parts = []
    for d in docs:
        meta = d["metadata"]
        source_display = Path(meta["source"]).name
        header = f"[Source: {source_display}, Page {meta['page']}]"
        parts.append(f"{header}\n{d['text']}")
    return "\n\n".join(parts)


def build_chain(store: VectorStore, k: int = DEFAULT_K) -> Runnable:
    """Build the full LCEL RAG chain.

    Args:
        store: A populated VectorStore.
        k: Number of chunks to retrieve per query.

    Returns:
        A Runnable that takes a question string and yields/returns the
        answer string. Supports `.invoke()`, `.stream()`, and async
        variants.
    """

    def retrieve_and_format(question: str) -> str:
        query_vec = embed([question])[0].tolist()
        docs = store.query(query_vec, n_results=k)
        return format_context(docs)

    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", USER_PROMPT)]
    )

    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

    answer_branch = prompt | llm | StrOutputParser()
    empty_branch = RunnableLambda(lambda _: EMPTY_CONTEXT_MESSAGE)

    return (
        RunnableParallel(
            context=RunnableLambda(retrieve_and_format),
            question=RunnablePassthrough(),
        )
        | RunnableBranch(
            (lambda x: not x["context"], empty_branch),
            answer_branch,
        )
    )
