"""LCEL RAG chains for streaming text and structured JSON answers.

Chain shape:

    str (question)
      |  RunnableParallel
      v
    {context: str, question: str}
      |  RunnableBranch (short-circuit if context is empty)
      v
    str (answer) or Answer (structured)

The empty-context branch returns a fixed response without calling the LLM,
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

from src.api.schemas import Answer
from src.ingestion.embedder import embed
from src.retrieval.vector_store import VectorStore

LLM_MODEL = "gpt-4o-mini"
DEFAULT_K = 4
EMPTY_CONTEXT_MESSAGE = "No context found in the indexed documents."

SYSTEM_PROMPT = (
    "You are a careful research assistant. Answer ONLY using the provided "
    "context. After each claim, cite the source inline as "
    "[Source: <source>, Page <n>]. If the context does not contain the "
    "answer, say so explicitly; do not invent facts."
)

USER_PROMPT = "Context:\n{context}\n\nQuestion: {question}"

STRUCTURED_SYSTEM_PROMPT = (
    "You are a careful research assistant. Answer ONLY using the provided "
    "context. Cite the exact source and page for each non-trivial claim. "
    "If the context does not contain the answer, say so explicitly, return "
    "no citations, set confidence to low, and set used_context to false."
)


def format_context(docs: list[dict]) -> str:
    """Render retrieved chunks as one string with source headers."""
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
    """Build the LCEL RAG chain that returns plain text."""

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


def build_structured_chain(store: VectorStore, k: int = DEFAULT_K) -> Runnable:
    """Build the LCEL RAG chain that returns an Answer schema."""

    def retrieve_and_format(question: str) -> str:
        embedding = embed([question])[0].tolist()
        docs = store.query(embedding, n_results=k)
        return format_context(docs)

    prompt = ChatPromptTemplate.from_messages(
        [("system", STRUCTURED_SYSTEM_PROMPT), ("human", USER_PROMPT)]
    )

    llm = ChatOpenAI(model=LLM_MODEL, temperature=0).with_structured_output(
        Answer
    )

    answer_branch = prompt | llm
    empty_branch = RunnableLambda(
        lambda _: Answer(
            text=EMPTY_CONTEXT_MESSAGE,
            citations=[],
            confidence="low",
            used_context=False,
        )
    )

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
