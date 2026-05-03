import hashlib
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
from src.ingestion.embedder import embed


class VectorStore:
    DISTANCE_METRIC: str = "cosine"

    def __init__(self, collection_name: str, persist_path: str) -> None:
        Path(persist_path).mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=persist_path)

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": self.DISTANCE_METRIC},
        )

        existing_metric = self.collection.metadata.get("hnsw:space")
        if existing_metric != self.DISTANCE_METRIC:
            raise ValueError(
                f"Collection '{collection_name}' uses metric "
                f"'{existing_metric}', expected '{self.DISTANCE_METRIC}'. "
                f"Delete the collection or use a different name."
            )

    @staticmethod
    def _hash_id(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def add(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: Optional[list[dict]] = None,
        ids: Optional[list[str]] = None,
    ) -> None:
        n = len(texts)
        if len(embeddings) != n:
            raise ValueError(
                f"texts ({n}) and embeddings ({len(embeddings)}) length mismatch"
            )
        if metadatas is not None and len(metadatas) != n:
            raise ValueError(
                f"texts ({n}) and metadatas ({len(metadatas)}) length mismatch"
            )
        if ids is not None and len(ids) != n:
            raise ValueError(
                f"texts ({n}) and ids ({len(ids)}) length mismatch"
            )

        if ids is None:
            ids = [self._hash_id(t) for t in texts]

        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def query(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        where: Optional[dict] = None,
    ) -> list[dict]:
        raw = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        documents = raw["documents"][0]
        metadatas = raw["metadatas"][0]
        distances = raw["distances"][0]

        return [
            {"text": doc, "metadata": meta, "distance": dist}
            for doc, meta, dist in zip(documents, metadatas, distances)
        ]

    def count(self) -> int:
        return self.collection.count()


if __name__ == "__main__":
    sentences = [
        "Quantum computing uses qubits for parallel computation.",
        "QAOA is a quantum algorithm for combinatorial optimization.",
        "RAG systems retrieve documents to ground LLM responses.",
        "HNSW is a graph-based approximate nearest neighbor algorithm.",
        "Python is a popular programming language for AI development.",
    ]

    embeddings = embed(sentences)

    store = VectorStore(
        collection_name="sanity_test",
        persist_path="./chroma_db",
    )

    store.add(
        texts=sentences,
        embeddings=embeddings,
        metadatas=[
            {"topic": "quantum"},
            {"topic": "quantum"},
            {"topic": "ai"},
            {"topic": "search"},
            {"topic": "ai"},
        ],
    )

    print(f"Total docs in collection: {store.count()}")

    query_text = "How does quantum optimization work?"
    query_vec = embed([query_text])[0]

    print(f"\nQuery: {query_text}")
    print("Top 3 results:")
    for i, hit in enumerate(store.query(query_vec, n_results=3), 1):
        print(f"  {i}. dist={hit['distance']:.4f} | {hit['metadata']} | {hit['text']}")

    print("\nFiltered query (topic=quantum):")
    for i, hit in enumerate(
        store.query(query_vec, n_results=5, where={"topic": "quantum"}), 1
    ):
        print(f"  {i}. dist={hit['distance']:.4f} | {hit['text']}")