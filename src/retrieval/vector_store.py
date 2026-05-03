import hashlib
from pathlib import Path

import chromadb
import numpy as np
from src.ingestion.embedder import embed


class VectorStore:
    """Persistent Chroma-backed vector store with cosine similarity.

    Example:
        from src.ingestion.embedder import embed
        store = VectorStore("my_docs", "./chroma_db")
        store.add(texts=["hello world"], embeddings=embed(["hello world"]))
        results = store.query(query_embedding=embed(["hi"])[0], n_results=3)

    Embeddings are caller-supplied (not generated internally) so the store
    stays decoupled from any specific embedding model.
    """

    DISTANCE_METRIC: str = "cosine"

    def __init__(self, collection_name: str, persist_path: str) -> None:
        """Create or load a persistent Chroma collection.

        Raises:
            ValueError: If an existing collection of this name uses a
                different distance metric than DISTANCE_METRIC.
        """
        self.collection_name = collection_name
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
        """Generate a deterministic SHA-256 based ID for a text chunk."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def add(
        self,
        texts: list[str],
        embeddings: np.ndarray | list[list[float]],
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
    ) -> None:
        """Insert or update documents and embeddings in Chroma.

        IDs default to a 16-char SHA-256 prefix of each text, so re-adding
        the same text upserts (idempotent) rather than duplicating.

        Raises:
            ValueError: If texts, embeddings, metadatas, or ids have
                mismatched lengths.
        """
        if isinstance(embeddings, np.ndarray):
            embeddings = embeddings.tolist()

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
            ids = [self._hash_id(text) for text in texts]

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
        where: dict | None = None,
    ) -> list[dict]:
        """Search for the most similar documents using a query embedding."""
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
        """Return the number of documents in the collection."""
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

    print(f"Collection: {store.collection_name}")
    print(f"Total docs: {store.count()}")

    query_text = "How does quantum optimization work?"
    query_vec = embed([query_text])[0].tolist()

    print(f"\nQuery: {query_text}")
    print("Top 3 results (no filter):")
    for i, hit in enumerate(store.query(query_vec, n_results=3), 1):
        print(f"  {i}. dist={hit['distance']:.4f} | {hit['metadata']} | {hit['text']}")

    print("\nFiltered query (topic=quantum):")
    for i, hit in enumerate(
        store.query(query_vec, n_results=5, where={"topic": "quantum"}), 1
    ):
        print(f"  {i}. dist={hit['distance']:.4f} | {hit['text']}")

    print("\nSelf-distance check (querying with sentence 0's own embedding):")
    self_hit = store.query(embeddings[0].tolist(), n_results=1)[0]
    print(f"  dist={self_hit['distance']:.6f} (expected ~0)")