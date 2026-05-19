import os
import time
import cohere
time.sleep(6.5)

class CohereReranker:
    """Rerank retrieved chunks using Cohere's rerank-v3.5 model."""

    MODEL: str = "rerank-v3.5"

    def __init__(self, api_key: str | None = None) -> None:
        api_key = api_key or os.environ["COHERE_API_KEY"]
        self.client = cohere.ClientV2(api_key=api_key)

    def rerank(
        self,
        query: str,
        chunks: list[dict],
        top_k: int,
    ) -> list[dict]:
        """Return the top_k chunks sorted by Cohere relevance, each with a
        new ``rerank_score`` field added.
        """
        if not chunks:
            return []

        texts = [c["text"] for c in chunks]

        response = self.client.rerank(
            model=self.MODEL,
            query=query,
            documents=texts,
            top_n=top_k,
        )

        return [
            {**chunks[result.index], "rerank_score": result.relevance_score}
            for result in response.results
        ]
