"""
Minimal embeddings playground: embed sentences, compute similarity matrix.
"""
import os
import numpy as np
import tiktoken
from openai import OpenAI, APIError, RateLimitError
from dotenv import load_dotenv


load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY not set in environment")

client = OpenAI()
MODEL = "text-embedding-3-small"
MAX_TOKENS = 8191
EMBEDDING_DIM = 1536
ENCODER = tiktoken.encoding_for_model(MODEL)


def count_tokens(text: str) -> int:
    """Count tokens in a string using the model's tokenizer.

    Args:
        text: Input string to tokenize.

    Returns:
        Number of tokens the model will see.
    """
    return len(ENCODER.encode(text))


def validate(texts: list[str]) -> None:
    """Ensure no input exceeds the model's token limit.

    Args:
        texts: List of input strings.

    Raises:
        ValueError: If any string exceeds MAX_TOKENS.
    """
    for i, t in enumerate(texts):
        n = count_tokens(t)
        if n > MAX_TOKENS:
            raise ValueError(f"input[{i}] has {n} tokens (max {MAX_TOKENS})")


def embed(texts: list[str]) -> np.ndarray:
    """Embed a batch of strings into a 2D array of vectors.

    Args:
        texts: List of input strings to embed.

    Returns:
        Array of shape (len(texts), EMBEDDING_DIM) with one row per input.
        Returns an empty (0, EMBEDDING_DIM) array if texts is empty.

    Raises:
        RuntimeError: On rate limit or API error.
        ValueError: If any input exceeds MAX_TOKENS.
    """
    if not texts:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float64)
    validate(texts)
    try:
        resp = client.embeddings.create(model=MODEL, input=texts)
    except RateLimitError as e:
        raise RuntimeError(f"rate limited: {e}")
    except APIError as e:
        raise RuntimeError(f"API error: {e}")
    return np.array([d.embedding for d in resp.data])


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine similarity in [-1, 1].
    """
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def similarity_matrix(vecs: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarities for any set of vectors.

    Normalizes rows internally, so works for any embedding model
    regardless of whether outputs are unit-length.

    Args:
        vecs: Array of shape (N, D).

    Returns:
        Array of shape (N, N) where entry [i, j] is cosine(vecs[i], vecs[j]).
    """
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    normalized = vecs / norms
    return normalized @ normalized.T


def main() -> None:
    """Sanity test: verify embeddings cluster related topics and separate unrelated ones."""
    sentences = [
        "The cat sat on the mat.",
        "A feline rested on the rug.",
        "The stock market crashed yesterday.",
        "Bitcoin prices surged overnight.",
        "Quantum mechanics describes particles.",
    ]

    vecs = embed(sentences)
    print(f"shape: {vecs.shape}")

    sim = similarity_matrix(vecs)
    print("\nSimilarity matrix:")
    print(np.round(sim, 3))

    assert int(np.argmax(sim[0, 1:])) + 1 == 1, "sentence 0 should be closest to 1"
    assert int(np.argmax(np.delete(sim[2], 2))) == 2, "sentence 2 should be closest to 3"
    assert sim[0, 1] > sim[0, 2], "intra-cluster sim must exceed inter-cluster sim"
    assert sim[0, 1] > sim[0, 4], "pets should be more similar than pet/physics"

    print("\n✓ All clustering and separation checks passed.")


if __name__ == "__main__":
    main()