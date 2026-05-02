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
        Array of shape (len(texts), 1536) with one row per input.

    Raises:
        RuntimeError: On rate limit or API error.
        ValueError: If any input exceeds MAX_TOKENS.
    """
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
    """Compute pairwise cosine similarities for pre-normalized vectors.

    Args:
        vecs: Array of shape (N, D) with L2-normalized rows.

    Returns:
        Array of shape (N, N) where entry [i, j] is cosine(vecs[i], vecs[j]).
    """
    return vecs @ vecs.T


def main() -> None:
    """Sanity test: embed 5 sentences and verify expected similarity structure."""
    sentences = [
        "The cat sat on the mat.",
        "A feline rested on the rug.",
        "Quantum computers use qubits.",
        "Machine learning models need data.",
        "Neural networks learn from examples.",
    ]

    vecs = embed(sentences)
    print(f"shape: {vecs.shape}")

    sim = similarity_matrix(vecs)
    print("\nSimilarity matrix:")
    print(np.round(sim, 3))

    closest_to_0 = int(np.argmax(sim[0, 1:])) + 1
    assert closest_to_0 == 1, f"expected sentence 1 closest to 0, got {closest_to_0}"
    print(f"\n✓ Sentence 1 is most similar to sentence 0 (cosine = {sim[0, 1]:.3f})")


if __name__ == "__main__":
    main()