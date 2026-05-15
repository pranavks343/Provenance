"""Global LLM response cache.

When enabled, identical (model, prompt) inputs return the cached response
instead of hitting the OpenAI API. Cache is SQLite-backed and survives
process restarts.

Why this exists:
- Eval runs hit the same questions repeatedly across iterations
- Demo sessions ask the same questions multiple times
- CI tests should not burn API budget on every run
"""

from pathlib import Path

from langchain_core.globals import set_llm_cache
from langchain_community.cache import SQLiteCache

DEFAULT_CACHE_PATH = ".llm_cache.sqlite"


def enable_llm_cache(path: str = DEFAULT_CACHE_PATH) -> None:
    """Install a global SQLite-backed LLM cache.

    Call this once at process startup. It mutates LangChain's global
    state, so every subsequent LLM call in the process benefits.

    Args:
        path: Filesystem path for the SQLite cache file.
    """
    # Ensure parent dir exists if path includes one
    parent = Path(path).parent
    if parent and not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)

    set_llm_cache(SQLiteCache(database_path=path))