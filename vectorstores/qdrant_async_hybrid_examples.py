"""Example of how to use the aquery_qdrant_hybrid function.

uv run python -m vectorstores.qdrant_async_hybrid_examples
"""

import asyncio

from vectorstores.qdrant_retrievers import aquery_qdrant_hybrid


if __name__ == "__main__":
    results = asyncio.run(
        aquery_qdrant_hybrid(
            query="software",
            k=10,
            score_threshold=0.1,
        )
    )
    print(results)
