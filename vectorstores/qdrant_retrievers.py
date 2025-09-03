"""Utility to run natural-language queries against a Qdrant collection.

Environment variables used:
- QDRANT_HOST_URL
- QDRANT_API_KEY
- QDRANT_COLLECTION_NAME (optional; defaults to "external_jobs_collection")
- QDRANT_DENSE_VECTOR_NAME (optional; e.g., "dense")
- QDRANT_DENSE_EMBEDDING_MODEL (optional; overrides resolver)
- OPENAI_API_KEY (for embeddings)

uv run python vectorstores/qdrant_retrievers.py
"""

# %%
import os
from typing import Any

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from langchain_qdrant.qdrant import QdrantVectorStoreError
from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.http import models as qmodels


try:  # Optional; provides dense 384/768-dim embeddings via FastEmbed
    from langchain_community.embeddings import FastEmbedEmbeddings
except ImportError:  # pragma: no cover - optional dependency
    FastEmbedEmbeddings = None  # type: ignore


load_dotenv(override=True)


def _get_env_dense_vector_name() -> str | None:
    return os.getenv("QDRANT_DENSE_VECTOR_NAME") or None


def _resolve_dense_vector_name(
    client: QdrantClient,
    collection_name: str,
    override: str | None = None,
) -> str:
    """Determine the correct dense vector name for a collection.

    Returns "" for unnamed single-vector collections; otherwise the provided name,
    "dense" if present, or the first available key.
    """
    if override is not None:
        return override

    info = client.get_collection(collection_name=collection_name)
    vectors_cfg = info.config.params.vectors

    if isinstance(vectors_cfg, dict):
        keys = list(vectors_cfg.keys())
        if not keys:
            return ""
        if "dense" in keys:
            return "dense"
        return keys[0]

    # Single unnamed vector
    return ""


async def _aresolve_dense_vector_name(
    client: AsyncQdrantClient,
    collection_name: str,
    override: str | None = None,
) -> str:
    if override is not None:
        return override
    info = await client.get_collection(collection_name=collection_name)
    vectors_cfg = info.config.params.vectors
    if isinstance(vectors_cfg, dict):
        keys = list(vectors_cfg.keys())
        if not keys:
            return ""
        if "dense" in keys:
            return "dense"
        return keys[0]
    return ""


def _get_collection_vector_size(
    client: QdrantClient, collection_name: str, vector_name: str
) -> int:
    info = client.get_collection(collection_name=collection_name)
    vectors_cfg = info.config.params.vectors
    if isinstance(vectors_cfg, dict):
        params = vectors_cfg.get(vector_name)
        if params is None:
            raise QdrantVectorStoreError(
                f"Vector name '{vector_name}' not found in collection "
                f"{collection_name}."
            )
        return int(params.size)
    return int(vectors_cfg.size)


def _resolve_dense_embeddings(
    client: QdrantClient,
    collection_name: str,
    vector_name: str,
) -> Any:
    """Pick an embeddings implementation that matches the stored vector size.

    Selection order:
    - If env QDRANT_DENSE_EMBEDDING_MODEL is set, use it.
      - If it starts with "text-embedding-", use OpenAIEmbeddings.
      - Otherwise, try FastEmbedEmbeddings (requires langchain-community).
    - Else, infer by collection vector size:
      - 3072 -> OpenAI "text-embedding-3-large"
      - 1536 -> OpenAI "text-embedding-3-small"
      - 768  -> FastEmbed "BAAI/bge-base-en-v1.5"
      - 384  -> FastEmbed "BAAI/bge-small-en-v1.5"

    References:
    - Qdrant filtering and hybrid search concepts: https://qdrant.tech/documentation/concepts/filtering/
    - LangChain Qdrant vector store: https://python.langchain.com/docs/integrations/vectorstores/qdrant/
    """
    override_model = os.getenv("QDRANT_DENSE_EMBEDDING_MODEL")

    def _fastembed_or_error(model: str) -> Any:
        if FastEmbedEmbeddings is None:
            raise ImportError(
                "FastEmbedEmbeddings is unavailable. Install 'langchain-community' "
                "or set QDRANT_DENSE_EMBEDDING_MODEL to an OpenAI model."
            )
        return FastEmbedEmbeddings(model_name=model)

    if override_model:
        if override_model.startswith("text-embedding-"):
            return OpenAIEmbeddings(model=override_model)
        return _fastembed_or_error(override_model)

    size = _get_collection_vector_size(client, collection_name, vector_name)
    if size >= 3000:
        return OpenAIEmbeddings(model="text-embedding-3-large")
    if size >= 1500:
        return OpenAIEmbeddings(model="text-embedding-3-small")
    if size == 768:
        return _fastembed_or_error("BAAI/bge-base-en-v1.5")
    if size == 384:
        return _fastembed_or_error("BAAI/bge-small-en-v1.5")

    raise QdrantVectorStoreError(
        "Unsupported dense vector size {size}. Set QDRANT_DENSE_EMBEDDING_MODEL "
        "to a compatible model."
    )


def _merge_filters(
    a: qmodels.Filter | None, b: qmodels.Filter | None
) -> qmodels.Filter | None:
    if a is None:
        return b
    if b is None:
        return a
    return qmodels.Filter(
        must=[*(a.must or []), *(b.must or [])],
        should=[*(a.should or []), *(b.should or [])],
        must_not=[*(a.must_not or []), *(b.must_not or [])],
    )


def _build_metadata_filter(
    *,
    candidate_uuid: str | None = None,
    equals: dict[str, Any] | None = None,
    ranges: dict[str, dict[str, float]] | None = None,
    any_in: dict[str, list[Any]] | None = None,
    must: list[qmodels.Condition] | None = None,
    should: list[qmodels.Condition] | None = None,
    must_not: list[qmodels.Condition] | None = None,
    metadata_filter: qmodels.Filter | None = None,
) -> qmodels.Filter | None:
    must_conditions: list[qmodels.Condition] = []
    should_conditions: list[qmodels.Condition] = []
    must_not_conditions: list[qmodels.Condition] = []

    if candidate_uuid:
        must_conditions.append(
            qmodels.FieldCondition(
                key="candidate_uuid",
                match=qmodels.MatchValue(value=candidate_uuid),
            )
        )

    if equals:
        for key, value in equals.items():
            # If a list is provided under equals, treat as any_in convenience
            if isinstance(value, list):
                any_in = any_in or {}
                any_in[key] = value
                continue
            must_conditions.append(
                qmodels.FieldCondition(
                    key=key,
                    match=qmodels.MatchValue(value=value),
                )
            )

    if ranges:
        for key, r in ranges.items():
            rng = qmodels.Range(
                gt=r.get("gt"), gte=r.get("gte"), lt=r.get("lt"), lte=r.get("lte")
            )
            must_conditions.append(qmodels.FieldCondition(key=key, range=rng))

    if any_in:
        match_any_cls = getattr(qmodels, "MatchAny", None)
        for key, values in any_in.items():
            if match_any_cls is not None:
                must_conditions.append(
                    qmodels.FieldCondition(key=key, match=match_any_cls(any=values))
                )
            else:
                # Fallback: at least one value must match using should
                should_conditions.extend(
                    [
                        qmodels.FieldCondition(
                            key=key, match=qmodels.MatchValue(value=v)
                        )
                        for v in values
                    ]
                )

    if must:
        must_conditions.extend(must)
    if should:
        should_conditions.extend(should)
    if must_not:
        must_not_conditions.extend(must_not)

    built = None
    if must_conditions or should_conditions or must_not_conditions:
        built = qmodels.Filter(
            must=must_conditions or None,
            should=should_conditions or None,
            must_not=must_not_conditions or None,
        )

    return _merge_filters(built, metadata_filter)


def query_qdrant(
    query: str,
    k: int = 5,
    *,
    candidate_uuid: str | None = None,
    collection_name: str | None = None,
    metadata_filter: qmodels.Filter | None = None,
    equals: dict[str, Any] | None = None,
    ranges: dict[str, dict[str, float]] | None = None,
    any_in: dict[str, list[Any]] | None = None,
    must: list[qmodels.Condition] | None = None,
    should: list[qmodels.Condition] | None = None,
    must_not: list[qmodels.Condition] | None = None,
) -> list[dict[str, Any]]:
    """Query Qdrant (dense) and return relevant chunks.

    Environment variables used: QDRANT_HOST_URL, QDRANT_API_KEY, and optionally
    QDRANT_COLLECTION_NAME (defaults to "external_jobs_collection"). Dense
    embedding selection is automatic based on the collection vector size but
    may be overridden by QDRANT_DENSE_VECTOR_NAME and/or
    QDRANT_DENSE_EMBEDDING_MODEL. Requires OPENAI_API_KEY if an OpenAI model is
    selected.

    Args:
        query: Natural-language query string.
        k: Number of results to return.
        candidate_uuid: If set, filters payload where "candidate_uuid" equals
            this value.
        collection_name: Optional override for the collection name.
        metadata_filter: Additional qmodels.Filter to combine with other
            conditions.
        equals: Mapping of field -> exact-match value.
        ranges: Mapping of field -> range dict with keys gt/gte/lt/lte.
        any_in: Mapping of field -> list of values; matches if any value
            equals.
        must: Additional conditions that must match.
        should: Additional conditions that should match.
        must_not: Additional conditions that must not match.

    Returns:
        List of {"content", "metadata", "score"} dicts.

    References:
        - Qdrant filtering: https://qdrant.tech/documentation/concepts/filtering/
        - LangChain Qdrant: https://python.langchain.com/docs/integrations/vectorstores/qdrant/
    """
    qdrant_url = os.getenv("QDRANT_HOST_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    collection = (
        collection_name
        or os.getenv("QDRANT_COLLECTION_NAME")
        or "external_jobs_collection"
    )

    if not qdrant_url:
        raise ValueError("QDRANT_HOST_URL environment variable must be set")
    if not qdrant_api_key:
        raise ValueError("QDRANT_API_KEY environment variable must be set")
    if not collection:
        raise ValueError("Qdrant collection name must be set")

    # Create Qdrant client and resolve vector name & embeddings
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    vector_name = _resolve_dense_vector_name(
        client, collection, _get_env_dense_vector_name()
    )
    embeddings = _resolve_dense_embeddings(client, collection, vector_name)
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection,
        embedding=embeddings,
        vector_name=vector_name,
    )

    q_filter = _build_metadata_filter(
        candidate_uuid=candidate_uuid,
        metadata_filter=metadata_filter,
        equals=equals,
        ranges=ranges,
        any_in=any_in,
        must=must,
        should=should,
        must_not=must_not,
    )

    results = vector_store.similarity_search_with_score(
        query=query,
        k=k,
        filter=q_filter,
    )

    results = [
        {
            "content": doc.page_content,
            "metadata": doc.metadata,
            "score": float(score),
        }
        for doc, score in results
    ]
    return results


async def aquery_qdrant(
    query: str,
    k: int = 5,
    *,
    candidate_uuid: str | None = None,
    collection_name: str | None = None,
    metadata_filter: qmodels.Filter | None = None,
    equals: dict[str, Any] | None = None,
    ranges: dict[str, dict[str, float]] | None = None,
    any_in: dict[str, list[Any]] | None = None,
    must: list[qmodels.Condition] | None = None,
    should: list[qmodels.Condition] | None = None,
    must_not: list[qmodels.Condition] | None = None,
) -> list[dict[str, Any]]:
    """Async dense query using AsyncQdrantClient.

    Embedding model is auto-selected based on collection vector size, with
    env-based overrides supported. If the chosen embeddings do not implement
    async, a thread offload is used.

    Args:
        query: Natural-language query string.
        k: Number of results to return.
        candidate_uuid: If set, filters payload where "candidate_uuid" equals
            this value.
        collection_name: Optional override for the collection name.
        metadata_filter: Additional qmodels.Filter to combine with other
            conditions.
        equals: Mapping of field -> exact-match value.
        ranges: Mapping of field -> range dict with keys gt/gte/lt/lte.
        any_in: Mapping of field -> list of values; matches if any value
            equals.
        must: Additional conditions that must match.
        should: Additional conditions that should match.
        must_not: Additional conditions that must not match.

    Returns:
        List of {"content", "metadata", "score"} dicts.
    """
    qdrant_url = os.getenv("QDRANT_HOST_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    collection = (
        collection_name
        or os.getenv("QDRANT_COLLECTION_NAME")
        or "external_jobs_collection"
    )

    if not qdrant_url:
        raise ValueError("QDRANT_HOST_URL environment variable must be set")
    if not qdrant_api_key:
        raise ValueError("QDRANT_API_KEY environment variable must be set")
    if not collection:
        raise ValueError("Qdrant collection name must be set")

    # Resolve embeddings based on stored vector size
    info_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    vector_name_info = _resolve_dense_vector_name(
        info_client, collection, _get_env_dense_vector_name()
    )
    embeddings = _resolve_dense_embeddings(info_client, collection, vector_name_info)

    q_filter = _build_metadata_filter(
        candidate_uuid=candidate_uuid,
        metadata_filter=metadata_filter,
        equals=equals,
        ranges=ranges,
        any_in=any_in,
        must=must,
        should=should,
        must_not=must_not,
    )

    # Use LangChain vector store for async search to avoid client-level API
    # differences across qdrant-client versions.
    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name=collection,
        url=qdrant_url,
        api_key=qdrant_api_key,
        retrieval_mode=RetrievalMode.DENSE,
        vector_name=vector_name_info,
    )

    docs_with_scores = await vector_store.asimilarity_search_with_score(
        query=query,
        k=k,
        filter=q_filter,
    )

    return [
        {
            "content": doc.page_content,
            "metadata": doc.metadata,
            "score": float(score),
        }
        for doc, score in docs_with_scores
    ]


def query_qdrant_hybrid(
    query: str,
    k: int = 5,
    *,
    candidate_uuid: str | None = None,
    collection_name: str | None = None,
    sparse_model_name: str = "Qdrant/bm25",
    sparse_vector_name: str = "langchain-sparse",
    metadata_filter: qmodels.Filter | None = None,
    equals: dict[str, Any] | None = None,
    ranges: dict[str, dict[str, float]] | None = None,
    any_in: dict[str, list[Any]] | None = None,
    must: list[qmodels.Condition] | None = None,
    should: list[qmodels.Condition] | None = None,
    must_not: list[qmodels.Condition] | None = None,
) -> list[dict[str, Any]]:
    """Hybrid dense+sparse search using QdrantVectorStore.

    Requires the collection to contain both dense and sparse vectors. Dense
    embeddings are chosen to match the stored vector dimensions; sparse uses
    FastEmbed BM25 by default.

    Args:
        query: Natural-language query string.
        k: Number of results to return.
        candidate_uuid: Optional filter by "candidate_uuid".
        collection_name: Optional override for the collection name.
        sparse_model_name: Name for sparse embedder (e.g., "Qdrant/bm25").
        sparse_vector_name: Name of the sparse vector in the collection.
        metadata_filter: Additional qmodels.Filter to combine with other
            conditions.
        equals: Mapping of field -> exact-match value.
        ranges: Mapping of field -> range dict with keys gt/gte/lt/lte.
        any_in: Mapping of field -> list of values; matches if any value
            equals.
        must: Additional conditions that must match.
        should: Additional conditions that should match.
        must_not: Additional conditions that must not match.

    Returns:
        List of {"content", "metadata", "score"} dicts.
    """
    qdrant_url = os.getenv("QDRANT_HOST_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    collection = (
        collection_name
        or os.getenv("QDRANT_COLLECTION_NAME")
        or "external_jobs_collection"
    )

    if not qdrant_url:
        raise ValueError("QDRANT_HOST_URL environment variable must be set")
    if not qdrant_api_key:
        raise ValueError("QDRANT_API_KEY environment variable must be set")
    if not collection:
        raise ValueError("Qdrant collection name must be set")

    # Resolve dense vector name and embeddings before constructing the store
    _client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    vector_name = _resolve_dense_vector_name(
        _client, collection, _get_env_dense_vector_name()
    )
    dense_embeddings = _resolve_dense_embeddings(_client, collection, vector_name)
    sparse_embeddings = FastEmbedSparse(model_name=sparse_model_name)

    try:
        vector_store = QdrantVectorStore.from_existing_collection(
            embedding=dense_embeddings,
            sparse_embedding=sparse_embeddings,
            collection_name=collection,
            url=qdrant_url,
            api_key=qdrant_api_key,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name=vector_name,
            sparse_vector_name=sparse_vector_name,
        )
    except QdrantVectorStoreError as exc:
        # If the collection lacks sparse vectors, fall back to dense retrieval.
        if "does not contain sparse vectors" in str(exc):
            return query_qdrant(
                query,
                k=k,
                candidate_uuid=candidate_uuid,
                collection_name=collection,
            )
        raise

    q_filter = _build_metadata_filter(
        candidate_uuid=candidate_uuid,
        metadata_filter=metadata_filter,
        equals=equals,
        ranges=ranges,
        any_in=any_in,
        must=must,
        should=should,
        must_not=must_not,
    )

    docs_with_scores = vector_store.similarity_search_with_score(
        query=query,
        k=k,
        filter=q_filter,
    )

    results: list[dict[str, Any]] = []
    for doc, score in docs_with_scores:
        results.append(
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score),
            }
        )
    return results


async def aquery_qdrant_hybrid(
    query: str,
    k: int = 5,
    *,
    candidate_uuid: str | None = None,
    collection_name: str | None = None,
    sparse_model_name: str = "Qdrant/bm25",
    sparse_vector_name: str = "langchain-sparse",
    metadata_filter: qmodels.Filter | None = None,
    equals: dict[str, Any] | None = None,
    ranges: dict[str, dict[str, float]] | None = None,
    any_in: dict[str, list[Any]] | None = None,
    must: list[qmodels.Condition] | None = None,
    should: list[qmodels.Condition] | None = None,
    must_not: list[qmodels.Condition] | None = None,
) -> list[dict[str, Any]]:
    """Async hybrid dense+sparse search using QdrantVectorStore.

    Dense embeddings are selected to match stored vector dimensions. If the
    selected embeddings implementation is sync-only, it is still compatible
    since LangChain's async methods perform the embedding step synchronously.
    """
    qdrant_url = os.getenv("QDRANT_HOST_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    collection = (
        collection_name
        or os.getenv("QDRANT_COLLECTION_NAME")
        or "external_jobs_collection"
    )

    if not qdrant_url:
        raise ValueError("QDRANT_HOST_URL environment variable must be set")
    if not qdrant_api_key:
        raise ValueError("QDRANT_API_KEY environment variable must be set")
    if not collection:
        raise ValueError("Qdrant collection name must be set")

    # Resolve dense vector name and embeddings before constructing the store
    info_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    vector_name = _resolve_dense_vector_name(
        info_client, collection, _get_env_dense_vector_name()
    )
    dense_embeddings = _resolve_dense_embeddings(info_client, collection, vector_name)
    sparse_embeddings = FastEmbedSparse(model_name=sparse_model_name)

    try:
        vector_store = QdrantVectorStore.from_existing_collection(
            embedding=dense_embeddings,
            sparse_embedding=sparse_embeddings,
            collection_name=collection,
            url=qdrant_url,
            api_key=qdrant_api_key,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name=vector_name,
            sparse_vector_name=sparse_vector_name,
        )
    except QdrantVectorStoreError as exc:
        if "does not contain sparse vectors" in str(exc):
            # Fallback to dense async retrieval
            return await aquery_qdrant(
                query,
                k=k,
                candidate_uuid=candidate_uuid,
                collection_name=collection,
            )
        raise

    q_filter = _build_metadata_filter(
        candidate_uuid=candidate_uuid,
        metadata_filter=metadata_filter,
        equals=equals,
        ranges=ranges,
        any_in=any_in,
        must=must,
        should=should,
        must_not=must_not,
    )

    docs_with_scores = await vector_store.asimilarity_search_with_score(
        query=query,
        k=k,
        filter=q_filter,
    )

    results: list[dict[str, Any]] = []
    for doc, score in docs_with_scores:
        results.append(
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score),
            }
        )
    return results


__all__ = [
    "aquery_qdrant",
    "aquery_qdrant_hybrid",
    "query_qdrant",
    "query_qdrant_hybrid",
]


if __name__ == "__main__":
    import asyncio
    import logging

    logging.basicConfig(level=logging.INFO)

    async def main():
        # Hybrid async test
        hybrid_docs = await aquery_qdrant_hybrid(
            "Baltimore, MD",
            k=3,
            any_in={"document_type": ["GENERIC", "SSN_CARD"]},
        )
        logging.info("Hybrid results: %d", len(hybrid_docs))
        logging.info(hybrid_docs)

    asyncio.run(main())
