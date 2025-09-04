"""Qdrant hybrid search tool.

Exposes a LangChain StructuredTool that performs hybrid (dense + sparse)
retrieval against a Qdrant collection and returns a human-readable string
with the top results. The tool is configured to return direct output so
agent frameworks can surface the tool's response immediately.
"""

# %%
import os

from dotenv import load_dotenv
from fastembed import SparseTextEmbedding
from langchain_core.tools import StructuredTool
from openai import OpenAI
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.http import models


load_dotenv(override=True)
openai_client = OpenAI()


def pretty_print_query_results(results, top_n=10, desc_max_len=200) -> str:
    """Format Qdrant hybrid search query results into a readable string.

    Args:
        results: list of ScoredPoint objects from qdrant_client.query_points
        top_n: number of top results to include
        desc_max_len: max characters to show for description

    Returns:
        A string containing the formatted results.
    """
    lines = []
    for rank, hit in enumerate(results[:top_n], start=1):
        payload = hit.payload or {}
        job_title = payload.get("job_title", "N/A")
        company = payload.get("company", "N/A")
        city = payload.get("city", "N/A")
        state = payload.get("state", "N/A")
        salary = payload.get("salary", "N/A")
        education = payload.get("education_needed", "N/A")
        is_remote = payload.get("is_remote", False)
        url = payload.get("job_url", "N/A")
        posted_date = payload.get("posted_date", "N/A")
        description = payload.get("description", "")
        description = (
            (description[:desc_max_len] + "...")
            if len(description) > desc_max_len
            else description
        )
        score = hit.score if hasattr(hit, "score") else "N/A"

        lines.append(f"Rank {rank} | Score: {score}")
        lines.append(f"Title: {job_title}")
        lines.append(f"Company: {company}")
        lines.append(f"Location: {city}, {state} | Remote: {is_remote}")
        lines.append(f"Salary: {salary} | Education: {education}")
        lines.append(f"Posted: {posted_date}")
        lines.append(f"URL: {url}")
        lines.append(f"Description: {description}")
        lines.append("-" * 80)

    return "\n".join(lines)


def query_qdrant_hybrid(query_text: str, state: str) -> str:
    """Query Qdrant using hybrid dense + sparse retrieval and format results.

    Args:
        query_text: The query text to search for jobs.
        state: The U.S. state to filter results by.

    Returns:
        A human-readable string with the top query results.
    """
    dense_response = openai_client.embeddings.create(
        model="text-embedding-3-large",  # or ada-002 if you want smaller
        input=query_text,
    )
    dense_embedding = dense_response.data[0].embedding
    model_sparse = SparseTextEmbedding(
        model_name="Qdrant/bm42-all-minilm-l6-v2-attentions"
    )
    sparse_embedding = next(iter(model_sparse.query_embed(query_text)))
    location_filter = models.Filter(
        must=[models.FieldCondition(key="state", match=models.MatchValue(value=state))]
    )
    qdrant_client = QdrantClient(
        url=os.getenv("QDRANT_HOST_URL"), api_key=os.getenv("QDRANT_API_KEY")
    )

    results = qdrant_client.query_points(
        collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
        prefetch=[
            models.Prefetch(query=dense_embedding, using="dense", limit=10),
            models.Prefetch(query=sparse_embedding.as_object(), using="bm42", limit=10),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        query_filter=location_filter,
        limit=10,
    )

    scored_points = results.points

    return pretty_print_query_results(scored_points, top_n=10)


class QdrantHybridSearchInput(BaseModel):
    """Input schema for qdrant_hybrid_search_tool."""

    query_text: str = Field(
        description=(
            "Natural language query describing the job or content to retrieve."
        )
    )
    state: str = Field(
        description=("Two- or full-name U.S. state filter to restrict search results.")
    )


qdrant_hybrid_search_tool = StructuredTool.from_function(
    name="query_qdrant_hybrid",
    description=(
        "Search for jobs based on a query, "
        "filtered by state. Returns a readable list of the top matches."
    ),
    func=query_qdrant_hybrid,
    args_schema=QdrantHybridSearchInput,
    return_direct=True,
)


if __name__ == "__main__":
    # Simple demonstration / test
    result = qdrant_hybrid_search_tool.invoke(
        {"query_text": "HR professional", "state": "Virginia"}
    )
    print(result)
