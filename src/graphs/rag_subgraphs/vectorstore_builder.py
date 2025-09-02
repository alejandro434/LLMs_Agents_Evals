"""Vector store builder with examples for basic and hybrid search with filtering.

This module demonstrates:
1. Basic vector store setup with Qdrant
2. Document addition and similarity search
3. Hybrid search combining dense and sparse embeddings
4. Advanced metadata filtering with multiple conditions
"""

# %%
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams


# Load environment variables from .env file
env_path = Path(__file__).parent.parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
    print(f"Loaded environment variables from {env_path}")
else:
    print(f"Warning: No .env file found at {env_path}")
    print("Make sure OPENAI_API_KEY is set in your environment")


# Mock embeddings class for demonstration when OpenAI embeddings are not available
class MockEmbeddings(Embeddings):
    """Mock embeddings for demonstration purposes."""

    def __init__(self, dimension: int = 1536):
        self.dimension = dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate mock embeddings for documents."""
        # Generate deterministic embeddings based on text content
        embeddings = []
        for text in texts:
            # Create a simple hash-based embedding
            np.random.seed(hash(text) % (2**32))
            embedding = np.random.randn(self.dimension).tolist()
            embeddings.append(embedding)
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        """Generate mock embedding for a query."""
        return self.embed_documents([text])[0]


# Check if OpenAI API key is available and has embedding access
def get_embeddings(model_name: str = "text-embedding-ada-002", dimension: int = 1536):
    """Get embeddings instance, using mock if OpenAI is not available."""
    if os.getenv("OPENAI_API_KEY"):
        try:
            # Try to use OpenAI embeddings
            embeddings = OpenAIEmbeddings(model=model_name)
            # Test if it works
            embeddings.embed_query("test")
            print(f"Using OpenAI embeddings: {model_name}")
            return embeddings
        except Exception as e:
            print(f"OpenAI embeddings not available: {e}")
            print("Falling back to mock embeddings for demonstration")
    else:
        print("OPENAI_API_KEY not found, using mock embeddings for demonstration")

    return MockEmbeddings(dimension=dimension)


client = QdrantClient(":memory:")

client.create_collection(
    collection_name="demo_collection",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
)
# Get embeddings (will use mock if OpenAI is not available)
embeddings = get_embeddings()

vector_store = QdrantVectorStore(
    client=client,
    collection_name="demo_collection",
    embedding=embeddings,
)
from uuid import uuid4

from langchain_core.documents import Document


document_1 = Document(
    page_content="I had chocolate chip pancakes and scrambled eggs for breakfast this morning.",
    metadata={"source": "tweet"},
)

document_2 = Document(
    page_content="The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees Fahrenheit.",
    metadata={"source": "news"},
)

document_3 = Document(
    page_content="Building an exciting new project with LangChain - come check it out!",
    metadata={"source": "tweet"},
)

document_4 = Document(
    page_content="Robbers broke into the city bank and stole $1 million in cash.",
    metadata={"source": "news"},
)

document_5 = Document(
    page_content="Wow! That was an amazing movie. I can't wait to see it again.",
    metadata={"source": "tweet"},
)

document_6 = Document(
    page_content="Is the new iPhone worth the price? Read this review to find out.",
    metadata={"source": "website"},
)

document_7 = Document(
    page_content="The top 10 soccer players in the world right now.",
    metadata={"source": "website"},
)

document_8 = Document(
    page_content="LangGraph is the best framework for building stateful, agentic applications!",
    metadata={"source": "tweet"},
)

document_9 = Document(
    page_content="The stock market is down 500 points today due to fears of a recession.",
    metadata={"source": "news"},
)

document_10 = Document(
    page_content="I have a bad feeling I am going to get deleted :(",
    metadata={"source": "tweet"},
)

documents = [
    document_1,
    document_2,
    document_3,
    document_4,
    document_5,
    document_6,
    document_7,
    document_8,
    document_9,
    document_10,
]
uuids = [str(uuid4()) for _ in range(len(documents))]
vector_store.add_documents(documents=documents, ids=uuids)
results = vector_store.similarity_search(
    "LangChain provides abstractions to make working with LLMs easy", k=2
)
for res in results:
    print(f"* {res.page_content} [{res.metadata}]")

# %%
# HYBRID SEARCH WITH METADATA FILTERING EXAMPLE
# -----------------------------------------------
print("\n" + "=" * 60)
print("HYBRID SEARCH WITH METADATA FILTERING EXAMPLE")
print("=" * 60 + "\n")

# Import required modules for hybrid search
from langchain_qdrant import FastEmbedSparse, RetrievalMode
from qdrant_client.http.models import (
    FieldCondition,
    Filter,
    MatchValue,
    SparseVectorParams,
)


# Create a new collection for hybrid search
hybrid_client = QdrantClient(":memory:")

# Configure collection with both dense and sparse vectors
# Note: Using size=1536 for text-embedding-ada-002 compatibility
hybrid_client.create_collection(
    collection_name="hybrid_collection",
    vectors_config={"dense": VectorParams(size=1536, distance=Distance.COSINE)},
    sparse_vectors_config={"sparse": SparseVectorParams()},
)

# Initialize sparse embeddings for BM25-like search
# Note: This requires the fastembed package to be installed
try:
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
    print("Using FastEmbed sparse embeddings")
except Exception as e:
    print(f"FastEmbed not available: {e}")
    print("Using mock sparse embeddings for demonstration")
    # Use mock embeddings as fallback
    from langchain_qdrant.sparse_embeddings import SparseEmbeddings

    class MockSparseEmbeddings(SparseEmbeddings):
        """Mock sparse embeddings for demonstration."""

        def embed_documents(self, texts: list[str]):
            """Generate mock sparse embeddings."""
            from qdrant_client.models import SparseVector

            embeddings = []
            for text in texts:
                # Create simple sparse vector based on text
                np.random.seed(hash(text) % (2**32))
                indices = np.random.choice(1000, size=10, replace=False).tolist()
                values = np.random.rand(10).tolist()
                embeddings.append(SparseVector(indices=indices, values=values))
            return embeddings

        def embed_query(self, text: str):
            """Generate mock sparse embedding for query."""
            return self.embed_documents([text])[0]

    sparse_embeddings = MockSparseEmbeddings()

# Get embeddings for hybrid search
hybrid_embeddings = get_embeddings(dimension=1536)

# Create hybrid vector store
hybrid_vector_store = QdrantVectorStore(
    client=hybrid_client,
    collection_name="hybrid_collection",
    embedding=hybrid_embeddings,  # Dense embeddings
    sparse_embedding=sparse_embeddings,  # Sparse embeddings (BM25)
    retrieval_mode=RetrievalMode.HYBRID,
    vector_name="dense",
    sparse_vector_name="sparse",
)

# Create documents with richer metadata for filtering
hybrid_documents = [
    Document(
        page_content="LangChain is a powerful framework for building LLM applications",
        metadata={
            "source": "blog",
            "category": "technical",
            "year": 2024,
            "author": "Alice",
        },
    ),
    Document(
        page_content="Breaking news: Stock market reaches all-time high amid tech boom",
        metadata={
            "source": "news",
            "category": "finance",
            "year": 2024,
            "author": "Bob",
        },
    ),
    Document(
        page_content="Tutorial: Building RAG applications with LangChain and vector databases",
        metadata={
            "source": "tutorial",
            "category": "technical",
            "year": 2024,
            "author": "Charlie",
        },
    ),
    Document(
        page_content="Sports update: Local team wins championship after dramatic overtime",
        metadata={
            "source": "news",
            "category": "sports",
            "year": 2023,
            "author": "David",
        },
    ),
    Document(
        page_content="Machine learning models are transforming software development",
        metadata={
            "source": "blog",
            "category": "technical",
            "year": 2024,
            "author": "Eve",
        },
    ),
    Document(
        page_content="Recipe: How to make the perfect chocolate chip cookies",
        metadata={
            "source": "website",
            "category": "cooking",
            "year": 2023,
            "author": "Frank",
        },
    ),
]

# Add documents to hybrid vector store
hybrid_uuids = [str(uuid4()) for _ in range(len(hybrid_documents))]
hybrid_vector_store.add_documents(documents=hybrid_documents, ids=hybrid_uuids)

# Example 1: Hybrid search without filtering
print("1. HYBRID SEARCH (No filtering):")
print("-" * 40)
hybrid_results = hybrid_vector_store.similarity_search(
    "How to build applications with LangChain", k=3
)
for i, res in enumerate(hybrid_results, 1):
    print(f"{i}. {res.page_content[:80]}...")
    print(f"   Metadata: {res.metadata}\n")

# Example 2: Hybrid search with single metadata filter
print("\n2. HYBRID SEARCH WITH CATEGORY FILTER (technical only):")
print("-" * 40)
category_filter = Filter(
    must=[FieldCondition(key="metadata.category", match=MatchValue(value="technical"))]
)

filtered_results = hybrid_vector_store.similarity_search(
    "Building applications with AI", k=3, filter=category_filter
)
for i, res in enumerate(filtered_results, 1):
    print(f"{i}. {res.page_content[:80]}...")
    print(f"   Metadata: {res.metadata}\n")

# Example 3: Hybrid search with multiple metadata filters
print("\n3. HYBRID SEARCH WITH MULTIPLE FILTERS (source=blog AND year=2024):")
print("-" * 40)
multi_filter = Filter(
    must=[
        FieldCondition(key="metadata.source", match=MatchValue(value="blog")),
        FieldCondition(key="metadata.year", match=MatchValue(value=2024)),
    ]
)

multi_filtered_results = hybrid_vector_store.similarity_search(
    "Latest developments in technology", k=2, filter=multi_filter
)
for i, res in enumerate(multi_filtered_results, 1):
    print(f"{i}. {res.page_content[:80]}...")
    print(f"   Metadata: {res.metadata}\n")

# Example 4: Hybrid search with OR condition
print("\n4. HYBRID SEARCH WITH OR CONDITION (news OR tutorial):")
print("-" * 40)
or_filter = Filter(
    should=[
        FieldCondition(key="metadata.source", match=MatchValue(value="news")),
        FieldCondition(key="metadata.source", match=MatchValue(value="tutorial")),
    ]
)

or_filtered_results = hybrid_vector_store.similarity_search(
    "Latest updates and guides", k=3, filter=or_filter
)
for i, res in enumerate(or_filtered_results, 1):
    print(f"{i}. {res.page_content[:80]}...")
    print(f"   Metadata: {res.metadata}\n")

# Example 5: Similarity search with score (shows relevance scores)
print("\n5. HYBRID SEARCH WITH RELEVANCE SCORES:")
print("-" * 40)
results_with_scores = hybrid_vector_store.similarity_search_with_score(
    "LangChain framework for LLM applications",
    k=3,
    filter=category_filter,  # Only technical content
)
for i, (doc, score) in enumerate(results_with_scores, 1):
    print(f"{i}. Score: {score:.4f}")
    print(f"   Content: {doc.page_content[:70]}...")
    print(f"   Metadata: {doc.metadata}\n")


if __name__ == "__main__":
    # Simple demonstration that the module runs successfully
    print("\n" + "=" * 60)
    print("Vector store examples completed successfully!")
    print("=" * 60)
