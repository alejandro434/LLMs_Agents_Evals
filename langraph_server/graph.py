"""LangGraph server graph definition.

This module exposes the compiled router subgraph for deployment via LangGraph server.
"""

# %%
from src.graphs.router_subgraph.lgraph_builder import subgraph


# Export the compiled graph for LangGraph server
graph = subgraph


if __name__ == "__main__":
    # Simple test to verify the graph is properly loaded
    print(f"Graph loaded successfully: {graph}")
    print(f"Graph nodes: {graph.nodes}")
