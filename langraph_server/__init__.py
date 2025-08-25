"""LangGraph server module for BuildWithin Evals.

This module provides the configuration and setup for running the router subgraph
as a LangGraph server.
"""

# %%
from langraph_server.graph import graph


__all__ = ["graph"]


if __name__ == "__main__":
    print("LangGraph server module loaded successfully")
