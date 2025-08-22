"""Build the router subgraph.

uv run -m src.graphs.router_subgraph.lgraph_builder
"""


# %%

from langgraph.graph import START, StateGraph

from src.graphs.router_subgraph.nodes_logic import router_node
from src.graphs.router_subgraph.schemas import RouterSubgraphState


builder = StateGraph(RouterSubgraphState)

builder.add_node("router_node", router_node)
builder.add_edge(START, "router_node")
subgraph = builder.compile()


if __name__ == "__main__":
    import asyncio

    async def test_subgraph() -> None:
        """Test the subgraph."""
        async for _ in subgraph.astream(
            {"user_input": "what is the meaning of life?"},
            stream_mode="updates",
            debug=True,
        ):
            pass

    asyncio.run(test_subgraph())
