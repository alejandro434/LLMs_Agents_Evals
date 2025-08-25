"""Build the ReAct subgraph.

uv run -m src.graphs.ReAct_subgraph.lgraph_builder
"""


# %%

from langgraph.graph import START, MessagesState, StateGraph

from src.graphs.ReAct_subgraph.nodes_logic import react_node


builder = StateGraph(MessagesState)

builder.add_node("react_node", react_node)
builder.add_edge(START, "react_node")

subgraph = builder.compile()

# %%
if __name__ == "__main__":
    import asyncio

    async def test_subgraph() -> None:
        """Test the subgraph."""
        async for _ in subgraph.astream(
            {"messages": [{"role": "user", "content": "what is the meaning of life?"}]},
            stream_mode="updates",
            debug=True,
        ):
            pass

    asyncio.run(test_subgraph())
