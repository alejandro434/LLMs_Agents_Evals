"""Build the Reasoner subgraph.

uv run -m src.graphs.reasoner_subgraph.lgraph_builder
"""


# %%

from langgraph.graph import START, MessagesState, StateGraph

from src.graphs.reasoner_subgraph.nodes_logic import reasoner_node


builder = StateGraph(MessagesState)

builder.add_node("reasoner_node", reasoner_node)
builder.add_edge(START, "reasoner_node")

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
