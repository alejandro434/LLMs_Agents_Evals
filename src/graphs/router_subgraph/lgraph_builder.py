"""Build the router subgraph.

uv run -m src.graphs.router_subgraph.lgraph_builder
"""


# %%

from langgraph.graph import START, StateGraph

from src.graphs.planner_executor_subgraph.lgraph_builder import (
    subgraph as planner_executor_subgraph,
)
from src.graphs.router_subgraph.nodes_logic import router_node
from src.graphs.router_subgraph.schemas import RouterSubgraphState


builder = StateGraph(RouterSubgraphState)

builder.add_node("router_node", router_node)


async def planner_executor_subgraph_node(
    state,
) -> None:
    """Planner executor subgraph node."""
    response = await planner_executor_subgraph.ainvoke(
        {"handoff_input": state.get("user_input")}
    )
    print(f"Planner executor response: {response}")


builder.add_node("planner_executor_subgraph", planner_executor_subgraph_node)

builder.add_edge(START, "router_node")
subgraph = builder.compile()


if __name__ == "__main__":
    import asyncio

    async def test_subgraph() -> None:
        """Test the subgraph."""
        async for _ in subgraph.astream(
            {
                "user_input": "Call rag, then call react and finally call the reasoner to find the answer."
            },
            stream_mode="updates",
            debug=True,
        ):
            pass

    asyncio.run(test_subgraph())

    async def router_subgraph_node(
        state,
    ) -> None:
        """Test the subgraph."""
        response = await subgraph.ainvoke({"user_input": state.get("user_input")})
        print(f"Router response: {response}")

    asyncio.run(
        router_subgraph_node(
            state={
                "user_input": "Call rag, then call react and finally call the reasoner to find the answer."
            }
        )
    )
