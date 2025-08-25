"""Build the planner subgraph.

uv run -m src.graphs.planner_executor_subgraph.lgraph_builder
"""


# %%

from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from src.graphs.planner_executor_subgraph.nodes_logic import executor_node, planner_node
from src.graphs.planner_executor_subgraph.schemas import (
    PlannerExecutorSubgraphState,
)


builder = StateGraph(PlannerExecutorSubgraphState)

builder.add_node("planner_node", planner_node)
builder.add_node("executor_node", executor_node)


async def rag_subgraph_node(
    state: PlannerExecutorSubgraphState,
) -> Command[Literal[END]]:
    """RAG subgraph node."""
    current_step = state.get("current_step")
    print(f"Current step: {current_step}")

    return Command(goto=END, update={})


async def react_subgraph_node(
    state: PlannerExecutorSubgraphState,
) -> Command[Literal[END]]:
    """React subgraph node."""
    current_step = state.get("current_step")
    print(f"Current step: {current_step}")

    return Command(goto=END, update={})


async def reasoner_subgraph_node(
    state: PlannerExecutorSubgraphState,
) -> Command[Literal[END]]:
    """Reasoner subgraph node."""
    current_step = state.get("current_step")
    print(f"Current step: {current_step}")

    return Command(goto=END, update={})


builder.add_node("rag_subgraph", rag_subgraph_node)
builder.add_node("react_subgraph", react_subgraph_node)
builder.add_node("reasoner_subgraph", reasoner_subgraph_node)

builder.add_edge(START, "planner_node")


subgraph = builder.compile()


if __name__ == "__main__":
    import asyncio

    async def test_planner_executor_subgraph() -> None:
        """Test the planner executor subgraph."""
        test_input = {
            "handoff_input": "Search for information about Python and then analyze it",
            "messages": [],
        }

        print("Testing planner executor subgraph...")
        print(f"Input: {test_input['handoff_input']}")
        print("-" * 50)

        # Stream through the subgraph to see each step
        async for chunk in subgraph.astream(test_input, stream_mode="updates"):
            for node_name, node_output in chunk.items():
                print(f"\nNode: {node_name}")
                if node_output is not None:
                    if "plan" in node_output:
                        print(
                            f"Plan updated: {node_output['plan'].model_dump_json(indent=2)}"
                        )
                    if "current_step" in node_output:
                        print(
                            f"Current step: {node_output['current_step'].model_dump_json(indent=2)}"
                        )
                else:
                    print("Node completed (END)")

    asyncio.run(test_planner_executor_subgraph())
