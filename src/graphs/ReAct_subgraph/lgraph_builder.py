"""Build the ReAct subgraph.

uv run -m src.graphs.ReAct_subgraph.lgraph_builder
"""


# %%

from langgraph.graph import START, MessagesState, StateGraph
from pydantic import Field

from src.graphs.planner_executor_subgraph.schemas import Step
from src.graphs.ReAct_subgraph.nodes_logic import react_node


class ReActSubgraphState(MessagesState):
    """ReAct subgraph state."""

    current_step: Step = Field(description="The current step to be executed")


builder = StateGraph(ReActSubgraphState)

builder.add_node("react_node", react_node)
builder.add_edge(START, "react_node")

subgraph = builder.compile()

if __name__ == "__main__":
    import asyncio

    async def test_subgraph() -> None:
        """Test the subgraph."""
        async for _ in subgraph.astream(
            {
                "current_step": Step(
                    instruction="Search for information about Python and then analyze it",
                    suggested_subgraph="rag",
                    reasoning="I need to search the web to find the answer to the question",
                    result="The answer to the question is 42",
                    is_complete=False,
                )
            },
            stream_mode="updates",
            debug=True,
        ):
            pass

    asyncio.run(test_subgraph())

    async def react_subgraph_node(
        state,
    ) -> None:
        """Test the subgraph."""
        response = await subgraph.ainvoke({"current_step": state.get("current_step")})
        print(f"ReAct response: {response}")

    asyncio.run(
        react_subgraph_node(
            state={
                "current_step": Step(
                    instruction="Search for information about Python and then analyze it",
                    suggested_subgraph="rag",
                    reasoning="I need to search the web to find the answer to the question",
                    result="The answer to the question is 42",
                    is_complete=False,
                )
            }
        )
    )
