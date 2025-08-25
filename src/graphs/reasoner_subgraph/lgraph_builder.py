"""Build the Reasoner subgraph.

uv run -m src.graphs.reasoner_subgraph.lgraph_builder
"""


# %%

from langgraph.graph import START, MessagesState, StateGraph
from pydantic import Field

from src.graphs.planner_executor_subgraph.schemas import Step
from src.graphs.reasoner_subgraph.nodes_logic import reasoner_node


class ReasonerSubgraphState(MessagesState):
    """Reasoner subgraph state."""

    current_step: Step = Field(description="The current step to be executed")


builder = StateGraph(ReasonerSubgraphState)

builder.add_node("reasoner_node", reasoner_node)
builder.add_edge(START, "reasoner_node")

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

    async def reasoner_subgraph_node(
        state,
    ) -> None:
        """Test the subgraph."""
        response = await subgraph.ainvoke({"current_step": state.get("current_step")})
        print(f"Reasoner response: {response}")

    asyncio.run(
        reasoner_subgraph_node(
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
