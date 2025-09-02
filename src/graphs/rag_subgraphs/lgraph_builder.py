"""Build the RAG subgraph.

uv run -m src.graphs.rag_subgraph.lgraph_builder
"""


# %%

from langgraph.graph import START, MessagesState, StateGraph
from pydantic import Field

from src.graphs.planner_executor_subgraph.schemas import Step
from src.graphs.rag_subgraph.nodes_logic import rag_node


class RAGSubgraphState(MessagesState):
    """RAG subgraph state."""

    current_step: Step = Field(description="The current step to be executed")


builder = StateGraph(RAGSubgraphState)

builder.add_node("rag_node", rag_node)
builder.add_edge(START, "rag_node")

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

    async def rag_subgraph_node(
        state,
    ) -> None:
        """Test the subgraph."""
        response = await subgraph.ainvoke({"current_step": state.get("current_step")})
        print(f"RAG response: {response}")

    asyncio.run(
        rag_subgraph_node(
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
