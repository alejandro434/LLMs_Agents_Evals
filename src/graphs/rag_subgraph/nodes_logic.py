"""RAG node logic.

uv run -m src.graphs.rag_subgraph.nodes_logic

"""

# %%
from typing import TYPE_CHECKING, Literal

from langchain_core.messages import HumanMessage
from langgraph.graph import END
from langgraph.types import Command


if TYPE_CHECKING:
    from src.graphs.rag_subgraph.lgraph_builder import RAGSubgraphState
from src.graphs.planner_executor_subgraph.schemas import Step


async def rag_node(state: "RAGSubgraphState") -> Command[Literal[END]]:
    """RAG node."""
    # Handle the current_step from the state
    current_step_raw = state.get("current_step")

    current_step: Step | None
    if isinstance(current_step_raw, Step):
        current_step = current_step_raw
    elif isinstance(current_step_raw, dict):
        # Accept dict input coming from RemoteGraph JSON and coerce to Step
        current_step = Step(**current_step_raw)
    else:
        current_step = None

    if current_step:
        print(f"RAG node received step: {current_step.instruction}")
        # Process the step instruction with RAG logic
        # For now, just mark it as complete with a mock result
        current_step.result = f"PROCESSED BY RAG: {current_step.instruction}"
        current_step.is_complete = True

        return Command(
            goto=END,
            update={"current_step": current_step},
        )

    # Fallback to handle messages if no current_step
    if state.get("messages"):
        print(f"RAG node received message: {state['messages'][-1]}")
        return Command(
            goto=END,
            update={"messages": "PROCESSED BY RAG: " + state["messages"][-1].content},
        )

    # If neither current_step nor messages, return empty update
    return Command(goto=END, update={})


if __name__ == "__main__":
    import asyncio

    from src.graphs.planner_executor_subgraph.schemas import Step
    from src.graphs.rag_subgraph.lgraph_builder import RAGSubgraphState

    async def main():
        """Main function."""
        state = RAGSubgraphState(
            messages=[HumanMessage(content="hello")],
            current_step=Step(
                instruction="Search for information about Python",
                suggested_subgraph="rag",
                reasoning="Need to search for information",
                result="",
                is_complete=False,
            ),
        )
        response = await rag_node(state)
        print(f"RAG node response: {response}")

    asyncio.run(main())
