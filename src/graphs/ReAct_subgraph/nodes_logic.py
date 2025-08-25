"""ReAct node logic.

uv run -m src.graphs.ReAct_subgraph.nodes_logic

"""

# %%

from typing import TYPE_CHECKING, Literal

from langchain_core.messages import HumanMessage
from langgraph.graph import END
from langgraph.types import Command


if TYPE_CHECKING:
    from src.graphs.ReAct_subgraph.lgraph_builder import ReActSubgraphState


async def react_node(state: "ReActSubgraphState") -> Command[Literal[END]]:
    """ReAct node."""
    # Handle the current_step from the state
    current_step = state.get("current_step")

    if current_step:
        print(f"ReAct node received step: {current_step.instruction}")
        # Process the step instruction with RAG logic
        # For now, just mark it as complete with a mock result
        current_step.result = f"PROCESSED BY REACT: {current_step.instruction}"
        current_step.is_complete = True

        return Command(
            goto=END,
            update={"current_step": current_step},
        )

    # Fallback to handle messages if no current_step
    if state.get("messages"):
        print(f"ReAct node received message: {state['messages'][-1]}")
        return Command(
            goto=END,
            update={"messages": "PROCESSED BY REACT: " + state["messages"][-1].content},
        )

    # If neither current_step nor messages, return empty update
    return Command(goto=END, update={})


if __name__ == "__main__":
    import asyncio

    from src.graphs.planner_executor_subgraph.schemas import Step
    from src.graphs.ReAct_subgraph.lgraph_builder import ReActSubgraphState

    async def main():
        """Main function."""
        state = ReActSubgraphState(
            messages=[HumanMessage(content="hello")],
            current_step=Step(
                instruction="Search for information about Python",
                suggested_subgraph="rag",
                reasoning="Need to search for information",
                result="",
                is_complete=False,
            ),
        )
        response = await react_node(state)
        print(f"ReAct node response: {response}")

    asyncio.run(main())
