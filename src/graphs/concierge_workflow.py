"""Concierge workflow.

uv run -m src.graphs.concierge_workflow
"""

# %%
import uuid
from pprint import pprint
from typing import Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command
from pydantic import Field

from src.graphs.receptionist_subgraph.lgraph_builder import (
    graph_with_in_memory_checkpointer as receptor_router,
)
from src.graphs.receptionist_subgraph.schemas import UserProfileSchema


class ConciergeGraphState(MessagesState):
    """Concierge state."""

    user_profile: UserProfileSchema | None = Field(default=None)
    task: str | None = Field(default=None)
    rationale_of_the_handoff: str | None = Field(default=None)
    selected_agent: Literal["react"] | None = Field(default=None)
    suggested_tools: list[str] | None = Field(default=None)
    tools_advisor_reasoning: str | None = Field(default=None)
    final_answer: str | None = Field(default=None)
    # Add field to capture interrupt responses from the receptionist
    direct_response_to_the_user: str | None = Field(default=None)


async def receptor_router_node(state: ConciergeGraphState) -> Command[Literal[END]]:
    """Receptor router."""
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    response = await receptor_router.ainvoke({"messages": state["messages"]}, config)

    # Extract fields from the response, which may be partial if interrupted
    user_profile = response.get("user_profile_schema")
    user_request = response.get("user_request")
    selected_agent = response.get("selected_agent")
    rationale_of_the_handoff = response.get("rationale_of_the_handoff")

    # Also check for the direct_response_to_the_user field which is set during interrupts
    direct_response = response.get("direct_response_to_the_user")

    return Command(
        goto=END,
        update={
            "user_profile": user_profile,
            "task": user_request.task if user_request else None,
            "rationale_of_the_handoff": rationale_of_the_handoff,
            "selected_agent": selected_agent,
            "messages": [direct_response],
        },
    )


builder = StateGraph(ConciergeGraphState)

builder.add_node("receptor_router", receptor_router_node)
builder.add_edge(START, "receptor_router")
graph_with_in_memory_checkpointer = builder.compile(checkpointer=MemorySaver())

if __name__ == "__main__":
    import asyncio

    CONFIG: dict = {"configurable": {"thread_id": str(uuid.uuid4())}}

    async def test_subgraph_async_streaming() -> None:
        """Test async streaming response with SQLite persistence."""
        print("\n=== Testing Receptionist Graph with SQLite Persistence ===")
        async with AsyncSqliteSaver.from_conn_string("in-memory") as saver:
            concierge_graph = builder.compile(checkpointer=saver)
            config = {"configurable": {"thread_id": str(uuid.uuid4())}}

            test_input = {
                "messages": [
                    "Hi, I'm Jane Doe. I'm unemployed and looking for work in Maryland. "
                ]
            }

            print("\nStreaming updates:")
            async for update in concierge_graph.astream(
                test_input,
                config,
                stream_mode="updates",
                debug=False,  # Turn off debug for cleaner output
            ):
                pprint(update)

    asyncio.run(test_subgraph_async_streaming())
    # %%
