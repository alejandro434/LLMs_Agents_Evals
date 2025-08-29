"""Concierge workflow."""

# %%
import uuid
from typing import Literal

from langgraph.graph import MessagesState
from langgraph.types import Command

from src.graphs.ReAct_subgraph.lgraph_builder import (
    graph_with_in_memory_checkpointer as react,
)
from src.graphs.receptionist_subgraph.lgraph_builder import (
    graph_with_in_memory_checkpointer as receptor_router,
)
from src.graphs.receptionist_subgraph.schemas import UserProfileSchema


class ConciergeGraphState(MessagesState):
    """Concierge state."""

    user_profile: UserProfileSchema
    task: str
    rationale_of_the_handoff: str
    selected_agent: Literal["react"]
    suggested_tools: list[str]
    tools_advisor_reasoning: str
    final_answer: str


async def receptor_router_node(state: ConciergeGraphState) -> Command[Literal["react"]]:
    """Receptor router."""
    config = {"configurable": {"thread_id": uuid.uuid4()}}

    response = await receptor_router.ainvoke(state["messages"], config)
    user_profile = response["user_profile_schema"]
    user_request = response["user_request"]
    selected_agent = response["selected_agent"]
    rationale_of_the_handoff = response["rationale_of_the_handoff"]

    return Command(
        goto=react,
        update={
            "user_profile": user_profile,
            "task": user_request,
            "rationale_of_the_handoff": rationale_of_the_handoff,
            "selected_agent": selected_agent,
        },
    )


# %%
