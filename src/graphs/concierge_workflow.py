"""Concierge workflow."""

# %%
from langgraph.graph import MessagesState

from src.graphs.receptionist_subgraph.schemas import UserProfileSchema


class ConciergeGraphState(MessagesState):
    """Concierge state."""

    user_profile: UserProfileSchema
    task: str
    why_this_agent_can_help: str
    suggested_tools: list[str]
    tools_advisor_reasoning: str
    final_answer: str


# %%
