"""Schemas for follow-up subgraph outputs."""

# %%
from typing import Literal

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field

from src.graphs.receptionist_subgraph.schemas import UserProfileSchema


class FollowupOutputSchema(BaseModel):
    """Followup output schema."""

    direct_response_to_the_user: str | None = Field(
        default=None,
        description=(
            "The direct response to the user."
            "If you have to handoff to the next agent, this should be a short, friendly transition message."
        ),
    )
    what_is_the_user_looking_for: list[str] | None = Field(
        default=None,
        description="A list of preferences, goals or requirements the user is mentioning in the conversation.",
    )
    next_agent: (
        Literal["Jobs", "Educator", "Events", "CareerCoach", "Entrepreneur"] | None
    ) = Field(default=None, description="The next agent to handoff to.")


class FollowupSubgraphState(MessagesState):
    """Followup subgraph state."""

    user_profile: UserProfileSchema = Field(default_factory=UserProfileSchema)
    guidance_for_distil_user_needs: str | None = Field(
        default=None,
        description="Guidance for the distil_user_needs node.",
    )
    next_agent: (
        Literal["Jobs", "Educator", "Events", "CareerCoach", "Entrepreneur"] | None
    ) = Field(default=None)
