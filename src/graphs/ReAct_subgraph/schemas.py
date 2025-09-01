from langgraph.graph import MessagesState
from pydantic import BaseModel, Field

from src.graphs.receptionist_subgraph.schemas import (
    UserProfileSchema,
    UserRequestExtractionSchema,
)


class SuggestedRelevantToolsOutputSchema(BaseModel):
    """Suggest relevant tools output schema."""

    suggested_tools: list[str] = Field(description="The tools that may be useful")
    tools_advisor_reasoning: str = Field(
        description="Brief, precise and concise reasoning for why these tools are relevant"
    )


class ReActSubgraphState(MessagesState):
    """ReAct graph state."""

    user_profile: UserProfileSchema = Field(
        default_factory=UserProfileSchema, description="The user profile"
    )
    user_request: UserRequestExtractionSchema = Field(
        default_factory=UserRequestExtractionSchema, description="The user request"
    )
    why_this_agent_can_help: str = Field(
        default="", description="The reason why you can help the user with the task"
    )
    suggested_tools: list[str] = Field(
        description="The tools that may be useful to help the user with the task"
    )
    tools_advisor_reasoning: str = Field(
        description="Brief, precise and concise reasoning for why suggested tools are relevant"
    )
    final_answer: str = Field(description="The final result of the task")


class ReActResponse(BaseModel):
    """Response format for the ReAct agent."""

    final_answer: str = Field(
        description="The final answer based on the results you got using the suggested tools"
    )
