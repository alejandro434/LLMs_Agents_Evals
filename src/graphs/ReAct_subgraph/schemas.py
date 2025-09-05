"""Schema definitions for the ReAct subgraph.

uv run -m src.graphs.ReAct_subgraph.schemas
"""

# %%
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field

from src.graphs.receptionist_subgraph.schemas import (
    UserProfileSchema,
    UserRequestExtractionSchema,
)


class SuggestedRelevantToolsOutputSchema(BaseModel):
    """Suggest relevant tools output schema."""

    job_search_query: str = Field(
        description=("The query (a natural language query) suitable for a job search.")
    )
    job_search_query_reasoning: str = Field(
        description=(
            "Brief, precise and concise reasoning for why this query is "
            "suitable for a job search."
        )
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
        default="", description="Why this agent can help the user with the task"
    )
    job_search_query: list[str] = Field(
        description=("The query (a natural language query) suitable for a job search.")
    )
    job_search_query_reasoning: str = Field(
        description=(
            "Brief, precise and concise reasoning for why this query is "
            "suitable for a job search."
        )
    )
    final_answer: str = Field(description="The final result of the task")
    direct_tool_message: str | None = Field(
        default=None,
        description="The direct tool message content extracted from the response",
    )


class ReActResponse(BaseModel):
    """Response format for the ReAct agent."""

    final_answer: str = Field(
        description=(
            "The final answer based on the results you got using the job search query"
        )
    )


if __name__ == "__main__":
    # Simple demonstration / test for schemas
    sample = SuggestedRelevantToolsOutputSchema(
        job_search_query=[
            "entry-level software developer jobs Virginia VA junior SWE roles",
        ],
        job_search_query_reasoning=(
            "Targets state and common synonyms to enrich semantic retrieval."
        ),
    )
    print(sample.model_dump_json(indent=2))
