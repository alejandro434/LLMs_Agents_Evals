"""Schemas for the planner subgraph."""

# %%
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field


class Step(BaseModel):
    """A step in the plan."""

    instruction: str = Field(description="The instruction to be executed in this step")
    suggested_agent: str = Field(description="The agent suggested to execute this step")
    reasoning: str = Field(description="The reasoning for this step")
    result: str = Field(description="The result of the step")
    is_complete: bool = Field(
        default_factory=lambda: False,
        description="Whether this step has been completed or not",
    )


class Plan(BaseModel):
    """A plan for the agent."""

    goal: str = Field(description="The goal of the plan based on the user's request")
    steps: list[Step] = Field(
        description="List of steps to be executed to achieve the goal"
    )

    @property
    def is_complete(self) -> bool:
        """Check if all steps are complete."""
        return all(step.is_complete for step in self.steps)


class PlannerExecutorSubgraphState(MessagesState):
    """State for the planner executor subgraph."""

    handoff_input: str = Field(description="The user's input")
    plan: Plan = Field(description="The plan for the agent")
    current_step: Step = Field(description="The current step to be executed")


if __name__ == "__main__":
    plan = Plan(
        goal="Find the answer to the question",
        steps=[
            Step(
                instruction="Find the answer to the question",
                suggested_agent="search",
                reasoning="I need to search the web to find the answer to the question",
                result="The answer to the question is 42",
                is_complete=True,
            )
        ],
    )
    print(plan.model_dump_json(indent=2))
    print(plan.is_complete)
