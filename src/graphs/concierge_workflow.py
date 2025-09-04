"""Concierge workflow.

uv run -m src.graphs.concierge_workflow
"""

# %%
import uuid
from pprint import pprint
from typing import Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command, interrupt
from pydantic import Field

from src.graphs.followup_node.nodes_logic import followup_node
from src.graphs.followup_node.schemas import FollowupSubgraphState
from src.graphs.ReAct_subgraph.lgraph_builder import (
    graph_with_in_memory_checkpointer as react_subgraph,
)
from src.graphs.receptionist_subgraph.lgraph_builder import (
    graph_with_in_memory_checkpointer as receptor_router_subgraph,
)
from src.graphs.receptionist_subgraph.schemas import (
    UserProfileSchema,
    UserRequestExtractionSchema,
)


class ConciergeGraphState(MessagesState):
    """Concierge state."""

    user_profile: UserProfileSchema | None = Field(default=None)
    task: str | None = Field(default=None)
    rationale_of_the_handoff: str | None = Field(default=None)
    selected_agent: (
        Literal["Jobs", "Educator", "Events", "CareerCoach", "Entrepreneur"] | None
    ) = Field(default=None)
    suggested_tools: list[str] | None = Field(default=None)
    tools_advisor_reasoning: str | None = Field(default=None)
    final_answer: str | None = Field(default=None)
    # Add field to capture interrupt responses from the receptionist
    direct_response_to_the_user: str | None = Field(default=None)


async def receptor_router(
    state: ConciergeGraphState,
) -> Command[Literal["Jobs", "Educator", "Events", "CareerCoach", "Entrepreneur"]]:
    """Receptor router."""
    # Use a consistent thread_id for the receptionist subgraph to maintain state
    # Create a stable thread_id based on the conversation to maintain state across turns
    # We'll use a hash of the first message or a fixed ID
    if state.get("messages"):
        # Use a hash of the first message to create a stable thread_id
        first_msg = str(state["messages"][0])
        import hashlib

        thread_hash = hashlib.md5(first_msg.encode()).hexdigest()[:8]
        config = {"configurable": {"thread_id": f"receptionist_{thread_hash}"}}
    else:
        config = {"configurable": {"thread_id": "receptionist_default"}}

    response = await receptor_router_subgraph.ainvoke(
        {"messages": state["messages"]}, config
    )

    # Extract fields from the response, which may be partial if interrupted
    pprint(f"receptor_router response: {response}")

    user_profile = response.get("user_profile")
    user_request = response.get("user_request")
    selected_agent = response.get("selected_agent")
    rationale_of_the_handoff = response.get("rationale_of_the_handoff")
    direct_response = response.get("direct_response_to_the_user")

    # If no agent selected (receptionist interrupted for more info), go to END
    # The direct_response will contain the question for the user
    if not selected_agent:
        return Command(
            goto=END,
            update={
                "user_profile": user_profile,
                "task": user_request.task if user_request else None,
                "rationale_of_the_handoff": rationale_of_the_handoff,
                "selected_agent": selected_agent,
                "direct_response_to_the_user": direct_response,
            },
        )

    # Agent was selected, route to the appropriate agent. Currently, only the
    # Jobs agent is implemented (ReAct subgraph). Route all selections to Jobs.
    # Route to the selected agent node if present (Jobs/Educator/Events/CareerCoach/Entrepreneur)
    next_node = (
        selected_agent
        if selected_agent
        in {"Jobs", "Educator", "Events", "CareerCoach", "Entrepreneur"}
        else "Jobs"
    )
    return Command(
        goto=next_node,
        update={
            "user_profile": user_profile,
            "task": user_request.task if user_request else None,
            "rationale_of_the_handoff": rationale_of_the_handoff,
            "selected_agent": selected_agent,
        },
    )


async def react(state: ConciergeGraphState) -> Command[Literal["ask_if_continue"]]:
    """React node."""
    # Use a consistent thread_id for the react subgraph to maintain state
    # Create a stable thread_id based on the conversation
    if state.get("messages"):
        # Use a hash of the first message to create a stable thread_id
        first_msg = str(state["messages"][0])
        import hashlib

        thread_hash = hashlib.md5(first_msg.encode()).hexdigest()[:8]
        config = {"configurable": {"thread_id": f"react_{thread_hash}"}}
    else:
        config = {"configurable": {"thread_id": "react_default"}}

    if not state.get("task"):
        raise ValueError("React node: task is missing")
    # Import the schema needed for the user_request

    # Create a UserRequestExtractionSchema object from the task string
    user_request = UserRequestExtractionSchema(task=state.get("task"))

    missing_or_invalid: list[str] = []
    if not state.get("user_profile"):
        missing_or_invalid.append("React node: user_profile is missing")
    elif not isinstance(state.get("user_profile"), UserProfileSchema):
        value_type = type(state.get("user_profile")).__name__
        missing_or_invalid.append(
            f"React node: user_profile has invalid type: {value_type}, expected UserProfileSchema"
        )
    if not state.get("rationale_of_the_handoff"):
        missing_or_invalid.append("React node: rationale_of_the_handoff is missing")

    if missing_or_invalid:
        issues = ", ".join(missing_or_invalid)
        print(f"react input issues: {issues}")
        raise ValueError(f"Missing/invalid fields: {issues}")

    _input = {
        "user_request": user_request,
        "user_profile": state.get("user_profile"),
        "why_this_agent_can_help": state.get("rationale_of_the_handoff"),
    }

    response = await react_subgraph.ainvoke(_input, config)

    pprint(f"react response: {response}")

    return Command(
        goto="ask_if_continue",
        update={
            "final_answer": response.get("final_answer"),
        },
    )


async def ask_if_continue(
    state: ConciergeGraphState,
) -> Command[Literal["follow_up", END]]:
    """Follow up node."""
    response_to_follow_up = interrupt("Do you have any other questions?")
    if response_to_follow_up == "Yes":
        return Command(
            goto="follow_up",
            update={
                "messages": [response_to_follow_up],
            },
        )
    return Command(
        goto=END,
        update={"messages": [response_to_follow_up]},
    )


async def follow_up(
    state: ConciergeGraphState,
) -> Command[Literal["follow_up", "receptor_router"]]:
    """Follow up node."""
    response = await followup_node(FollowupSubgraphState(messages=state["messages"]))
    if not response.next_agent:
        return Command(
            goto="follow_up",
            update={
                "messages": [response.direct_response_to_the_user],
            },
        )
    guidance_for_distil_user_needs = f"""
    The user has provided the following information:
    {response.what_is_the_user_looking_for}
    The suggested next agent is: {response.next_agent}
    Please distil the user's needs and handoff to the appropriate agent.
    """
    return Command(
        goto="receptor_router",
        update={
            "messages": [response.direct_response_to_the_user],
            "rationale_of_the_handoff": guidance_for_distil_user_needs,
            "user_request": response.what_is_the_user_looking_for,
        },
    )


builder = StateGraph(ConciergeGraphState)

builder.add_node("receptor_router", receptor_router)
builder.add_node("ask_if_continue", ask_if_continue)
builder.add_node("follow_up", follow_up)
builder.add_node("Jobs", react)
builder.add_node("Educator", react)
builder.add_node("Events", react)
builder.add_node("CareerCoach", react)
builder.add_node("Entrepreneur", react)

builder.add_edge(START, "receptor_router")
graph_with_in_memory_checkpointer = builder.compile(checkpointer=MemorySaver())
graph = builder.compile()

if __name__ == "__main__":
    import asyncio

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    async def test_in_memory_checkpointer_direct() -> None:
        """Test the graph_with_in_memory_checkpointer directly."""
        print("\n" + "=" * 70)
        print("TESTING IN-MEMORY CHECKPOINTER with ainvoke")
        print("=" * 70)

        test_input_1 = {
            "messages": [
                "Hi, I'm John Smith. I'm looking for a job. My zip code is 20850, I'm looking for a job in Virginia. I'm unemployed."
            ]
        }
        # test_input_2 = {
        #     "messages": ["I also interested in getting a degree in computer science."]
        # }
        _ = await graph_with_in_memory_checkpointer.ainvoke(
            test_input_1, config, debug=True
        )
        # _ = await graph_with_in_memory_checkpointer.ainvoke(
        #     test_input_2, config, debug=True
        # )

    asyncio.run(test_in_memory_checkpointer_direct())
