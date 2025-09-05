"""Concierge workflow.

uv run -m src.graphs.concierge_workflow
"""

# %%
from pprint import pprint
from typing import Literal

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command, interrupt
from pydantic import Field

from src.graphs.followup_node.nodes_logic import followup_node
from src.graphs.followup_node.schemas import FollowupSubgraphState
from src.graphs.qualifier.lgraph_builder import subgraph as qualify_user_subgraph
from src.graphs.qualifier.schemas import UserInfoOutputSchema
from src.graphs.ReAct_subgraph.lgraph_builder import (
    subgraph as react_subgraph,
)
from src.graphs.receptionist_subgraph.lgraph_builder import (
    subgraph as receptor_router_subgraph,
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
    user_state_zip_and_age: str | None = Field(default=None)


async def qualify_user(
    state: ConciergeGraphState,
) -> Command[Literal[END, "receptor_router"]]:
    """Qualify user."""
    response = await qualify_user_subgraph.ainvoke({"messages": state["messages"]})
    if isinstance(response.get("is_user_qualified"), bool):
        if response.get("is_user_qualified"):
            if (
                response.get("collected_user_info")
                and isinstance(
                    response.get("collected_user_info"), UserInfoOutputSchema
                )
                and response.get("collected_user_info").zip_code is not None
                and response.get("collected_user_info").age is not None
            ):
                user_state_zip_and_age = (
                    f"The user's state, zip code, and age are: "
                    f"{response.get('collected_user_info')}"
                )
                entering_message = "You qualifed!"

                return Command(
                    goto="receptor_router",
                    update={
                        "user_state_zip_and_age": user_state_zip_and_age,
                        "messages": [AIMessage(content=entering_message)],
                    },
                )
            return Command(
                goto=END, update={"messages": ["FAILED TO COLLECT USER INFO!"]}
            )
        return Command(
            goto=END,
            update={"messages": [AIMessage(content=response.get("why_not_qualified"))]},
        )
    return Command(goto=END, update={"messages": ["FAILED TO QUALIFY USER!"]})


async def receptor_router(
    state: ConciergeGraphState,
) -> Command[
    Literal[
        "receptor_router",
        "Jobs",
        "Educator",
        "Events",
        "CareerCoach",
        "Entrepreneur",
    ]
]:
    """Receptor router."""
    _msgs = list(state["messages"]) if isinstance(state.get("messages"), list) else []
    if isinstance(state.get("user_state_zip_and_age"), str):
        _msgs.append(state["user_state_zip_and_age"])

    response = await receptor_router_subgraph.ainvoke({"messages": _msgs})

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
        user_resp_to_interrupt = interrupt(direct_response)
        return Command(
            goto="receptor_router",
            update={
                "user_profile": user_profile,
                "messages": [user_resp_to_interrupt],
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
            "messages": [AIMessage(content=f"Transferring to {selected_agent}.")],
        },
    )


async def react(state: ConciergeGraphState) -> Command[Literal["ask_if_continue"]]:
    """React node."""
    # Use a consistent thread_id for the react subgraph to maintain state
    # Create a stable thread_id based on the conversation

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

    response = await react_subgraph.ainvoke(_input)
    if (
        response.get("final_answer")
        and isinstance(response.get("final_answer"), ToolMessage)
        and response.get("final_answer").content != ""
    ):
        return Command(
            goto="ask_if_continue",
            update={
                "final_answer": response.get("final_answer"),
                "messages": [AIMessage(content=response.get("final_answer").content)],
            },
        )
    return Command(goto=END, update={"messages": ["FAILED TO GET FINAL ANSWER!"]})


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
        update={
            "messages": [AIMessage(content="okey, i'll be here to help you later")]
        },
    )


async def follow_up(
    state: ConciergeGraphState,
) -> Command[Literal["receptor_router", END]]:
    """Follow up node."""
    # Ensure we pass a list to MessagesState
    last_msg = state["messages"][-1] if state.get("messages") else ""
    response = await followup_node(FollowupSubgraphState(messages=[last_msg]))
    if not response.next_agent:
        direct = response.direct_response_to_the_user
        msgs = (
            [AIMessage(content=direct)]
            if isinstance(direct, str) and direct.strip() != ""
            else []
        )
        return Command(goto=END, update={"messages": msgs})
    guidance_for_distil_user_needs = f"""
    The user has provided the following information:
    {response.what_is_the_user_looking_for}
    The suggested next agent is: {response.next_agent}
    Please distil the user's needs and handoff to the appropriate agent.
    """
    direct = response.direct_response_to_the_user
    msgs = (
        [AIMessage(content=direct)]
        if isinstance(direct, str) and direct.strip() != ""
        else []
    )
    # Map list of needs to a concise task string
    needs_list = response.what_is_the_user_looking_for or []
    if isinstance(needs_list, list):
        task_text = "; ".join(str(item) for item in needs_list if str(item).strip())
    else:
        task_text = str(needs_list)
    return Command(
        goto="receptor_router",
        update={
            "messages": msgs,
            "rationale_of_the_handoff": guidance_for_distil_user_needs,
            "task": task_text if task_text else None,
        },
    )


builder = StateGraph(ConciergeGraphState)
builder.add_node("qualify_user", qualify_user)
builder.add_node("receptor_router", receptor_router)
builder.add_node("ask_if_continue", ask_if_continue)
builder.add_node("follow_up", follow_up)
builder.add_node("Jobs", react)
builder.add_node("Educator", react)
builder.add_node("Events", react)
builder.add_node("CareerCoach", react)
builder.add_node("Entrepreneur", react)

builder.add_edge(START, "qualify_user")
graph_with_in_memory_checkpointer = builder.compile(checkpointer=MemorySaver())
graph = builder.compile()

if __name__ == "__main__":
    import asyncio
    import uuid

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    INPUT_MESSAGE_1 = (
        "i'm 22 years old. zip code is 20001. My name is Alex, and im unemployed."
    )
    INPUT_MESSAGE_2 = "My name is Alex, and im unemployed, and i'm looking for a job  as  a software engineer."
    messages = [INPUT_MESSAGE_1, INPUT_MESSAGE_2]

    async def test_concierge_workflow() -> None:
        """Test the concierge workflow."""
        for message in messages:
            next_node = graph_with_in_memory_checkpointer.get_state(config).next

            async for _ in graph_with_in_memory_checkpointer.astream(
                {"messages": [message]} if not next_node else Command(resume=message),
                config,
                stream_mode="updates",
                debug=True,
            ):
                pass

    asyncio.run(test_concierge_workflow())
