"""ReAct node logic.

uv run -m src.graphs.ReAct_subgraph.nodes_logic

"""

# %%

from typing import Literal

from langchain_core.messages import HumanMessage
from langgraph.graph import END
from langgraph.types import Command

from src.graphs.ReAct_subgraph.chains import get_tools_advisor_chain, react_chain
from src.graphs.ReAct_subgraph.schemas import (
    ReActSubgraphState,
)
from src.graphs.ReAct_subgraph.tools.websearch import websearch_tool
from src.graphs.receptionist_subgraph.schemas import (
    UserProfileSchema,
    UserRequestExtractionSchema,
)
from src.utils import format_command


tools_advisor_chain = get_tools_advisor_chain()
tool_list = [websearch_tool]


async def tools_advisor_node(
    state: ReActSubgraphState,
) -> Command[Literal["react_node", END]]:
    """ReAct node."""
    missing_or_invalid: list[str] = []
    if not state.get("user_request"):
        missing_or_invalid.append("Tools advisor node: user_request is missing")
    if not state.get("user_profile"):
        missing_or_invalid.append("Tools advisor node: user_profile is missing")
    elif not isinstance(state.get("user_profile"), UserProfileSchema):
        value_type = type(state.get("user_profile")).__name__
        missing_or_invalid.append(
            f"Tools advisor node: user_profile has invalid type: {value_type}, expected UserProfileSchema"
        )
    if not state.get("why_this_agent_can_help"):
        missing_or_invalid.append(
            "Tools advisor node: why_this_agent_can_help is missing"
        )

    if missing_or_invalid:
        issues = ", ".join(missing_or_invalid)
        print(f"tools_advisor_node input issues: {issues}")
        raise ValueError(f"Missing/invalid fields: {issues}")
    task = state.get("user_request").task
    user_profile = state.get("user_profile")
    why_this_agent_can_help = state.get("why_this_agent_can_help")
    context_injection = (
        f"In order to suggest relevant or useful tools to execute the above mentioned task, you have to consider the following:\n"
        f"The user's profile is: {user_profile}\n"
        f"The reason why these tools can help the user is: {why_this_agent_can_help}\n"
        f"AVAILABLE tools names and tools descriptions are:\n"
        f"{'\n'.join([f'Name: {tool.name}, Description: {tool.description}' for tool in tool_list])}"
    )

    tools_advisor_response = await tools_advisor_chain.ainvoke(
        {"messages": [HumanMessage(content=task)]},
        runtime_context_injection=context_injection,
    )
    return Command(
        goto="react_node",
        update={
            "suggested_tools": tools_advisor_response.suggested_tools,
            "tools_advisor_reasoning": tools_advisor_response.tools_advisor_reasoning,
            "user_request": state.get("user_request"),
            "user_profile": state.get("user_profile"),
            "why_this_agent_can_help": state.get("why_this_agent_can_help"),
        },
    )


async def react_node(state: ReActSubgraphState) -> Command[Literal[END]]:
    """ReAct node."""
    missing_or_invalid: list[str] = []
    if not state.get("user_request"):
        missing_or_invalid.append("ReAct node: user_request is missing")
    if not state.get("user_profile"):
        missing_or_invalid.append("ReAct node: user_profile is missing")
    elif not isinstance(state.get("user_profile"), UserProfileSchema):
        value_type = type(state.get("user_profile")).__name__
        missing_or_invalid.append(
            f"ReAct node: user_profile has invalid type: {value_type}, expected UserProfileSchema"
        )
    if not state.get("why_this_agent_can_help"):
        missing_or_invalid.append("ReAct node: why_this_agent_can_help is missing")
    if not state.get("suggested_tools"):
        missing_or_invalid.append("ReAct node: suggested_tools is missing")
    if not state.get("tools_advisor_reasoning"):
        missing_or_invalid.append("ReAct node: tools_advisor_reasoning is missing")

    if missing_or_invalid:
        issues = ", ".join(missing_or_invalid)
        print(f"react_node input issues: {issues}")
        raise ValueError(f"Missing/invalid fields: {issues}")
    task = state.get("user_request").task
    user_profile = state.get("user_profile")
    why_this_agent_can_help = state.get("why_this_agent_can_help")
    suggested_tools = state.get("suggested_tools")
    tools_advisor_reasoning = state.get("tools_advisor_reasoning")
    react_input = f"""
    The task you have to execute is: {task}
    The user's profile is: {user_profile.model_dump_json(indent=2)}
    The reason why you can help the user is: {why_this_agent_can_help}
    The suggested tools are: {suggested_tools}
    The reasoning for why these tools may be relevant is: {tools_advisor_reasoning}

    AVAILABLE tools names and tools descriptions are:
    {"\n".join([f"Name: {tool.name}, Description: {tool.description}" for tool in tool_list])}

    """
    # Add recursion_limit to prevent GraphRecursionError
    react_response = await react_chain.ainvoke(
        {"messages": [react_input]}, 
        config={"recursion_limit": 50}
    )
    return Command(
        goto=END,
        update={
            "final_answer": react_response["structured_response"].final_answer,
        },
    )


if __name__ == "__main__":
    import asyncio

    async def test_tools_advisor_node() -> None:
        """Test receptor node."""
        state = ReActSubgraphState(
            user_request=UserRequestExtractionSchema(
                task="Find in the web job fairs events in the next 30 days.",
            ),
            user_profile=UserProfileSchema(
                name="John Doe",
                current_address="123 Main St, Anytown, USA",
                employment_status="employed",
                last_job="Software Engineer",
                last_job_location="Anytown, USA",
            ),
            why_this_agent_can_help="I can help the user find job fairs events in the next 30 days.",
        )
        response = await tools_advisor_node(state)
        print(format_command(response))

    asyncio.run(test_tools_advisor_node())

    async def test_react_node() -> None:
        """Test receptor node."""
        state = ReActSubgraphState(
            user_request=UserRequestExtractionSchema(
                task="Find in the web job fairs events in the next 30 days.",
            ),
            user_profile=UserProfileSchema(
                name="John Doe",
                current_address="123 Main St, Anytown, USA",
                employment_status="employed",
                last_job="Software Engineer",
                last_job_location="Anytown, USA",
            ),
            why_this_agent_can_help="I can help the user find job fairs events in the next 30 days.",
            suggested_tools=["websearch"],
            tools_advisor_reasoning="websearch is ideal for finding up-to-date and accurate information about upcoming job fairs within a specific time frame and location. Its advanced search capabilities, including time range filters, ensure that John Doe can discover relevant job fair events in the next 30 days near his area.",
        )
        response = await react_node(state)
        print(format_command(response))

    asyncio.run(test_react_node())
