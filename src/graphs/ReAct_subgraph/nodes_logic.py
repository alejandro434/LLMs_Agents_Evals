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
from src.graphs.ReAct_subgraph.tools.qdrant_hybrid_search import (
    qdrant_hybrid_search_tool,
)
from src.graphs.receptionist_subgraph.schemas import (
    UserProfileSchema,
    UserRequestExtractionSchema,
)
from src.utils import format_command


tools_advisor_chain = get_tools_advisor_chain()
tool_list = [qdrant_hybrid_search_tool]


async def tools_advisor_node(
    state: ReActSubgraphState,
) -> Command[Literal["react_node", END]]:
    """ReAct node."""
    missing_or_invalid: list[str] = []
    if not state.get("user_request"):
        missing_or_invalid.append(
            "Job search query advisor node: user_request is missing"
        )
    if not state.get("user_profile"):
        missing_or_invalid.append(
            "Job search query advisor node: user_profile is missing"
        )
    elif not isinstance(state.get("user_profile"), UserProfileSchema):
        value_type = type(state.get("user_profile")).__name__
        missing_or_invalid.append(
            f"Job search query advisor node: user_profile has invalid type: {value_type}, expected UserProfileSchema"
        )
    # Allow fallback to receptionist rationale field for integration
    reason = state.get("why_this_agent_can_help") or state.get(
        "rationale_of_the_handoff"
    )
    if not reason:
        missing_or_invalid.append(
            "Job search query advisor node: why_this_agent_can_help/rationale_of_the_handoff is missing"
        )

    if missing_or_invalid:
        issues = ", ".join(missing_or_invalid)
        print(f"job_search_query_advisor_node input issues: {issues}")
        raise ValueError(f"Missing/invalid fields: {issues}")
    task = state.get("user_request").task
    user_profile = state.get("user_profile")
    why_this_agent_can_help = reason
    input_message = (
        f"This is the requested task: {task}\n"
        f"Re-formulate the task as a query (a natural language query) suitable for a job search.\n"
        f"The query should allow filtering by one state, present synonyms to enrich semantic search.\n"
    )
    context_injection = (
        f"When making the query consider:\n"
        f"The user's profile is: {user_profile}\n"
        f"The reason why a job search can help the user is: {why_this_agent_can_help}\n"
        f"AVAILABLE tools names and tools descriptions for jobs search are:\n"
        f"{'\n'.join([f'Name: {tool.name}, Description: {tool.description}' for tool in tool_list])}"
    )

    tools_advisor_response = await tools_advisor_chain.ainvoke(
        {
            "input": input_message,
            "runtime_context_injection": context_injection,
        }
    )
    return Command(
        goto="react_node",
        update={
            "job_search_query": tools_advisor_response.job_search_query,
            "job_search_query_reasoning": tools_advisor_response.job_search_query_reasoning,
            "user_request": state.get("user_request"),
            "user_profile": state.get("user_profile"),
            "why_this_agent_can_help": why_this_agent_can_help,
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
    reason = state.get("why_this_agent_can_help") or state.get(
        "rationale_of_the_handoff"
    )
    if not reason:
        missing_or_invalid.append(
            "ReAct node: why_this_agent_can_help/rationale_of_the_handoff is missing"
        )
    if not state.get("job_search_query"):
        missing_or_invalid.append("ReAct node: job_search_query is missing")
    if not state.get("job_search_query_reasoning"):
        missing_or_invalid.append("ReAct node: job_search_query_reasoning is missing")

    if missing_or_invalid:
        issues = ", ".join(missing_or_invalid)
        print(f"react_node input issues: {issues}")
        raise ValueError(f"Missing/invalid fields: {issues}")

    job_search_query = state.get("job_search_query")
    job_search_query_reasoning = state.get("job_search_query_reasoning")
    react_input = f"""
    The task you have to execute is: {job_search_query}
    The reasoning for why this query is suitable for a job search is: {job_search_query_reasoning}
    AVAILABLE tools names and tools descriptions for jobs search are:
    {"\n".join([f"Name: {tool.name}, Description: {tool.description}" for tool in tool_list])}

    """
    # Add recursion_limit to prevent GraphRecursionError
    react_response = await react_chain.ainvoke(
        {"messages": [HumanMessage(content=react_input)]},
        config={"recursion_limit": 50},
    )

    # Extract ToolMessage from the response messages
    from langchain_core.messages import ToolMessage

    messages_from_react = react_response.get("messages", [])
    direct_tool_message = None

    # Find ToolMessage in the messages
    for msg in messages_from_react:
        if isinstance(msg, ToolMessage):
            direct_tool_message = msg.content
            break

    # Get the final answer which should be the last message's content
    final_message_content = (
        messages_from_react[-1].content if messages_from_react else ""
    )

    return Command(
        goto=END,
        update={
            "direct_tool_message": direct_tool_message,
            "messages": final_message_content,
        },
    )


if __name__ == "__main__":
    import asyncio

    async def test_tools_advisor_node() -> None:
        """Test tools advisor node (LLM-backed)."""
        state = ReActSubgraphState(
            user_request=UserRequestExtractionSchema(
                task="Find software engineer jobs in Virginia.",
            ),
            user_profile=UserProfileSchema(
                name="John Doe",
                current_employment_status="employed",
                zip_code="20850",
                what_is_the_user_looking_for=[
                    "Find software engineer jobs in Virginia"
                ],
            ),
            why_this_agent_can_help=(
                "I can help the user find software engineer jobs in Virginia."
            ),
        )
        response = await tools_advisor_node(state)
        print(format_command(response))

    asyncio.run(test_tools_advisor_node())

    async def test_react_node() -> None:
        """Test react node (LLM-backed)."""
        state = ReActSubgraphState(
            user_request=UserRequestExtractionSchema(
                task="Find software engineer jobs in Virginia.",
            ),
            user_profile=UserProfileSchema(
                name="John Doe",
                current_employment_status="employed",
                zip_code="20850",
                what_is_the_user_looking_for=[
                    "Find software engineer jobs in Virginia"
                ],
            ),
            why_this_agent_can_help=(
                "I can help the user find software engineer jobs in Virginia."
            ),
            job_search_query=["software engineer jobs Virginia VA"],
            job_search_query_reasoning=(
                "This query targets software engineer jobs in Virginia, "
                "including common synonyms and state variations (VA) to enrich "
                "semantic search."
            ),
        )
        response = await react_node(state)
        print(format_command(response))

    asyncio.run(test_react_node())
