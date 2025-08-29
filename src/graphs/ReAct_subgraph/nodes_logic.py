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
    SuggestedRelevantToolsOutputSchema,
)
from src.graphs.ReAct_subgraph.tools.websearch import websearch_tool
from src.utils import format_command


tools_advisor_chain = get_tools_advisor_chain()
tool_list = [websearch_tool]


async def tools_advisor_node(
    state: ReActSubgraphState,
) -> Command[Literal["react_node"]]:
    """ReAct node."""
    task = state.get("task")
    user_profile = state.get("user_profile")
    why_this_agent_can_help = state.get("why_this_agent_can_help")
    context_injection = (
        f"In order to suggest relevant or useful tools to execute the above mentioned task, you have to consider the following:\n"
        f"The user's profile is: {user_profile}\n"
        f"The reason why you can help the user is: {why_this_agent_can_help}\n"
        f"AVAILABLE tools names and tools descriptions are:\n"
        f"{'\n'.join([f'Name: {tool.name}, Description: {tool.description}' for tool in tool_list])}"
    )

    tools_advisor_response = await tools_advisor_chain.ainvoke(
        {"messages": [HumanMessage(content=task)]},
        runtime_context_injection=context_injection,
    )
    return Command(
        goto=END,
        update={
            "suggested_tools": tools_advisor_response.suggested_tools,
            "tools_advisor_reasoning": tools_advisor_response.tools_advisor_reasoning,
        },
    )


async def react_node(state: ReActSubgraphState) -> Command[Literal[END]]:
    """ReAct node."""
    task = state.get("task")
    user_profile = state.get("user_profile")
    why_this_agent_can_help = state.get("why_this_agent_can_help")
    suggested_tools = state.get("suggested_tools")
    react_input = f"""
    The task you have to execute is: {task}
    The user's profile is: {user_profile}
    The reason why you can help the user is: {why_this_agent_can_help}
    The suggested tools are: {suggested_tools.suggested_tools}
    The reasoning for why these tools may be relevant is: {suggested_tools.tools_advisor_reasoning}
    """
    react_response = react_chain.invoke({"messages": [react_input]})
    return Command(
        goto=END,
        update={"final_answer": react_response["structured_response"].final_answer},
    )


if __name__ == "__main__":
    import asyncio

    async def test_tools_advisor_node() -> None:
        """Test receptor node."""
        state = ReActSubgraphState(
            task="Find in the web job fairs events in the next 30 days.",
            user_profile="I am a job seeker looking for a job in the tech industry.",
            why_this_agent_can_help="I can help the user find job fairs events in the next 30 days.",
        )
        response = await tools_advisor_node(state)
        print(format_command(response))

    asyncio.run(test_tools_advisor_node())

    async def test_react_node() -> None:
        """Test receptor node."""
        state = ReActSubgraphState(
            task="Find in the web job fairs events in the next 30 days.",
            user_profile="I am a job seeker looking for a job in the tech industry.",
            why_this_agent_can_help="I can help the user find job fairs events in the next 30 days.",
            suggested_tools=SuggestedRelevantToolsOutputSchema(
                suggested_tools=["websearch"],
                tools_advisor_reasoning="I can help the user find job fairs events in the next 30 days.",
            ),
        )
        response = await react_node(state)
        print(format_command(response))

    asyncio.run(test_react_node())
