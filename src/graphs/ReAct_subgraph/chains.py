"""ReAct subgraph chains.

uv run -m src.graphs.ReAct_subgraph.chains
"""

# %%
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml
from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.prebuilt import create_react_agent

from src.graphs.llm_chains_factory.assembling import (
    build_structured_chain,
)
from src.graphs.ReAct_subgraph.schemas import (
    ReActResponse,
    SuggestedRelevantToolsOutputSchema,
)
from src.graphs.ReAct_subgraph.tools.websearch import websearch_tool


_BASE_DIR = Path(__file__).parent

tool_list = [websearch_tool]


def _load_system_prompt(relative_file: str, key: str) -> str:
    """Load a system prompt string from a YAML file in this package."""
    with (_BASE_DIR / relative_file).open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)[key]


def get_tools_advisor_chain(
    *,
    k: int = 5,
    temperature: float = 0,
    current_history: Sequence[BaseMessage | dict[str, Any]] | None = None,
):
    """Construct the tools advisor structured-output chain for ReAct subgraph.

    Args:
        k: Number of few-shot examples to include.
        temperature: LLM temperature for generation.
        current_history: Optional conversation history to include in the prompt.

    Returns:
        A structured chain that suggests relevant tools for a given task.
    """
    system_prompt = _load_system_prompt(
        "prompts/system_prompt.yml", "Tools_advisor_system_prompt"
    )
    return build_structured_chain(
        system_prompt=system_prompt,
        output_schema=SuggestedRelevantToolsOutputSchema,
        k=k,
        temperature=temperature,
        postprocess=None,
        group="suggest_relevant_tools_examples",
        yaml_path=_BASE_DIR / "prompts" / "fewshots.yml",
        current_history=list(current_history) if current_history else None,
    )


model = init_chat_model("gpt-4.1-mini", temperature=0)


react_chain = create_react_agent(
    model=model,
    tools=tool_list,
    prompt="""
    Given a user's task, user's profile, why you can help the user, and suggested tools,
    you have to execute the task using the suggested tools.
    When you are done, you have to return a final answer to the user.
    """,
    response_format=ReActResponse,
)


if __name__ == "__main__":
    # Simple demonstration / test

    tools_advisor_chain = get_tools_advisor_chain()
    TEST_INPUT = HumanMessage(
        content="Find in the web job fairs events in the next 30 days."
    )
    suggested_tools = tools_advisor_chain.invoke(
        {"messages": [TEST_INPUT]},
        runtime_context_injection=f"""
        AVAILABLE tools names and tools descriptions are:
        {"\n".join([f"Name: {tool.name}, Description: {tool.description}" for tool in tool_list])}
        """,
    )
    print("Tools advisor response:", suggested_tools)

    REACT_INPUT = f"""
    The task you have to execute is: {TEST_INPUT.content}
    The suggested tools are: {suggested_tools.suggested_tools}
    The reasoning for why these tools are relevant is: {suggested_tools.reasoning}
    """
    react_response = react_chain.invoke({"messages": [REACT_INPUT]})
    print("ReAct response:", react_response["structured_response"].final_answer)
