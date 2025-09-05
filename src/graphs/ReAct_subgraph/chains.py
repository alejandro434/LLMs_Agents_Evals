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
    SuggestedRelevantToolsOutputSchema,
)
from src.graphs.ReAct_subgraph.tools.qdrant_hybrid_search import (
    qdrant_hybrid_search_tool,
)


_BASE_DIR = Path(__file__).parent

tool_list = [qdrant_hybrid_search_tool]


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


from langchain_core.messages import SystemMessage


# Create a more explicit system message for the agent
system_prompt = SystemMessage(
    content="""You are a helpful job search assistant.
Given a user's task and profile, you will use the available tools to find relevant job listings.

IMPORTANT: After using the tools to search for jobs, you MUST provide a final answer that:
1. Summarizes the job search results
2. Highlights the most relevant positions found
3. Provides helpful information to the user

Do not end your response with just tool results. Always provide a comprehensive final answer after using the tools."""
)

react_chain = create_react_agent(
    model=model,
    tools=tool_list,
    prompt=system_prompt,
    # Note: response_format with create_react_agent doesn't work as expected
    # The agent returns tool messages instead of structured output
    # response_format=ReActResponse,
)


if __name__ == "__main__":
    # Simple demonstration / test

    tools_advisor_chain = get_tools_advisor_chain()
    TEST_INPUT_TEXT = "Find software engineer jobs in Virginia."
    advisor_output = tools_advisor_chain.invoke(
        {
            "input": TEST_INPUT_TEXT,
            "runtime_context_injection": (
                "AVAILABLE tools names and tools descriptions are:\n"
                + "\n".join(
                    [
                        f"Name: {tool.name}, Description: {tool.description}"
                        for tool in tool_list
                    ]
                )
            ),
        }
    )
    print("Tools advisor response (query):", advisor_output.job_search_query)
    print("Tools advisor reasoning:", advisor_output.job_search_query_reasoning)

    REACT_INPUT = (
        "The task you have to execute is: "
        f"{advisor_output.job_search_query}\n"
        "The reasoning for why this query is suitable for a job search is: "
        f"{advisor_output.job_search_query_reasoning}\n"
        "AVAILABLE tools names and tools descriptions for jobs search are:\n"
        + "\n".join(
            [
                f"Name: {tool.name}, Description: {tool.description}"
                for tool in tool_list
            ]
        )
    )
    react_response = react_chain.invoke(
        {"messages": [HumanMessage(content=REACT_INPUT)]}
    )
    print(f"{react_response['messages'][-1].content}")
