"""Receptionist subgraph chains.

uv run -m src.graphs.receptionist_subgraph.chains
"""

# %%
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import yaml
from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from src.graphs.llm_chains_factory.assembling import (
    build_structured_chain,
)
from src.graphs.receptionist_subgraph.receptor_chain_factory.assembling import (
    build_structured_chain as build_structured_chain_from_receptor_chain_factory,
)
from src.graphs.receptionist_subgraph.schemas import (
    ReceptionistOutputSchema,
    UserProfileSchema,
    UserRequestExtractionSchema,
)


_BASE_DIR = Path(__file__).parent


def _load_system_prompt(relative_file: str, key: str) -> str:
    """Load a system prompt string from a YAML file in this package."""
    with (_BASE_DIR / relative_file).open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)[key]


def get_receptionist_chain(
    *,
    k: int = 5,
    temperature: float = 0,
    current_history: Sequence[BaseMessage | dict[str, Any]] | None = None,
    prefer_langsmith: bool = True,
    dataset_name: str | None = None,
):
    """Construct the receptionist structured-output chain.

    Args:
        k: Number of few-shot examples to include
        temperature: LLM temperature for response generation
        current_history: Optional conversation history to include in the prompt
        prefer_langsmith: If True, prefer loading few-shots from LangSmith dataset
        dataset_name: Optional LangSmith dataset name (defaults to internal name)

    Returns:
        A structured chain configured for receptionist tasks

    Raises:
        FileNotFoundError: If system prompt or fewshots file not found
        yaml.YAMLError: If YAML files are malformed
    """
    try:
        system_prompt = _load_system_prompt(
            "prompts/system_prompt.yml", "SYSTEM_PROMPT_RECEPTIONIST"
        )
    except (FileNotFoundError, yaml.YAMLError) as e:
        raise RuntimeError(f"Failed to load receptionist system prompt: {e}") from e

    return build_structured_chain_from_receptor_chain_factory(
        system_prompt=system_prompt,
        output_schema=ReceptionistOutputSchema,
        k=k,
        temperature=temperature,
        postprocess=None,
        yaml_path=_BASE_DIR / "prompts" / "fewshots.yml",
        current_history=list(current_history) if current_history else None,
        prefer_langsmith=prefer_langsmith,
        dataset_name=dataset_name,
    )


def get_profiling_chain(
    *,
    k: int = 5,
    temperature: float = 0,
):
    """Construct the profiling structured-output chain (field mapping).

    This chain maps ReceptionistOutputSchema fields to UserProfileSchema fields
    in a strict 1:1 manner without inference or hallucination.

    Args:
        k: Number of few-shot examples to include
        temperature: LLM temperature (should be 0 for deterministic mapping)

    Returns:
        A structured chain configured for profile field mapping

    Raises:
        FileNotFoundError: If system prompt or fewshots file not found
        yaml.YAMLError: If YAML files are malformed
    """
    try:
        system_prompt = _load_system_prompt(
            "prompts/profiling_system_prompt.yml",
            "SYSTEM_PROMPT_RECEPTIONIST_PROFILING",
        )
    except (FileNotFoundError, yaml.YAMLError) as e:
        raise RuntimeError(f"Failed to load profiling system prompt: {e}") from e

    return build_structured_chain(
        system_prompt=system_prompt,
        output_schema=UserProfileSchema,
        k=k,
        temperature=temperature,
        postprocess=None,
        group="Profiling_examples",
        yaml_path=_BASE_DIR / "prompts" / "profiling_fewshots.yml",
    )


def get_user_request_extraction_chain(
    *,
    k: int = 5,
    temperature: float = 0,
):
    """Construct the user request extraction chain."""
    try:
        system_prompt = _load_system_prompt(
            "prompts/user_request_extraction_system_prompt.yml",
            "SYSTEM_PROMPT_RECEPTIONIST_USER_REQUEST_EXTRACTION",
        )
    except (FileNotFoundError, yaml.YAMLError) as e:
        raise RuntimeError(f"Failed to load profiling system prompt: {e}") from e

    return build_structured_chain(
        system_prompt=system_prompt,
        output_schema=UserRequestExtractionSchema,
        k=k,
        temperature=temperature,
        postprocess=None,
        group="User_request_extraction_examples",
        yaml_path=_BASE_DIR / "prompts" / "user_request_extraction_fewshots.yml",
    )


class AgentSelectionSchema(BaseModel):
    """Agent selection schema."""

    agent_name: Literal["Jobs", "Educator", "Events", "CareerCoach", "Entrepreneur"]
    rationale_of_the_handoff: str


def get_agent_selection_chain(
    *,
    k: int = 5,
    temperature: float = 0,
):
    """Construct the agent selection chain."""
    try:
        system_prompt = _load_system_prompt(
            "prompts/agent_selection_system_prompt.yml",
            "SYSTEM_PROMPT_RECEPTIONIST_AGENT_SELECTION",
        )
    except (FileNotFoundError, yaml.YAMLError) as e:
        raise RuntimeError(f"Failed to load profiling system prompt: {e}") from e

    return build_structured_chain(
        system_prompt=system_prompt,
        output_schema=AgentSelectionSchema,
        k=k,
        temperature=temperature,
        postprocess=None,
        group="Agent_selection_examples",
        yaml_path=_BASE_DIR / "prompts" / "agent_selection_fewshots.yml",
    )


# Module-level chains (default configs)
receptionist_chain = get_receptionist_chain()
profiling_chain = get_profiling_chain()
user_request_extraction_chain = get_user_request_extraction_chain()
agent_selection_chain = get_agent_selection_chain()

if __name__ == "__main__":
    import asyncio

    async def receptionist_chain_test() -> None:
        """Quick demo for receptionist_chain."""
        test_input = "Hi, I'm looking for an entry-level retail job."
        # Optional history passed at invocation time
        current_history = [
            {"human": "Hi, I'm in Arlington, VA exploring cybersecurity."},
            {"ai": "What's your name and current address?"},
            {"human": "James Patel, 1100 Wilson Blvd, Arlington, VA."},
        ]
        runtime_context_injection = (
            "User previously indicated preference for full-time roles and retail."
        )

        result = await receptionist_chain.ainvoke(
            test_input,
            current_history=current_history,
            runtime_context_injection=runtime_context_injection,
        )
        print(result.model_dump_json(indent=2))

    asyncio.run(receptionist_chain_test())

    async def profiling_chain_test() -> None:
        """Quick demo for profiling_chain."""
        test_input = ReceptionistOutputSchema(
            direct_response_to_the_user="I can share local job resources.",
            name="John Doe",
            current_employment_status="unemployed",
            zip_code="21201",
            what_is_the_user_looking_for=("Entry-level IT support in Baltimore"),
        ).model_dump_json()

        result = await profiling_chain.ainvoke(test_input)
        print(result.model_dump_json(indent=2))

    asyncio.run(profiling_chain_test())

    async def user_request_extraction_chain_test() -> None:
        """Test suite for user_request_extraction_chain."""
        print("\n" + "=" * 60)
        print("Testing user_request_extraction_chain")
        print("=" * 60)

        test_cases = [
            "I need to find job fairs happening in my area next month",
            "Can you help me look for remote software engineering positions?",
            "I want to know what training programs are available for healthcare workers in Maryland",
            "Help me find entry-level retail jobs near Arlington, VA",
            "I'm looking for information about unemployment benefits in Virginia",
            "Can you search for warehouse jobs that pay over $20 per hour?",
            "I need help finding vocational schools that offer HVAC training",
            "Show me government job openings in the DC area",
        ]

        for i, test_input in enumerate(test_cases, 1):
            print(f"\nTest Case {i}:")
            print(f"Input: {test_input}")
            result = await user_request_extraction_chain.ainvoke(test_input)
            print(f"Extracted Task: {result.task}")
            assert result.task, f"Failed to extract task for test case {i}"

        print("\n✅ All user_request_extraction_chain tests passed!")

    asyncio.run(user_request_extraction_chain_test())

    async def agent_selection_chain_test() -> None:
        """Test suite for agent_selection_chain."""
        print("\n" + "=" * 60)
        print("Testing agent_selection_chain")
        print("=" * 60)

        test_tasks = [
            "Search for and provide information about job fairs happening in the user's area within the next month",
            "Search for remote software engineering job positions that match the user's qualifications",
            "Find and list available training programs for healthcare workers in Maryland",
            "Search for entry-level retail job opportunities in or near Arlington, VA",
            "Provide comprehensive information about unemployment benefits eligibility, application process, and requirements in Virginia",
            "Find warehouse job positions with hourly pay rates exceeding $20 per hour",
            "Search for and provide information about vocational schools offering HVAC training programs",
            "Search for current government job openings in the Washington DC metropolitan area",
        ]

        for i, test_task in enumerate(test_tasks, 1):
            print(f"\nTest Case {i}:")
            print(f"Task: {test_task[:80]}...")
            result = await agent_selection_chain.ainvoke(test_task)
            print(f"Selected Agent: {result.agent_name}")
            print(f"Rationale: {result.rationale_of_the_handoff[:100]}...")

            # Validate that agent_name is one of the configured agents
            allowed_agents = {
                "Jobs",
                "Educator",
                "Events",
                "CareerCoach",
                "Entrepreneur",
            }
            assert result.agent_name in allowed_agents, (
                f"Unexpected agent '{result.agent_name}'"
            )
            assert result.rationale_of_the_handoff, (
                f"Missing rationale for test case {i}"
            )

        print("\n✅ All agent_selection_chain tests passed!")

    asyncio.run(agent_selection_chain_test())

    async def integration_test() -> None:
        """Integration test: Extract task from user input, then select agent."""
        print("\n" + "=" * 60)
        print("Integration Test: User Input → Task Extraction → Agent Selection")
        print("=" * 60)

        user_inputs = [
            "I need to find job fairs in Baltimore next week",
            "Help me search for nursing jobs at local hospitals",
            "Can you look up CDL training programs near me?",
        ]

        for user_input in user_inputs:
            print(f"\n🔹 User Input: '{user_input}'")

            # Step 1: Extract task
            extraction_result = await user_request_extraction_chain.ainvoke(user_input)
            extracted_task = extraction_result.task
            print(f"   → Extracted Task: '{extracted_task}'")

            # Step 2: Select agent
            selection_result = await agent_selection_chain.ainvoke(extracted_task)
            print(f"   → Selected Agent: {selection_result.agent_name}")
            print(
                f"   → Rationale: {selection_result.rationale_of_the_handoff[:80]}..."
            )

        print("\n✅ Integration test completed successfully!")

    asyncio.run(integration_test())
