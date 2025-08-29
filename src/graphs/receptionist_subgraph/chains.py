"""Receptionist subgraph chains.

uv run -m src.graphs.receptionist_subgraph.chains
"""

# %%
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml
from langchain_core.messages import BaseMessage

from src.graphs.llm_chains_factory.assembling import (
    build_structured_chain,
)
from src.graphs.receptionist_subgraph.schemas import (
    ReceptionistOutputSchema,
    UserProfileSchema,
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
):
    """Construct the receptionist structured-output chain.

    Args:
        k: Number of few-shot examples to include
        temperature: LLM temperature for response generation
        current_history: Optional conversation history to include in the prompt

    Returns:
        A structured chain configured for receptionist tasks

    Raises:
        FileNotFoundError: If system prompt or fewshots file not found
        yaml.YAMLError: If YAML files are malformed
    """
    try:
        system_prompt = _load_system_prompt(
            "system_prompt.yml", "SYSTEM_PROMPT_RECEPTIONIST"
        )
    except (FileNotFoundError, yaml.YAMLError) as e:
        raise RuntimeError(f"Failed to load receptionist system prompt: {e}") from e

    return build_structured_chain(
        system_prompt=system_prompt,
        output_schema=ReceptionistOutputSchema,
        k=k,
        temperature=temperature,
        postprocess=None,
        group="Receptionist_user_conversations",
        yaml_path=_BASE_DIR / "fewshots.yml",
        current_history=list(current_history) if current_history else None,
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
            "profiling_system_prompt.yml", "SYSTEM_PROMPT_RECEPTIONIST_PROFILING"
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
        yaml_path=_BASE_DIR / "profiling_fewshots.yml",
    )


# Module-level chains (default configs)
receptionist_chain = get_receptionist_chain()
profiling_chain = get_profiling_chain()

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
            user_name="John Doe",
            user_current_address="123 Main St, Baltimore, MD",
            user_employment_status="unemployed",
            user_last_job="Warehouse associate",
            user_last_job_location="Baltimore, MD",
            user_last_job_company="Acme Logistics",
            user_job_preferences="Entry-level IT support in Baltimore",
        ).model_dump_json()

        result = await profiling_chain.ainvoke(test_input)
        print(result.model_dump_json(indent=2))

    asyncio.run(profiling_chain_test())
