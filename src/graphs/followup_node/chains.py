"""Followup subgraph chains.

uv run -m src.graphs.followup_node.chains
"""

# %%
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml
from langchain_core.messages import BaseMessage

from src.graphs.followup_node.schemas import FollowupOutputSchema
from src.graphs.llm_chains_factory.assembling import (
    build_structured_chain,
)


_BASE_DIR = Path(__file__).parent


def _load_system_prompt(relative_file: str, key: str) -> str:
    """Load a system prompt string from a YAML file in this package."""
    with (_BASE_DIR / relative_file).open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)[key]


def get_followup_chain(
    *,
    k: int = 5,
    temperature: float = 0,
    current_history: Sequence[BaseMessage | dict[str, Any]] | None = None,
):
    """Construct the followup structured-output chain.

    Args:
        k: Number of few-shot examples to include
        temperature: LLM temperature for response generation
        current_history: Optional conversation history to include in the prompt

    Returns:
        A structured chain configured for followup tasks

    Raises:
        FileNotFoundError: If system prompt or fewshots file not found
        yaml.YAMLError: If YAML files are malformed
    """
    try:
        system_prompt = _load_system_prompt(
            "prompts/system_prompt.yml", "SYSTEM_PROMPT_FOLLOWUP"
        )
    except (FileNotFoundError, yaml.YAMLError) as e:
        raise RuntimeError(f"Failed to load followup system prompt: {e}") from e

    return build_structured_chain(
        system_prompt=system_prompt,
        output_schema=FollowupOutputSchema,
        k=k,
        temperature=temperature,
        postprocess=None,
        yaml_path=_BASE_DIR / "prompts" / "fewshots.yml",
        current_history=list(current_history) if current_history else None,
    )


followup_chain = get_followup_chain()

if __name__ == "__main__":
    import asyncio

    async def main():
        """Main function."""
        response = await followup_chain.ainvoke("Hi, I'm looking for a job.")
        print(response.model_dump_json(indent=2))

    asyncio.run(main())
