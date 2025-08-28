"""Smoke test for receptionist dynamic few-shots system prompt assembly.

Exercises Conversation_history parsing and example injection into the prompt.

    uv run python src/graphs/receptionist_subgraph/test_dynamic_fewshots_system_prompt.py
"""

# %%
from __future__ import annotations

from pathlib import Path

import yaml

from src.graphs.llm_chains_factory.assembling import (
    build_structured_chain_with_renderer,
)
from src.graphs.receptionist_subgraph.schemas import (
    ReceptionistOutputSchema,
    UserProfileSchema,
)


fewshots_path = Path(__file__).parent / "fewshots.yml"
system_prompt_path = Path(__file__).parent / "system_prompt.yml"
profiling_fewshots_path = Path(__file__).parent / "profiling_fewshots.yml"
profiling_system_prompt_path = Path(__file__).parent / "profiling_system_prompt.yml"

with system_prompt_path.open("r", encoding="utf-8") as f:
    system_prompt = yaml.safe_load(f)["SYSTEM_PROMPT_RECEPTIONIST"]

with profiling_system_prompt_path.open("r", encoding="utf-8") as f:
    profiling_system_prompt = yaml.safe_load(f)["SYSTEM_PROMPT_RECEPTIONIST_PROFILING"]


if __name__ == "__main__":
    # Build chain bundle and render the final system block
    # current_history = [
    #     HumanMessage(
    #         content=("Hi! I'm in Arlington, VA exploring cybersecurity and analytics.")
    #     ),
    #     AIMessage(content="What's your name and current address?"),
    #     HumanMessage(content="James Patel, 1100 Wilson Blvd, Arlington, VA."),
    # ]

    bundle = build_structured_chain_with_renderer(
        system_prompt=system_prompt,
        output_schema=ReceptionistOutputSchema,
        k=5,
        yaml_path=fewshots_path,
        # current_history=current_history,
    )

    DEMO_INPUT = "Which Virginia programs and employers should I target?"
    system_block = bundle.render_system_block({"input": DEMO_INPUT})
    print(system_block)

    print("\n" + "=" * 80)
    print("Profiling chain system block")
    print("=" * 80 + "\n")

    profiling_bundle = build_structured_chain_with_renderer(
        system_prompt=profiling_system_prompt,
        output_schema=UserProfileSchema,
        k=5,
        yaml_path=profiling_fewshots_path,
        # current_history=current_history,
    )

    DEMO_INPUT_PROFILE = "Map receptionist output to user profile"
    profiling_system_block = profiling_bundle.render_system_block(
        {"input": DEMO_INPUT_PROFILE}
    )
    print(profiling_system_block)
