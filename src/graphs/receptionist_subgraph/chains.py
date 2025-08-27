"""Receptionist subgraph chains.

uv run -m src.graphs.receptionist_subgraph.chains
"""

# %%
from pathlib import Path

import yaml

from src.graphs.llm_chains_factory.assembling import build_structured_chain
from src.graphs.receptionist_subgraph.schemas import (
    ReceptionistOutputSchema,
    UserProfileSchema,
)


with Path("src/graphs/receptionist_subgraph/system_prompt.yml").open(
    encoding="utf-8"
) as f:
    SYSTEM_PROMPT = yaml.safe_load(f)["SYSTEM_PROMPT_RECEPTIONIST"]
receptionist_chain = build_structured_chain(
    system_prompt=SYSTEM_PROMPT,
    output_schema=ReceptionistOutputSchema,
    k=5,
    temperature=0,
    postprocess=None,
    group="Receptionist_user_conversations",
    yaml_path=Path(__file__).parent / "fewshots.yml",
)

with Path("src/graphs/receptionist_subgraph/profiling_system_prompt.yml").open(
    encoding="utf-8"
) as f:
    SYSTEM_PROMPT_PROFILING = yaml.safe_load(f)["SYSTEM_PROMPT_RECEPTIONIST_PROFILING"]

profiling_chain = build_structured_chain(
    system_prompt=SYSTEM_PROMPT_PROFILING,
    output_schema=UserProfileSchema,
    k=5,
    temperature=0,
    postprocess=None,
    group="Profiling_examples",
    yaml_path=Path(__file__).parent / "profiling_fewshots.yml",
)

if __name__ == "__main__":
    import asyncio

    async def receptionist_chain_test() -> None:
        """Quick demo for receptionist_chain."""
        test_input = "Hi, I'm looking for an entry-level retail job."

        result = await receptionist_chain.ainvoke(test_input)
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
            handoff_needed=False,
        ).model_dump_json()

        result = await profiling_chain.ainvoke(test_input)
        print(result.model_dump_json(indent=2))

    asyncio.run(profiling_chain_test())
