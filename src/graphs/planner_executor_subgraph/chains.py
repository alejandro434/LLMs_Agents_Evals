"""Planner subgraph chains.

uv run -m src.graphs.planner_executor_subgraph.chains
"""

# %%
from pathlib import Path

import yaml

from src.graphs.llm_chains_factory.assembling import build_structured_chain
from src.graphs.planner_executor_subgraph.schemas import Plan


with Path("src/graphs/planner_executor_subgraph/system_prompt.yml").open(
    encoding="utf-8"
) as f:
    SYSTEM_PROMPT = yaml.safe_load(f)["SYSTEM_PROMPT_PLANNER"]
chain = build_structured_chain(
    system_prompt=SYSTEM_PROMPT,
    output_schema=Plan,
    k=5,
    temperature=0,
    postprocess=None,
    group="FEW_SHOTS_PLANNER",
    yaml_path=Path(__file__).parent / "fewshots.yml",
)

if __name__ == "__main__":
    import asyncio

    async def main():
        """Main function."""
        response = await chain.ainvoke(
            "Descripción/resumen de los proyectos de la región de antofagasta"
        )
        print(f"Is complete: {response.is_complete}")
        print(f"Goal: {response.goal}")
        for step in response.steps:
            print(f"Instruction: {step.instruction}")
            print(f"Suggested tool: {step.suggested_tool}")
            print(f"Reasoning: {step.reasoning}")
            print(f"Result: {step.result}")
            print(f"Is complete: {step.is_complete}")
            print("-" * 100)

    asyncio.run(main())
