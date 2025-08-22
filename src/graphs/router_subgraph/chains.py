"""Router subgraph chains.

uv run -m src.graphs.router_subgraph.chains
"""

# %%
from pathlib import Path

import yaml

from src.graphs.llm_chains_factory.assembling import build_structured_chain
from src.graphs.router_subgraph.schemas import RouterOutputSchema


with Path("src/graphs/router_subgraph/handoff-subgraphs.yml").open(
    encoding="utf-8"
) as f:
    subgraphs = yaml.safe_load(f)["handoff_subgraphs"]
    SUBGRAPH_DESCRIPTION = "\n".join(
        f"- {subgraph['name']}: {subgraph['description']}" for subgraph in subgraphs
    )


with Path("src/graphs/router_subgraph/system_prompt.yml").open(encoding="utf-8") as f:
    SYSTEM_PROMPT = yaml.safe_load(f)["prompt"].format(
        subgraphs_description=SUBGRAPH_DESCRIPTION
    )

chain = build_structured_chain(
    system_prompt=SYSTEM_PROMPT,
    output_schema=RouterOutputSchema,
    k=5,
    temperature=0,
    postprocess=None,
    group="Routing_examples",
    yaml_path=Path(__file__).parent / "fewshots.yml",
)

if __name__ == "__main__":
    import asyncio

    async def main():
        """Main function."""
        response = await chain.ainvoke("what is the meaning of life?")
        print(response.correctness)
        print(response.model_dump_json(indent=2))

    asyncio.run(main())
