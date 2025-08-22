"""This module contains the logic to build LLM chains.

uv run python src/graphs/llm_chains_factory/assembling.py
"""

# %%
from collections.abc import Callable
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel

from src.graphs.llm_chains_factory.dynamic_fewshots import create_dynamic_fewshooter
from src.utils import get_llm


# %%
def build_prompt(
    system_prompt: str,
    k: int = 5,
    group: str | None = None,
    yaml_path: Path | None = None,
) -> ChatPromptTemplate:
    """Build a ChatPromptTemplate with dynamic few-shot selection.

    For AWS Bedrock compatibility, all system content is consolidated
    into a single system message at the beginning.
    """
    few_shooter = create_dynamic_fewshooter(k=k, group=group, yaml_path=yaml_path)

    # Consolidate all system content into one message for Bedrock compatibility
    consolidated_system = (
        f"{system_prompt}\n\n"
        "## A continuación, ejemplos de requerimientos y respuestas parecidas:"
    )

    return ChatPromptTemplate.from_messages(
        [
            ("system", consolidated_system),
            few_shooter,  # This generates human/AI message pairs
            ("human", "{input}"),
        ]
    )


def build_structured_chain(
    *,
    system_prompt: str,
    output_schema: type[BaseModel],
    k: int = 5,
    temperature: float = 0,
    postprocess: Callable | None = None,
    group: str | None = None,
    yaml_path: Path | None = None,
) -> Runnable:
    """Create a structured-output chain for an arbitrary system prompt and schema."""
    llm = get_llm().bind(temperature=temperature)
    prompt = build_prompt(
        system_prompt=system_prompt,
        k=k,
        group=group,
        yaml_path=yaml_path,
    )
    pipeline: Runnable = prompt | llm.with_structured_output(output_schema)
    if postprocess is not None:
        pipeline = pipeline | RunnableLambda(postprocess)
    return pipeline.with_retry(stop_after_attempt=3)


if __name__ == "__main__":

    class TestOutputSchema(BaseModel):
        """Test output schema."""

        query: str
        response: str

    chain = build_structured_chain(
        system_prompt="You are a helpful assistant that can answer questions about the graph.",
        output_schema=TestOutputSchema,
        k=5,
        temperature=0,
        postprocess=None,
        group="TARGET_LLM_CHAIN_1",
        yaml_path=Path(__file__).parent / "test_fewshots.yml",
    )
    print(chain.invoke("numero de proyectos?").model_dump_json(indent=2))
