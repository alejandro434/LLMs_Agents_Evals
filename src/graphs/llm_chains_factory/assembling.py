"""This module contains the logic to build LLM chains.

uv run python src/graphs/llm_chains_factory/assembling.py
"""

# %%
from collections.abc import Callable
from pathlib import Path

import yaml
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel

from src.graphs.llm_chains_factory.dynamic_fewshots import (
    create_dynamic_fewshooter,
    render_examples_for_system,
)
from src.utils import get_llm


def build_prompt(
    system_prompt: str,
    k: int = 5,
    group: str | None = None,
    yaml_path: Path | None = None,
) -> ChatPromptTemplate:
    """Build a ChatPromptTemplate that expects a pre-rendered system block.

    The actual rendering of few-shots into the system content is handled
    upstream (e.g., in build_structured_chain), and injected as
    `{system_block}`.
    """
    # Keep a single system message with a placeholder for the rendered block
    return ChatPromptTemplate.from_messages(
        [
            ("system", "{system_block}"),
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
    # We'll compute the system block using the dynamic fewshot selector
    few_shooter = create_dynamic_fewshooter(k=k, group=group, yaml_path=yaml_path)

    def _build_system_block(raw: str | dict) -> dict:
        # Normalize input
        if isinstance(raw, dict):
            user_input = str(raw.get("input", ""))
        else:
            user_input = str(raw)

        selector = getattr(few_shooter, "example_selector", None)
        selected = []
        if selector is not None:
            selected = selector.select_examples({"input": user_input})

        rendered = render_examples_for_system(selected)
        system_block = (
            f"{system_prompt}\n\n"
            "== Final prompt (messages in order) ==\n"
            "System prompt:\n"
            f"{system_prompt}\n\n"
            f"{rendered}\n"
            "Now, continue the following conversation with the user, and fill the "
            "required structure output schema:\n"
            "- input:\n"
            f"    {user_input}\n"
            "- direct_response_to_the_user:\n"
            "(continue)\n"
        )
        return {"system_block": system_block}

    prompt = build_prompt(
        system_prompt=system_prompt,
        k=k,
        group=group,
        yaml_path=yaml_path,
    )

    pipeline: Runnable = (
        RunnableLambda(_build_system_block)
        | prompt
        | llm.with_structured_output(output_schema)
    )
    if postprocess is not None:
        pipeline = pipeline | RunnableLambda(postprocess)
    return pipeline.with_retry(stop_after_attempt=3)


if __name__ == "__main__":

    class TestOutputSchema(BaseModel):
        """Test output schema."""

        query: str
        response: str

    # Load receptionist system prompt from YAML
    with Path("src/graphs/receptionist_subgraph/system_prompt.yml").open(
        encoding="utf-8"
    ) as f:
        SYSTEM_PROMPT = yaml.safe_load(f)["SYSTEM_PROMPT_RECEPTIONIST"]

    # Build chain and print the final formatted system prompt too
    chain = build_structured_chain(
        system_prompt=SYSTEM_PROMPT,
        output_schema=TestOutputSchema,
        k=5,
        temperature=0,
        postprocess=None,
        group="TARGET_LLM_CHAIN_1",
        yaml_path=Path(__file__).parent / "test_fewshots.yml",
    )

    # Render the system block once for inspection using a small helper
    from src.graphs.llm_chains_factory.dynamic_fewshots import (
        create_dynamic_fewshooter as _create,
        render_examples_for_system as _render,
    )

    FEW = _create(
        k=5,
        group="TARGET_LLM_CHAIN_1",
        yaml_path=Path(__file__).parent / "test_fewshots.yml",
    )
    SELECTOR = getattr(FEW, "example_selector", None)
    SELECTED = (
        SELECTOR.select_examples(
            {"input": "Which Virginia programs and employers should I target?"}
        )
        if SELECTOR
        else []
    )
    SYSTEM_BLOCK = (
        f"{SYSTEM_PROMPT}\n\n"
        "== Final prompt (messages in order) ==\n"
        "System prompt:\n"
        f"{SYSTEM_PROMPT}\n\n"
        f"{_render(SELECTED)}\n"
        "Now, continue the following conversation with the user, and fill the required structure output schema:\n"
        "- input:\n"
        "    Which Virginia programs and employers should I target?\n"
        "- direct_response_to_the_user:\n"
        "(continue)\n"
    )
    print(SYSTEM_BLOCK)
