"""Receptionist chain assembly using concierge few-shots.

Mirrors the API of `src/graphs/llm_chains_factory/assembling.py`, sourcing
examples from `fewshots.yml` via the local dynamic few-shots builder.
uv run src/graphs/receptionist_subgraph/receptor_chain_factory/assembling.py
"""

# %%
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import yaml
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel

from src.graphs.receptionist_subgraph.receptor_chain_factory.dynamic_fewshots import (
    create_dynamic_fewshooter,
    render_examples_for_system,
)
from src.utils import get_llm


load_dotenv(override=True)


def build_prompt() -> ChatPromptTemplate:
    """Build a ChatPromptTemplate that expects `{system_block}`."""
    return ChatPromptTemplate.from_messages([("system", "{system_block}")])


def build_structured_chain(
    *,
    system_prompt: str,
    output_schema: type[BaseModel],
    k: int = 5,
    temperature: float = 0,
    postprocess: Callable | None = None,
    yaml_path: Path | None = None,
    embeddings: Embeddings | None = None,
    current_history: list[BaseMessage | dict] | None = None,
    prefer_langsmith: bool = True,
    dataset_name: str | None = None,
) -> Runnable:
    """Create a structured-output chain using receptionist few-shots."""
    bundle_pair = build_structured_chain_with_renderer(
        system_prompt=system_prompt,
        output_schema=output_schema,
        k=k,
        temperature=temperature,
        postprocess=postprocess,
        yaml_path=yaml_path,
        embeddings=embeddings,
        current_history=current_history,
        prefer_langsmith=prefer_langsmith,
        dataset_name=dataset_name,
    )
    return bundle_pair.chain


class StructuredChainBundle(NamedTuple):
    """Bundle with runnable chain and helper to render system block."""

    chain: Runnable
    render_system_block: Callable[[str | dict], str]


def build_structured_chain_with_renderer(
    *,
    system_prompt: str,
    output_schema: type[BaseModel],
    k: int = 5,
    temperature: float = 0,
    postprocess: Callable | None = None,
    yaml_path: Path | None = None,
    embeddings: Embeddings | None = None,
    current_history: list[BaseMessage | dict] | None = None,
    prefer_langsmith: bool = True,
    dataset_name: str | None = None,
) -> StructuredChainBundle:
    """Create the chain and return a system-block renderer helper."""
    # Build few-shots selector first
    few_shooter = create_dynamic_fewshooter(
        k=k,
        yaml_path=yaml_path,
        embeddings=embeddings,
        prefer_langsmith=prefer_langsmith,
        dataset_name=dataset_name,
    )

    def _render_history_lines(history: list[BaseMessage | dict]) -> list[str]:
        lines: list[str] = []
        for turn in history:
            role = None
            turn_content = ""
            if isinstance(turn, HumanMessage | AIMessage):
                role = "human" if isinstance(turn, HumanMessage) else "ai"
                turn_content = str(getattr(turn, "content", ""))
            elif isinstance(turn, dict):
                if "human" in turn:
                    role = "human"
                    turn_content = str(turn.get("human", ""))
                elif "ai" in turn:
                    role = "ai"
                    turn_content = str(turn.get("ai", ""))
            if role:
                lines.append("    - " + role + ":")
                for line in str(turn_content).splitlines() or [""]:
                    lines.append("        " + line)
        return lines

    def _render_schema_lines(model: type[BaseModel]) -> list[str]:
        schema = model.model_json_schema()
        properties: dict = schema.get("properties", {})
        required: set[str] = set(schema.get("required", []) or [])

        def _type_of(prop: dict) -> str:
            t = prop.get("type")
            if isinstance(t, list):
                return "/".join(sorted(str(x) for x in t))
            return str(t or "object")

        lines: list[str] = []
        for name, prop in properties.items():
            typ = _type_of(prop)
            suffix = "(required)" if name in required else "(optional)"
            lines.append(f"- {name}: {typ} {suffix}")
        return lines

    def _build_system_block(raw: str | dict, **kwargs) -> dict:
        user_input = str(raw.get("input", "")) if isinstance(raw, dict) else str(raw)
        runtime_history = None
        if isinstance(raw, dict) and "current_history" in raw:
            runtime_history = raw.get("current_history")
        elif "current_history" in kwargs:
            runtime_history = kwargs.get("current_history")

        selector = getattr(few_shooter, "example_selector", None)
        selected = selector.select_examples({"input": user_input}) if selector else []
        rendered = render_examples_for_system(selected)

        lines: list[str] = []
        lines.append(system_prompt)
        lines.append("")
        lines.append("== FINAL PROMPT (MESSAGES IN ORDER) ==")
        lines.append("System prompt:")
        lines.append(system_prompt)
        lines.append("")
        lines.append("=== SECTION 1: FEW-SHOT EXAMPLES (READ-ONLY) ===")
        lines.append("<EXAMPLES_START>")
        lines.append(rendered)
        lines.append("<EXAMPLES_END>")
        lines.append("")
        lines.append("=== SECTION 2: ACTUAL CURRENT CONVERSATION (ANSWER THIS) ===")
        lines.append("Instructions:")
        lines.append("- Use the conversation below to produce the structured output.")
        lines.append("- If examples conflict with the current conversation,")
        lines.append("  prioritize the current conversation.")
        lines.append("")
        lines.append(
            "Output schema to fill (respond strictly as valid JSON matching this "
        )
        lines.append("schema):")
        for ln in _render_schema_lines(output_schema):
            lines.append(ln)
        lines.append("")
        lines.append("Short definitions:")
        lines.append("- '- input:' is the last user message to answer.")
        lines.append(
            "- '- direct_response_to_the_user:' is your reply to that message."
        )
        lines.append("")
        lines.append("<CONVERSATION_START>")
        history_for_block = runtime_history or current_history
        if history_for_block:
            lines.append("CURRENT CONVERSATION HISTORY:")
            lines.extend(_render_history_lines(history_for_block))
        lines.append("- input:")
        for line in str(user_input).splitlines() or [""]:
            lines.append("    " + line)
        lines.append("- direct_response_to_the_user:")
        lines.append("(you CONTINUE here the CURRENT conversation)")
        system_block = "\n".join(lines)
        return {"system_block": system_block}

    prompt = build_prompt()
    llm = get_llm().bind(temperature=temperature)
    pipeline: Runnable = (
        RunnableLambda(_build_system_block)
        | prompt
        | llm.with_structured_output(output_schema)
    )
    if postprocess is not None:
        pipeline = pipeline | RunnableLambda(postprocess)
    pipeline = pipeline.with_retry(stop_after_attempt=3)

    def _render_system_block(raw: str | dict) -> str:
        return _build_system_block(raw)["system_block"]

    return StructuredChainBundle(
        chain=pipeline, render_system_block=_render_system_block
    )


if __name__ == "__main__":

    class TestOutputSchema(BaseModel):
        """Lightweight schema for local testing."""

        query: str
        response: str

    with Path("src/graphs/receptionist_subgraph/system_prompt.yml").open(
        encoding="utf-8"
    ) as f:
        SYSTEM_PROMPT = yaml.safe_load(f)["SYSTEM_PROMPT_RECEPTIONIST"]

    CURRENT_HISTORY = [
        HumanMessage(content=("Hi! I'm in Arlington, VA exploring cybersecurity.")),
        AIMessage(content="What's your name and current address?"),
        HumanMessage(content="James Patel, 1100 Wilson Blvd, Arlington, VA."),
    ]

    bundle = build_structured_chain_with_renderer(
        system_prompt=SYSTEM_PROMPT,
        output_schema=TestOutputSchema,
        k=5,
        temperature=0,
        yaml_path=Path(__file__).resolve().parents[1] / "fewshots.yml",
        current_history=CURRENT_HISTORY,
    )

    SAMPLE_INPUT = "Which Virginia programs and employers should I target?"
    SYSTEM_BLOCK = bundle.render_system_block({"input": SAMPLE_INPUT})
    print(SYSTEM_BLOCK)
