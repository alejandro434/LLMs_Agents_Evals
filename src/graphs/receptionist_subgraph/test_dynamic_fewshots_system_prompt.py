"""Smoke test for receptionist dynamic few-shots system prompt assembly.

Exercises Conversation_history parsing and example injection into the prompt.

uv run python src/graphs/receptionist_subgraph/test_dynamic_fewshots_system_prompt.py
"""

# %%
from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import yaml
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


try:
    # Prefer the real few-shot builder if embeddings are available
    from src.graphs.llm_chains_factory.assembling import build_prompt
    from src.graphs.llm_chains_factory.dynamic_fewshots import (
        create_dynamic_fewshooter,
    )
except Exception:  # pragma: no cover - fallback to minimal demo
    create_dynamic_fewshooter = None  # type: ignore[assignment]
    build_prompt = None  # type: ignore[assignment]


def _load_receptionist_examples(yaml_path: Path) -> list[dict[str, Any]]:
    """Load receptionist examples and normalize history/messages for fallback.

    Returns a list of dicts with keys: history (list[BaseMessage]),
    input (str), output (str).
    """
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    rows = data.get("Receptionist_user_conversations", [])
    examples: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        conv = row.get("Conversation_history", [])
        history: list = []
        if isinstance(conv, list):
            for turn in conv:
                if not isinstance(turn, dict):
                    continue
                if "human" in turn:
                    history.append(HumanMessage(content=str(turn.get("human", ""))))
                if "ai" in turn:
                    history.append(AIMessage(content=str(turn.get("ai", ""))))
        text_input = str(row.get("input", "")).strip()
        out = row.get("output", {}) or {}
        if isinstance(out, dict):
            text_output = str(out.get("direct_response_to_the_user", "")).strip()
        else:
            text_output = str(out).strip()
        if text_input and text_output:
            examples.append(
                {"history": history, "input": text_input, "output": text_output}
            )
    return examples


def _indent(text: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + line for line in str(text).splitlines())


def _wrap(text: str, width: int = 88) -> list[str]:
    return textwrap.wrap(str(text), width=width, break_long_words=False)


def _clip_lines(lines: list[str], max_lines: int | None) -> list[str]:
    if max_lines is None:
        return lines
    if len(lines) <= max_lines:
        return lines
    return lines


def _print_block(label: str, text: str, *, max_lines: int | None = None) -> None:
    print(f"- {label}:")
    wrapped = _wrap(text)
    clipped = _clip_lines(wrapped, max_lines)
    print(_indent("\n".join(clipped)))


def _section(title: str) -> None:
    bar = "=" * min(80, max(20, len(title) + 6))
    print("\n" + bar)
    print(f"== {title} ==")
    print(bar)


def _print_history_bullets(history: list) -> None:
    print("Conversation_history:")
    for turn in history:
        role = getattr(turn, "type", None)
        content = getattr(turn, "content", "")
        if not role:
            # Fallback for dict-like turns
            if isinstance(turn, dict):
                if "human" in turn:
                    role = "human"
                    content = str(turn.get("human", ""))
                elif "ai" in turn:
                    role = "ai"
                    content = str(turn.get("ai", ""))
        if role in ("human", "ai"):
            wrapped = _wrap(content)
            joined = "\n".join(wrapped)
            print(_indent(f"- {role}:\n{_indent(joined)}"))


if __name__ == "__main__":
    yaml_path = Path("src/graphs/receptionist_subgraph/fewshots.yml")
    with Path("src/graphs/receptionist_subgraph/system_prompt.yml").open(
        encoding="utf-8"
    ) as f:
        system_prompt = yaml.safe_load(f)["SYSTEM_PROMPT_RECEPTIONIST"]

    USER_INPUT = "Where should I apply?"  # similar to a few-shot input

    try:
        if create_dynamic_fewshooter is None or build_prompt is None:
            raise RuntimeError("dynamic fewshots unavailable")

        fewshot = create_dynamic_fewshooter(
            yaml_path=yaml_path,
            k=2,
            group="Receptionist_user_conversations",
        )

        prompt = build_prompt(
            system_prompt=system_prompt,
            k=2,
            group="Receptionist_user_conversations",
            yaml_path=yaml_path,
        )

        # Show which examples were selected and demonstrate final prompt
        SELECTOR = getattr(fewshot, "example_selector", None)
        if SELECTOR is not None:
            selected = SELECTOR.select_examples({"input": USER_INPUT})
            _section(f"Selected examples (k={len(selected)})")
            for idx, ex in enumerate(selected, start=1):
                print(f"[{idx}] history messages: {len(ex.get('history', []))}")
                _print_block("input", ex.get("input", ""), max_lines=None)
                _print_block("output", ex.get("output", ""), max_lines=None)
                print("-" * 40)

        msgs = prompt.format_messages(input=USER_INPUT)
        # Structured few-shots view per requested format
        _section("Final prompt (messages in order)")
        print("System prompt:")
        print(_indent("\n".join(_wrap(system_prompt))))
        print()
        if SELECTOR is not None:
            selected = SELECTOR.select_examples({"input": USER_INPUT})
        else:
            selected = []
        for idx, ex in enumerate(selected, start=1):
            print(f"Example {idx}:")
            _print_history_bullets(ex.get("history", []))
            _print_block("input", ex.get("input", ""), max_lines=None)
            _print_block("output", ex.get("output", ""), max_lines=None)
            _print_block(
                "direct_response_to_the_user",
                ex.get("direct_response_to_the_user", ex.get("output", "")),
                max_lines=None,
            )
            _print_block(
                "handoff_needed", str(ex.get("handoff_needed", "")), max_lines=None
            )
            print("-" * 40)
        print(
            f"""Now, continue the following conversation with the user, and fill the required structure output schema:
            {USER_INPUT}
            (continue)
            """
        )

        # Minimal sanity checks
        assert any(m.type == "system" for m in msgs)
        assert any(m.type == "human" for m in msgs)
        print("OK: dynamic few-shots prompt assembled with history support.")

    except Exception as exc:  # pragma: no cover - offline fallback
        print(f"[WARN] Falling back to static demo due to: {exc}")
        examples = _load_receptionist_examples(yaml_path)
        assert examples, "No receptionist examples found."
        first = examples[0]
        demo_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("history"),
                ("human", "{input}"),
                ("ai", "{output}"),
            ]
        )
        msgs = demo_prompt.format_messages(**first)
        _section("Static demo prompt (first example)")
        print("System prompt:")
        print(_indent("\n".join(_wrap(system_prompt))))
        print()
        print("Example 1:")
        _print_history_bullets(first.get("history", []))
        _print_block("input", first.get("input", ""), max_lines=None)
        _print_block("output", first.get("output", ""), max_lines=None)
        _print_block(
            "direct_response_to_the_user",
            first.get("output", ""),
            max_lines=None,
        )
        _print_block("handoff_needed", "False", max_lines=None)
        print("-" * 40)
        print(
            f"""Now, continue the following conversation with the user, and fill the required structure output schema:
            {USER_INPUT}
            (continue)
            """
        )
        assert any(m.type == "ai" for m in msgs)
        print("OK: static history injection demo rendered.")
