"""This module contains the logic to create a dynamic few-shot chat prompt template.

uv run python src/graphs/llm_chains_factory/dynamic_fewshots.py
"""

# %%
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings


load_dotenv(override=True)

# Default path to the few-shots YAML colocated with this module
DEFAULT_FEWSHOTS_PATH = Path(__file__).parent / "test_fewshots.yaml"


def _transform_sequential_pairs(
    items: list[dict],
) -> list[dict]:
    """Transform sequential input/output pairs while preserving combined items.

    - If an item already contains both 'input' and 'output', keep it as-is
      (including any extra keys like 'Conversation_history').
    - Otherwise, pair {'input': ...} followed by {'output': ...} into one dict.
    - Skip incomplete or malformed entries safely.
    """
    transformed: list[dict] = []
    index = 0
    n = len(items)
    while index < n:
        item1 = items[index]
        if not isinstance(item1, dict):  # Skip non-dict items defensively
            index += 1
            continue
        # Preserve combined items (may include Conversation_history)
        if "input" in item1 and "output" in item1:
            transformed.append(item1)
            index += 1
            continue
        # Try to pair with the next item
        if "input" in item1 and index + 1 < n:
            item2 = items[index + 1]
            if isinstance(item2, dict) and "output" in item2:
                transformed.append(
                    {
                        "input": item1.get("input", ""),
                        "output": item2.get("output", ""),
                    }
                )
                index += 2
                continue
        # Fallback: advance by one to allow recovery
        index += 1
    return transformed


def _coerce_output_text(value: Any) -> str:
    """Convert arbitrary YAML 'output' values to a reasonable text string.

    - If the value is a mapping, prefer well-known text fields.
    - Fall back to YAML-dumping the structure for stability.
    """
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        # Prefer common text-bearing fields
        for key in (
            "direct_response_to_the_user",
            "answer",
            "text",
            "response",
            "next_subgraph",
        ):
            if key in value and value[key] is not None:
                text = str(value[key]).strip()
                if text:
                    return text
        # Fallback: stable textual form
        try:
            dumped = yaml.safe_dump(value, sort_keys=True)
            return str(dumped).strip()
        except Exception:  # pragma: no cover
            return str(value)
    return str(value).strip()


def render_examples_for_system(examples: list[dict]) -> str:
    """Render selected examples as a formatted system block.

    Format:
        Example N:
        Conversation_history:
            - human: ...
            - ai: ...
        - input:
            ...
        - output:
            ...
        - direct_response_to_the_user:
            ...
        - handoff_needed:
            ...
        ----------------------------------------
    """
    lines: list[str] = []
    for idx, ex in enumerate(examples, start=1):
        lines.append(f"Example {idx}:")
        lines.append("Conversation_history:")

        history = ex.get("history", [])
        for turn in history:
            # Support BaseMessage or dict turns
            role = None
            content = ""
            if isinstance(turn, (HumanMessage, AIMessage)):
                role = "human" if isinstance(turn, HumanMessage) else "ai"
                content = str(getattr(turn, "content", ""))
            elif isinstance(turn, dict):
                if "human" in turn:
                    role = "human"
                    content = str(turn.get("human", ""))
                elif "ai" in turn:
                    role = "ai"
                    content = str(turn.get("ai", ""))
            if role:
                lines.append(f"    - {role}:")
                for line in str(content).splitlines() or [""]:
                    lines.append(f"        {line}")

        def _blk(label: str, value: str) -> None:
            lines.append(f"- {label}:")
            for line in str(value).splitlines() or [""]:
                lines.append(f"    {line}")

        _blk("input", str(ex.get("input", "")))
        output_text = str(ex.get("output", ""))
        _blk(
            "direct_response_to_the_user",
            str(ex.get("direct_response_to_the_user", output_text)),
        )
        _blk("handoff_needed", str(ex.get("handoff_needed", "")))
        lines.append("----------------------------------------")

    return "\n".join(lines)


def create_dynamic_fewshooter(
    yaml_path: Path | None = None,
    input_key: str = "input",
    output_key: str = "output",
    *,
    k: int = 2,
    selector_input_variable: str = "input",
    group: str | None = None,
) -> FewShotChatMessagePromptTemplate:
    """Create a dynamic few-shot chat prompt template using semantic selection.

    - Loads examples from a YAML list of dicts or from a grouped YAML dict
      where the values are lists (e.g., FEW_SHOTS_QUESTIONS_GENERATION,
      FEW_SHOTS_CYPHER_QUERY).
    - Auto-detects input/output keys if defaults are not present.
    - Skips incomplete items safely.

    Args:
        yaml_path: Optional path to YAML. Defaults to `fewshots.yaml` in this module.
        input_key: Preferred input field name if present.
        output_key: Preferred output field name if present.
        k: Number of examples to select.
        selector_input_variable: The variable name whose value will drive example selection.
        group: Optional group name inside the YAML when it is a mapping of
            groups to lists. If not provided and the YAML is grouped, the
            function auto-detects the best group based on key-pair matches.

    Returns:
        FewShotChatMessagePromptTemplate ready to be composed in a chat prompt.
    """
    yaml_path = yaml_path or DEFAULT_FEWSHOTS_PATH

    # Read and validate YAML structure
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    rows: list[dict]

    # Auto-detect best matching key pair by frequency
    candidate_pairs: list[tuple[str, str]] = [
        (input_key, output_key),
        ("pregunta", "generated_queries"),  # Questions generation dataset
        ("pregunta", "cypher_query"),  # Cypher dataset
        ("question", "answer"),  # Common fallback
    ]

    def count_pair(items: Iterable[dict], inp: str, out: str) -> int:
        return sum(1 for it in items if inp in it and out in it)

    def _select_rows_from_grouped_yaml(obj: dict) -> list[dict]:
        # Validate candidate lists inside the mapping
        list_like_groups: dict[str, list] = {
            k: v for k, v in obj.items() if isinstance(v, list)
        }
        if not list_like_groups:
            raise ValueError(
                "YAML mapping has no list-like groups. Available keys: "
                f"{sorted(obj.keys())}"
            )

        # Pre-transform sequential pairs if detected
        for g, items in list_like_groups.items():
            if all("input" in d or "output" in d for d in items if isinstance(d, dict)):
                list_like_groups[g] = _transform_sequential_pairs(items)

        if group is not None:
            if group not in list_like_groups:
                raise KeyError(
                    "Requested group '{group}' not found. Available groups: "
                    f"{sorted(list_like_groups.keys())}"
                )
            selected_rows = [
                it for it in list_like_groups[group] if isinstance(it, dict)
            ]
            return selected_rows

        # Auto-detect best group by which one yields the highest key-pair coverage
        scored: list[tuple[str, int]] = []
        for gname, items in list_like_groups.items():
            dict_items = [it for it in items if isinstance(it, dict)]
            score = 0
            for cand in candidate_pairs:
                score += count_pair(dict_items, *cand)
            scored.append((gname, score))
        scored.sort(key=lambda x: x[1], reverse=True)

        best_group, best_score = scored[0]
        if best_score == 0:
            # Fallback: flatten all lists
            merged: list[dict] = []
            for items in list_like_groups.values():
                merged.extend([it for it in items if isinstance(it, dict)])
            return merged
        return [it for it in list_like_groups[best_group] if isinstance(it, dict)]

    if isinstance(data, list):
        rows = [row for row in data if isinstance(row, dict)]
        # Check for sequential pair format in a top-level list
        if all("input" in d or "output" in d for d in rows):
            rows = _transform_sequential_pairs(rows)
    elif isinstance(data, dict):
        rows = _select_rows_from_grouped_yaml(data)
    else:
        raise ValueError(
            "Unsupported YAML root type "
            f"{type(data).__name__}. "
            "Expected list or mapping of lists: "
            f"{yaml_path}"
        )

    if not rows:
        raise ValueError(f"No dictionary examples found in YAML file: {yaml_path}")

    pair_counts = [(pair, count_pair(rows, *pair)) for pair in candidate_pairs]
    pair_counts.sort(key=lambda x: x[1], reverse=True)
    best_pair, best_count = pair_counts[0]
    if best_count == 0:
        available_keys = sorted({k for it in rows for k in it})
        raise KeyError(
            f"Could not infer input/output keys for {yaml_path}. "
            f"Tried {candidate_pairs}. Available keys: {available_keys}"
        )
    source_input_key, source_output_key = best_pair

    # Build normalized examples; include optional Conversation_history
    examples: list[dict] = []
    for it in rows:
        if (
            source_input_key in it
            and source_output_key in it
            and str(it[source_input_key]).strip()
            and str(it[source_output_key]).strip()
        ):
            history_msgs: list[BaseMessage] = []
            conv = it.get("Conversation_history")
            if isinstance(conv, list):
                for turn in conv:
                    if not isinstance(turn, dict):
                        continue
                    if "human" in turn:
                        history_msgs.append(
                            HumanMessage(content=str(turn.get("human", "")))
                        )
                    if "ai" in turn:
                        history_msgs.append(AIMessage(content=str(turn.get("ai", ""))))
            # Extract and normalize output + flags
            output_obj = it.get(source_output_key)
            direct_text = _coerce_output_text(output_obj)
            handoff_needed_val: str = ""
            if isinstance(output_obj, dict):
                hn = output_obj.get("handoff_needed", None)
                if hn is not None:
                    handoff_needed_val = str(hn)

            # Pre-render history as plain text for system injection
            history_text = "\n".join(
                [
                    (
                        ("human: " + m.content)
                        if isinstance(m, HumanMessage)
                        else ("ai: " + m.content)
                    )
                    for m in history_msgs
                    if getattr(m, "content", "")
                ]
            )

            examples.append(
                {
                    "input": str(it[source_input_key]).strip(),
                    "output": direct_text,
                    "direct_response_to_the_user": direct_text,
                    "handoff_needed": handoff_needed_val,
                    # Always include history key for template stability
                    "history": history_msgs,
                    "history_text": history_text,
                }
            )
    if not examples:
        raise ValueError(f"No valid examples after normalization from: {yaml_path}")

    # Prepare texts to vectorize (explicit order for stability)
    # Vectorize only the user's question (input) for similarity matching
    to_vectorize = [ex["input"] for ex in examples]

    # Initialize embeddings with robust fallback
    embeddings = None
    init_errors: list[str] = []
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    except Exception as exc:
        init_errors.append(f"OpenAIEmbeddings failed: {exc}")
    if embeddings is None:
        try:
            embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        except Exception as exc:
            init_errors.append(f"OpenAIEmbeddings failed: {exc}")
            raise RuntimeError(
                "No embeddings backend available. "
                "Set Azure OpenAI or OpenAI credentials. " + "; ".join(init_errors)
            ) from exc

    # Build selector with safe k
    effective_k = max(1, min(k, len(examples)))
    vectorstore = InMemoryVectorStore.from_texts(
        to_vectorize, embeddings, metadatas=examples
    )
    example_selector = SemanticSimilarityExampleSelector(
        vectorstore=vectorstore,
        k=effective_k,
    )

    return FewShotChatMessagePromptTemplate(
        # The input variables select the values to pass to the example_selector
        input_variables=[selector_input_variable],
        example_selector=example_selector,
        # Define how each example will be formatted.
        # Each example injects full context into a system segment, then replays
        # the history and the input/output as chat messages.
        example_prompt=ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "handoff_needed: {handoff_needed}",
                ),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}"),
                ("ai", "{output}"),
            ]
        ),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create and test a dynamic few-shot prompt."
    )
    parser.add_argument(
        "--yaml_path",
        type=Path,
        default=Path(__file__).parent / "test_fewshots.yml",
        help="Path to the YAML file with few-shot examples.",
    )
    parser.add_argument("-k", type=int, default=2, help="Number of examples to select.")
    parser.add_argument(
        "--group",
        type=str,
        default="TARGET_LLM_CHAIN_2",
        help="Group name inside the YAML file.",
    )
    args = parser.parse_args()

    SYSTEM_PROMPT = (
        "You are a helpful assistant that can answer questions about the graph."
    )
    FEW_SHOT_PROMPT = create_dynamic_fewshooter(
        yaml_path=args.yaml_path, k=args.k, group=args.group
    )
    # Consolidate system messages for Bedrock compatibility
    CONSOLIDATED_SYSTEM = (
        f"{SYSTEM_PROMPT}\n\n"
        "## A continuación, ejemplos de preguntas y respuestas parecidas:"
    )
    TEST_PROMPT = ChatPromptTemplate.from_messages(
        [
            ("system", CONSOLIDATED_SYSTEM),
            FEW_SHOT_PROMPT,
            ("human", "{input}"),
        ]
    )
    # Demo: inspect which few-shot examples were selected for a given input
    DEMO_INPUT = "proyectos en las comunas Antofagasta o Mejillones"
    SELECTOR = getattr(FEW_SHOT_PROMPT, "example_selector", None)

    def _indent(text: str, prefix: str = "    ") -> str:
        return "\n".join(prefix + line for line in str(text).splitlines())

    def _section(title: str) -> None:
        print("\n" + "=" * 80)
        print(title)
        print("=" * 80)

    if SELECTOR is not None:
        selected = SELECTOR.select_examples({"input": DEMO_INPUT})

        _section("Entrada de demostración")
        print(DEMO_INPUT)

        _section(f"Ejemplos few-shot seleccionados (k={len(selected)})")
        for idx, ex in enumerate(selected, start=1):
            INPUT_TEXT = str(ex.get("input", "")).strip()
            OUTPUT_TEXT = str(ex.get("output", "")).strip()
            print(f"[{idx}] INPUT:")
            print(_indent(INPUT_TEXT))
            print("    OUTPUT:")
            print(_indent(OUTPUT_TEXT))
            print("-" * 80)

        _section("Prompt final (mensajes en orden)")
        messages = TEST_PROMPT.format_messages(input=DEMO_INPUT)
        for i, msg in enumerate(messages, start=1):
            content = getattr(msg, "content", "")
            print(f"{i:02d}. [{msg.type}]")
            print(_indent(content))
            print()
    else:
        print("No example_selector found on FEW_SHOT_PROMPT.")
