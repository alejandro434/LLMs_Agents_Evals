"""Receptionist dynamic few-shots builder using concierge fewshots.yml.

This mirrors the API of `src/graphs/llm_chains_factory/dynamic_fewshots.py`,
but sources examples from `src/graphs/receptionist_subgraph/fewshots.yml`.
uv run src/graphs/receptionist_subgraph/receptor_chain_factory/dynamic_fewshots.py
"""

# %%
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings


# Support both module and script execution by preferring absolute import and
# falling back to injecting the repository root into sys.path when needed.
try:  # First, try absolute import (works when run as a package/module)
    from src.graphs.receptionist_subgraph.receptor_chain_factory.langsmith_datasets.load_examples_dataset import (
        DEFAULT_DATASET_NAME,
        LoadedExample,
        load_examples as load_langsmith_examples,
    )
except Exception:  # pragma: no cover - fallback for direct script execution
    import sys

    REPO_ROOT = Path(__file__).resolve().parents[4]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from src.graphs.receptionist_subgraph.receptor_chain_factory.langsmith_datasets.load_examples_dataset import (
        DEFAULT_DATASET_NAME,
        LoadedExample,
        load_examples as load_langsmith_examples,
    )


DEFAULT_FEWSHOTS_PATH = Path(__file__).resolve().parents[1] / "fewshots.yml"
load_dotenv(override=True)


def _coerce_output_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in (
            "direct_response_to_the_user",
            "answer",
            "text",
            "response",
        ):
            if key in value and value[key] is not None:
                text = str(value[key]).strip()
                if text:
                    return text
        try:
            dumped = yaml.safe_dump(value, sort_keys=True)
            return str(dumped).strip()
        except yaml.YAMLError:
            return str(value)
    return str(value).strip()


def _normalize_from_concierge_yaml(data: Any) -> list[dict]:
    """Extract examples from `Concierge_examples` into common format.

    Each returned dict contains: input, output, history (BaseMessage list),
    history_text, and direct_response_to_the_user.
    """
    examples: list[dict] = []
    concierge_examples: list[Any] = []
    if isinstance(data, dict):
        concierge_examples = list(data.get("Concierge_examples", []) or [])
    if not isinstance(concierge_examples, list):
        return examples

    def _build_history_messages(history_raw: Any) -> list[BaseMessage]:
        msgs: list[BaseMessage] = []
        if isinstance(history_raw, list):
            for turn in history_raw:
                if not isinstance(turn, dict):
                    continue
                if "human" in turn:
                    msgs.append(HumanMessage(content=str(turn.get("human", ""))))
                if "ai" in turn:
                    msgs.append(AIMessage(content=str(turn.get("ai", ""))))
        return msgs

    def _as_history_text(msgs: list[BaseMessage]) -> str:
        return "\n".join(
            [
                (
                    ("human: " + m.content)
                    if isinstance(m, HumanMessage)
                    else ("ai: " + m.content)
                )
                for m in msgs
                if getattr(m, "content", "")
            ]
        )

    def _extract_example(ex: Any) -> dict | None:
        if not isinstance(ex, dict):
            return None
        history_raw = ex.get("Conversation_history", []) or []
        last_user = ex.get("last_input_from_human_user", "")
        response_block = (
            ex.get("response_from_ia_to_the_last_input_from_user", {}) or {}
        )
        output_text = _coerce_output_text(
            response_block.get("direct_response_to_the_user")
        )
        history_msgs = _build_history_messages(history_raw)
        history_text = _as_history_text(history_msgs)
        if str(last_user).strip() and str(output_text).strip():
            return {
                "input": str(last_user).strip(),
                "output": str(output_text).strip(),
                "direct_response_to_the_user": str(output_text).strip(),
                "history": history_msgs,
                "history_text": history_text,
            }
        return None

    for ex in concierge_examples:
        row = _extract_example(ex)
        if row is not None:
            examples.append(row)
    return examples


def _normalize_from_langsmith_dataset(
    loaded: list[LoadedExample],
) -> list[dict]:
    """Extract examples from LangSmith dataset into common format.

    Each returned dict contains: input, output, history (BaseMessage list),
    history_text, and direct_response_to_the_user.
    """
    examples: list[dict] = []

    def _build_history_messages(history_raw: Any) -> list[BaseMessage]:
        msgs: list[BaseMessage] = []
        if isinstance(history_raw, list):
            for turn in history_raw:
                if not isinstance(turn, dict):
                    continue
                role = str(turn.get("role", "")).strip().lower()
                content = str(turn.get("content", ""))
                if role == "human":
                    msgs.append(HumanMessage(content=content))
                elif role == "ai":
                    msgs.append(AIMessage(content=content))
        return msgs

    def _as_history_text(msgs: list[BaseMessage]) -> str:
        return "\n".join(
            [
                (
                    ("human: " + m.content)
                    if isinstance(m, HumanMessage)
                    else ("ai: " + m.content)
                )
                for m in msgs
                if getattr(m, "content", "")
            ]
        )

    for ex in loaded:
        inputs = ex.inputs or {}
        outputs = ex.outputs or {}
        history_raw = inputs.get("conversation_history", []) or []
        last_user = inputs.get("last_input_from_human_user", "")
        output_text = _coerce_output_text(outputs.get("direct_response_to_the_user"))
        history_msgs = _build_history_messages(history_raw)
        history_text = _as_history_text(history_msgs)
        if str(last_user).strip() and str(output_text).strip():
            examples.append(
                {
                    "input": str(last_user).strip(),
                    "output": str(output_text).strip(),
                    "direct_response_to_the_user": str(output_text).strip(),
                    "history": history_msgs,
                    "history_text": history_text,
                }
            )
    return examples


def _read_yaml(path: Path) -> Any:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"fewshots.yml not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {path}: {exc}") from exc


def create_dynamic_fewshooter(
    yaml_path: Path | None = None,
    *,
    k: int = 5,
    selector_input_variable: str = "input",
    embeddings: Embeddings | None = None,
    prefer_langsmith: bool = True,
    dataset_name: str | None = None,
) -> FewShotChatMessagePromptTemplate:
    """Create a dynamic few-shot chat prompt for receptionist concierge.

    Behavior:
    - If `prefer_langsmith` is True, attempt to load examples from the LangSmith
      dataset `dataset_name` (defaults to a sensible name). If loading fails or
      returns no usable examples, fall back to reading from `fewshots.yml`.
    - Selects up to k semantically similar examples based on the last user input
      and formats them as messages: history + human/ai turns.
    """
    examples: list[dict] = []

    if prefer_langsmith:
        ds_name = dataset_name or DEFAULT_DATASET_NAME
        try:
            loaded = load_langsmith_examples(dataset_name=ds_name)
            examples = _normalize_from_langsmith_dataset(loaded)
        except Exception:
            # Silent fallback to YAML if dataset is unavailable/misconfigured
            examples = []

    if not examples:
        yaml_path = yaml_path or DEFAULT_FEWSHOTS_PATH
        data = _read_yaml(yaml_path)
        examples = _normalize_from_concierge_yaml(data)
        if not examples:
            raise ValueError(f"No valid concierge examples found in: {yaml_path}")

    to_vectorize = [ex["input"] for ex in examples]
    embeddings = embeddings or OpenAIEmbeddings(model="text-embedding-3-small")
    effective_k = max(1, min(k, len(examples)))
    vectorstore = InMemoryVectorStore.from_texts(
        to_vectorize, embeddings, metadatas=examples
    )
    example_selector = SemanticSimilarityExampleSelector(
        vectorstore=vectorstore,
        k=effective_k,
    )

    return FewShotChatMessagePromptTemplate(
        input_variables=[selector_input_variable],
        example_selector=example_selector,
        example_prompt=ChatPromptTemplate.from_messages(
            [
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}"),
                ("ai", "{output}"),
            ]
        ),
    )


def render_examples_for_system(examples: list[dict]) -> str:
    """Render selected examples for inclusion inside a system block."""
    lines: list[str] = []
    for idx, row in enumerate(examples, start=1):
        lines.append(f"Example {idx}:")
        lines.append("Conversation_history:")
        for turn in row.get("history", []):
            role = "human" if isinstance(turn, HumanMessage) else "ai"
            content = getattr(turn, "content", "")
            lines.append(f"    - {role}:")
            for line in str(content).splitlines() or [""]:
                lines.append(f"        {line}")
        lines.append("- input:")
        for line in str(row.get("input", "")).splitlines() or [""]:
            lines.append(f"    {line}")
        lines.append("- direct_response_to_the_user:")
        out = str(row.get("direct_response_to_the_user", row.get("output", "")))
        for line in out.splitlines() or [""]:
            lines.append(f"    {line}")
        lines.append("----------------------------------------")
    return "\n".join(lines)


if __name__ == "__main__":
    tmpl = create_dynamic_fewshooter()
    selector = getattr(tmpl, "example_selector", None)
    if selector is not None:
        selected = selector.select_examples({"input": "Looking for retail jobs"})
        print(render_examples_for_system(selected))
