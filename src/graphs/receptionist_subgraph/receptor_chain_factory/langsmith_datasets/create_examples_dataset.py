"""Create a LangSmith dataset from concierge few-shot YAML examples.

This script reads `src/graphs/receptionist_subgraph/fewshots.yml`, converts
the concierge examples into LangSmith dataset examples, and uploads them using
the LangSmith Python SDK.

It expects LangSmith credentials to be configured via environment variables.
See the LangSmith docs for programmatic dataset management:
https://docs.langchain.com/langsmith/manage-datasets-programmatically

uv run src/graphs/receptionist_subgraph/receptor_chain_factory/langsmith_datasets/create_examples_dataset.py
"""

# %%
from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from langsmith import Client


load_dotenv(override=True)

FEWSHOTS_YAML_PATH = (
    Path(__file__).resolve().parents[3] / "receptionist_subgraph" / "fewshots.yml"
)


REQUIRED_INFO_KEYS = (
    "name",
    "zip_code",
    "current_employment_status",
    "what_is_the_user_looking_for",
)


@dataclass(frozen=True)
class TransformedExample:
    """Container for the transformed example ready for LangSmith upload."""

    inputs: dict[str, Any]
    outputs: dict[str, Any]
    metadata: dict[str, Any]


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file and return a Python dictionary.

    Args:
        path: Absolute path to the YAML file.

    Returns:
        Parsed YAML content as a dictionary.
    """
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_current_info_block(block: Any) -> dict[str, Any]:
    r"""Parse the `current_collected_user_info` block into a dictionary.

    This field is stored as a folded scalar string that itself contains YAML
    (e.g., "name: Jane\nzip_code: 12345\n..."). We parse it to a dict and
    normalize missing keys to None.
    """
    if block is None:
        parsed: dict[str, Any] = {}
    elif isinstance(block, dict):
        parsed = block
    elif isinstance(block, str):
        # The string likely comes from a folded scalar (">") which removed
        # newlines, making it invalid YAML mapping. Parse by tokenizing known
        # keys and extracting their values.
        text = block.strip()
        key_pattern = (
            r"(name|zip_code|current_employment_status|"
            r"what_is_the_user_looking_for)\s*:\s*"
        )
        matches = list(re.finditer(key_pattern, text))
        tmp: dict[str, Any] = {}
        for i, m in enumerate(matches):
            key = m.group(1)
            start_val = m.end()
            end_val = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            value = text[start_val:end_val].strip()
            if value == "" or value.lower() == "null":
                tmp[key] = None
            else:
                tmp[key] = value
        parsed = tmp
    else:
        parsed = {}

    # Ensure required keys exist; set missing ones to None
    normalized: dict[str, Any] = {}
    for key in REQUIRED_INFO_KEYS:
        normalized[key] = parsed.get(key, None)
    return normalized


def normalize_conversation(history: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    """Normalize conversation messages to a list of {role, content} dicts.

    Input items are dictionaries with a single key: "human" or "ai".
    """
    normalized: list[dict[str, str]] = []
    for item in history:
        if not isinstance(item, dict) or not item:
            # Skip unexpected shapes
            continue
        ((role_key, content),) = item.items()
        role = "human" if role_key.lower() == "human" else "ai"
        normalized.append({"role": role, "content": str(content)})
    return normalized


def transform_example(example: dict[str, Any], index: int) -> TransformedExample:
    """Transform a concierge YAML example to LangSmith inputs/outputs.

    Inputs include the full conversation and the final user message.
    Outputs include the structured concierge response fields.
    """
    history_raw: list[dict[str, str]] = example.get("Conversation_history", [])
    last_user: str = example.get("last_input_from_human_user", "")
    response: dict[str, Any] = example.get(
        "response_from_ia_to_the_last_input_from_user", {}
    )

    inputs: dict[str, Any] = {
        "conversation_history": normalize_conversation(history_raw),
        "last_input_from_human_user": last_user,
    }

    current_info = parse_current_info_block(response.get("current_collected_user_info"))

    outputs: dict[str, Any] = {
        "current_collected_user_info": current_info,
        "any_missing_info_before_handoff": response.get(
            "any_missing_info_before_handoff"
        ),
        "direct_response_to_the_user": response.get("direct_response_to_the_user"),
        "next_agent": response.get("next_agent"),
        "task_to_handoff_to_the_next_agent": response.get(
            "task_to_handoff_to_the_next_agent"
        ),
    }

    metadata: dict[str, Any] = {
        "source": str(FEWSHOTS_YAML_PATH),
        "example_index": index,
        "example_type": "concierge_fewshot",
    }

    return TransformedExample(inputs=inputs, outputs=outputs, metadata=metadata)


def extract_examples(data: dict[str, Any]) -> list[TransformedExample]:
    """Extract and transform all concierge examples from the YAML payload."""
    concierge_examples = data.get("Concierge_examples", [])
    transformed: list[TransformedExample] = []
    for idx, ex in enumerate(concierge_examples):
        transformed.append(transform_example(ex, idx))
    return transformed


def get_or_create_dataset(client: Client, name: str, description: str):
    """Return an existing dataset by name or create a new one."""
    existing = list(client.list_datasets(dataset_name=name))
    if existing:
        return existing[0]
    return client.create_dataset(dataset_name=name, description=description)


def create_langsmith_examples(
    client: Client, dataset_id: str, examples: list[TransformedExample]
) -> None:
    """Bulk create examples in LangSmith from transformed examples."""
    payload = [
        {
            "inputs": ex.inputs,
            "outputs": ex.outputs,
            "metadata": ex.metadata,
        }
        for ex in examples
    ]
    if not payload:
        return
    client.create_examples(dataset_id=dataset_id, examples=payload)


def main(dataset_name: str | None = None) -> None:
    """Create a LangSmith dataset from the concierge few-shot examples."""
    name = dataset_name or "concierge-examples"
    description = (
        "Few-shot concierge examples extracted from receptionist_subgraph/fewshots.yml"
    )

    data = load_yaml(FEWSHOTS_YAML_PATH)
    transformed = extract_examples(data)

    client = Client()
    dataset = get_or_create_dataset(client, name=name, description=description)
    create_langsmith_examples(client, dataset_id=dataset.id, examples=transformed)

    print(
        f"Dataset '{dataset.name}' (id={dataset.id}) populated with "
        f"{len(transformed)} examples from fewshots.yml"
    )


if __name__ == "__main__":
    # Simple demonstration / test
    try:
        main()
    except Exception as exc:
        # Provide a succinct error message without suppressing the exception.
        print(f"Failed to create dataset: {exc}")
        raise
