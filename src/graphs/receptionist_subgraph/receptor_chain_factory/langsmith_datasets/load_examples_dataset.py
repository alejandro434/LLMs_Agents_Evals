"""Load the LangSmith dataset created from concierge few-shot YAML examples.

This module fetches the dataset created by the companion script and returns
the list of examples. It uses the LangSmith Python SDK as documented here:
https://docs.langchain.com/langsmith/manage-datasets-programmatically
"""

# %%
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from langsmith import Client


load_dotenv(override=True)


DEFAULT_DATASET_NAME = "concierge-examples"


@dataclass(frozen=True)
class LoadedExample:
    """A loaded example from LangSmith.

    Attributes:
        id: The example ID in LangSmith.
        inputs: The inputs payload for the example.
        outputs: The outputs payload for the example, if present.
        metadata: Any metadata attached to the example, if present.
    """

    id: str
    inputs: dict[str, Any]
    outputs: dict[str, Any] | None
    metadata: dict[str, Any] | None


def get_dataset_by_name(client: Client, dataset_name: str):
    """Return an existing dataset by exact name or raise if not found."""
    datasets = list(client.list_datasets(dataset_name=dataset_name))
    if not datasets:
        msg = (
            f"Dataset '{dataset_name}' not found. Ensure it exists or run the "
            "creation script first."
        )
        raise RuntimeError(msg)
    return datasets[0]


def load_examples(dataset_name: str = DEFAULT_DATASET_NAME) -> list[LoadedExample]:
    """Load examples from a LangSmith dataset by name.

    Args:
        dataset_name: The name of the dataset to load.

    Returns:
        A list of LoadedExample objects.
    """
    client = Client()
    dataset = get_dataset_by_name(client, dataset_name)
    examples = list(client.list_examples(dataset_id=dataset.id))

    loaded: list[LoadedExample] = []
    for ex in examples:
        loaded.append(
            LoadedExample(
                id=str(ex.id),
                inputs=dict(ex.inputs or {}),
                outputs=dict(ex.outputs or {}) if ex.outputs else None,
                metadata=dict(getattr(ex, "metadata", {}) or {})
                if getattr(ex, "metadata", None)
                else None,
            )
        )
    return loaded


if __name__ == "__main__":
    # Simple demonstration / test
    try:
        loaded_examples = load_examples()
        print(f"Loaded {len(loaded_examples)} examples from '{DEFAULT_DATASET_NAME}'")
        # Show a small preview for sanity
        if loaded_examples:
            first = loaded_examples[0]
            print(
                {
                    "id": first.id,
                    "inputs_keys": list(first.inputs.keys()),
                    "outputs_keys": list(first.outputs.keys()) if first.outputs else [],
                }
            )
    except Exception as exc:
        print(f"Failed to load dataset: {exc}")
        raise
