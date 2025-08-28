"""Utility functions for the agentic workflow.

uv run python src/utils.py
"""

# %%
from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


load_dotenv(override=True)

# Defaults for environment variables (may be overridden by actual env/.env)
os.environ.setdefault("RUN_ADVANCED_RECEPTOR_TEST", "1")


def get_llm():
    """Get the LLM chain."""
    return ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )


def to_plain_data(obj: Any, remove_none: bool = True) -> Any:
    """Recursively convert objects to JSON-serializable plain Python types.

    Handles dataclasses, Pydantic models (v1/v2), mappings, sequences, sets,
    enums, paths, and falls back to ``repr`` for unknown types.
    """
    # Primitives and simple cases
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return str(obj)
        return obj
    if isinstance(obj, Enum):
        return obj.value

    # Dataclasses
    if is_dataclass(obj):
        return to_plain_data(asdict(obj), remove_none=remove_none)

    # Pydantic v2
    if (
        hasattr(obj, "model_dump")
        and callable(obj.model_dump)
        and getattr(obj.__class__, "__module__", "").startswith("pydantic")
    ):
        data = obj.model_dump(mode="json", by_alias=True, exclude_none=remove_none)
        return to_plain_data(data, remove_none=remove_none)

    # Pydantic v1
    if (
        hasattr(obj, "dict")
        and callable(obj.dict)
        and getattr(obj.__class__, "__module__", "").startswith("pydantic")
    ):
        data = obj.dict(exclude_none=remove_none)
        return to_plain_data(data, remove_none=remove_none)

    # Mappings
    if isinstance(obj, Mapping):
        result: dict[str, Any] = {}
        for key, value in obj.items():
            if remove_none and value is None:
                continue
            result[str(key)] = to_plain_data(value, remove_none=remove_none)
        return result

    # Sequences / Sets
    if isinstance(obj, (list, tuple, set, frozenset)):
        items = [to_plain_data(v, remove_none=remove_none) for v in obj]
        if isinstance(obj, (set, frozenset)):
            try:
                return sorted(items, key=str)
            except (TypeError, ValueError):
                return items
        return items

    # Paths
    if isinstance(obj, Path):
        return str(obj)

    # Bytes
    if isinstance(obj, (bytes, bytearray, memoryview)):
        try:
            return bytes(obj).decode("utf-8")
        except UnicodeDecodeError:
            return repr(obj)

    # Generic object with __dict__ (e.g., langgraph Command)
    if hasattr(obj, "__dict__"):
        try:
            public_attrs = {k: v for k, v in vars(obj).items() if not k.startswith("_")}
            if public_attrs:
                return to_plain_data(public_attrs, remove_none=remove_none)
        except TypeError:
            # vars() argument does not have __dict__
            pass

    # Fallback
    return repr(obj)


def format_as_json(obj: Any, include_none: bool = False) -> str:
    """Return a pretty JSON string for any Python/third-party object.

    ``include_none`` controls whether ``None``-valued fields are kept or
    omitted from mappings and supported model types.
    """
    plain = to_plain_data(obj, remove_none=not include_none)
    try:
        return json.dumps(plain, indent=2, ensure_ascii=False, sort_keys=True)
    except TypeError:
        # As a safety net, allow ``repr`` for anything not serializable
        return json.dumps(
            plain, indent=2, ensure_ascii=False, sort_keys=True, default=repr
        )


def format_command(command: Any, include_none: bool = False) -> str:
    """Format a langgraph ``Command`` (or similar) as pretty JSON."""
    return format_as_json(command, include_none=include_none)


def pretty_print(obj: Any, include_none: bool = False) -> None:
    """Print a readable JSON representation of ``obj`` to stdout."""
    print(format_as_json(obj, include_none=include_none))


if __name__ == "__main__":
    # Demo: pretty-print a Command-like object safely without extra deps
    from dataclasses import dataclass

    @dataclass
    class DemoCommand:
        goto: str | None
        update: dict[str, Any]

    demo = DemoCommand(
        goto="validate_user_profile",
        update={
            "receptionist_output_schema": {
                "direct_response_to_the_user": (
                    "Happy to help. What's your name and current address?"
                ),
                "user_name": None,
                "user_current_address": None,
            }
        },
    )

    print("Pretty JSON for demo Command:\n")
    pretty_print(demo)

    # Optional: run the LLM demo only when explicitly requested
    if os.environ.get("RUN_LLM_DEMO") == "1":
        import asyncio

        async def _llm_demo() -> None:
            result = await get_llm().ainvoke("What is the capital of France?")
            print(result)

        asyncio.run(_llm_demo())

# %%
