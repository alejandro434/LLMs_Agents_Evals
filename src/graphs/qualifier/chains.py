"""Qualifier subgraph chains and minimal demo tests for manual execution.

uv run -m src.graphs.qualifier.chains
"""

# %%
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableLambda

from src.graphs.llm_chains_factory.assembling import (
    build_structured_chain,
)
from src.graphs.qualifier.schemas import (
    QualifierOutputSchema,
    UserInfoOutputSchema,
)


_BASE_DIR = Path(__file__).parent


def _load_system_prompt(relative_file: str, key: str) -> str:
    """Load a system prompt string from a YAML file in this package."""
    with (_BASE_DIR / relative_file).open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)[key]


def _require(condition: bool, message: str) -> None:
    """Raise AssertionError with message if condition is False.

    Used instead of bare asserts to satisfy linting rules.
    """
    if not condition:
        raise AssertionError(message)


def _extract_zip5(raw_zip: str | None) -> int | None:
    """Return the first 5-digit ZIP as int if present; otherwise None.

    Accepts strings like "22102", "22102-1234", or noisy text containing digits.
    """
    if not raw_zip:
        return None
    digits = "".join(ch for ch in str(raw_zip) if ch.isdigit())
    if len(digits) < 5:
        return None
    try:
        return int(digits[:5])
    except ValueError:
        return None


def _infer_state_from_zip(zip5: int) -> str | None:
    """Map a 5-digit ZIP code to a US state (subset used by our use-cases).

    Returns the full state name if recognized, else None.
    Ranges (inclusive) are approximate and targeted to MD/VA examples:
      - Maryland: 20600–21999
      - Virginia: 20100–20199 and 22000–24699
      - California: 90000–96199
      - New York: 10000–14999
      - Texas: 75000–79999, 88500–88599
    """
    if 20600 <= zip5 <= 21999:
        return "Maryland"
    if 20100 <= zip5 <= 20199 or 22000 <= zip5 <= 24699:
        return "Virginia"
    if 90000 <= zip5 <= 96199:
        return "California"
    if 10000 <= zip5 <= 14999:
        return "New York"
    if 75000 <= zip5 <= 79999 or 88500 <= zip5 <= 88599:
        return "Texas"
    return None


def _enforce_state_from_zip_user_info(
    result: UserInfoOutputSchema,
) -> UserInfoOutputSchema:
    """Ensure state is inferred from ZIP when ZIP is present.

    - If a ZIP is provided (even if state is missing or conflicting), set state
      based on ZIP. If ZIP can't be mapped, leave state as-is.
    - Return a possibly updated model instance.
    """
    zip5 = _extract_zip5(getattr(result, "zip_code", None))
    if zip5 is None:
        return result
    inferred = _infer_state_from_zip(zip5)
    if inferred is None:
        return result
    # Prefer ZIP-derived state
    return result.model_copy(update={"state": inferred})


def _build_runtime_injection(raw_input: str) -> str | None:
    """Create a small context snippet with ZIP→state inference for the LLM.

    This is injected into the system block so the LLM is explicitly told the
    ZIP-derived state and to prefer it over textual mentions when conflicting.
    """
    zip5 = _extract_zip5(raw_input)
    if zip5 is None:
        return None
    inferred = _infer_state_from_zip(zip5)
    if inferred is None:
        return None
    return (
        "INFERRED CONTEXT (ZIP-BASED):\n"
        f"- Inferred state: {inferred}\n"
        f"- ZIP: {zip5:05d}\n"
        "- If any textual state conflicts, prefer the ZIP-derived state."
    )


def get_user_info_collection_chain(
    *,
    k: int = 5,
    temperature: float = 0,
    current_history: Sequence[BaseMessage | dict[str, Any]] | None = None,
):
    """Construct the user info collection structured-output chain for qualifier subgraph.

    Args:
        k: Number of few-shot examples to include.
        temperature: LLM temperature for generation.
        current_history: Optional conversation history to include in the prompt.

    Returns:
        A structured chain that collects user information.
    """
    system_prompt = _load_system_prompt(
        "prompts/system_prompt.yml", "UserInfoCollectionSystemPrompt"
    )
    return build_structured_chain(
        system_prompt=system_prompt,
        output_schema=UserInfoOutputSchema,
        k=k,
        temperature=temperature,
        postprocess=_enforce_state_from_zip_user_info,
        group="UserInfoCollectionFewshots",
        yaml_path=_BASE_DIR / "prompts" / "fewshots.yml",
        current_history=list(current_history) if current_history else None,
    )


def get_qualifier_chain(
    *,
    k: int = 5,
    temperature: float = 0,
    current_history: Sequence[BaseMessage | dict[str, Any]] | None = None,
):
    """Construct the qualifier chain with ZIP→state inference enforced.

    We inject a runtime context snippet that states the ZIP-derived state so the
    LLM consistently prefers the ZIP-based inference when present.
    """
    system_prompt = _load_system_prompt(
        "prompts/system_prompt.yml", "QualifierSystemPrompt"
    )
    base_chain = build_structured_chain(
        system_prompt=system_prompt,
        output_schema=QualifierOutputSchema,
        k=k,
        temperature=temperature,
        postprocess=None,
        group="QualifierFewshots",
        yaml_path=_BASE_DIR / "prompts" / "fewshots.yml",
        current_history=list(current_history) if current_history else None,
    )

    async def _invoke_with_zip_injection(
        raw: str | dict[str, Any],
    ) -> QualifierOutputSchema:
        # Normalize input and preserve optional runtime history if provided
        if isinstance(raw, dict):
            raw_input = str(raw.get("input", ""))
            runtime_history = raw.get("current_history")
        else:
            raw_input = str(raw)
            runtime_history = None

        injection = _build_runtime_injection(raw_input)
        payload: dict[str, Any] = {"input": raw_input}
        if injection:
            payload["runtime_context_injection"] = injection
        if runtime_history is not None:
            payload["current_history"] = runtime_history

        result = await base_chain.ainvoke(payload)
        return result

    return RunnableLambda(_invoke_with_zip_injection)


user_info_collection_chain = get_user_info_collection_chain()
qualifier_chain = get_qualifier_chain()
if __name__ == "__main__":
    import asyncio

    async def demo_user_info_collection_chain() -> None:
        """Exercise user info collection with multiple realistic inputs."""
        inputs = [
            ("I'm 25 years old and I live in Maryland. My zip code is 21201.",),
            ("I'm 17 and from Virginia.",),
            ("I'm 30, living in California, zip 94016.",),
            ("I live in VA, zip 22102.",),
        ]
        for (text,) in inputs:
            result = await user_info_collection_chain.ainvoke(text)
            # Basic schema validations
            _require(
                isinstance(result, UserInfoOutputSchema),
                "Expected UserInfoOutputSchema result",
            )
            if result.age is not None:
                _require(isinstance(result.age, int), "age must be int or None")
                _require(result.age >= 0, "age must be non-negative")
            if result.state is not None:
                _require(isinstance(result.state, str), "state must be str or None")
                _require(len(result.state) > 0, "state must be non-empty")
            if result.zip_code is not None:
                _require(
                    isinstance(result.zip_code, str),
                    "zip_code must be str or None",
                )
                # Allow empty string (treated as missing/unknown ZIP)
            print("UserInfo:", result.model_dump_json(indent=2))

    async def demo_qualifier_chain() -> None:
        """Exercise qualifier logic and check core invariants from schemas."""
        cases = [
            ("I'm 25 in Maryland. Zip 21201.", True),
            ("I'm 20 in VA, zip 22102.", True),
            ("I'm 17 in Virginia.", False),
            ("I'm 30 in California.", False),
        ]
        for text, expected in cases:
            result = await qualifier_chain.ainvoke(text)
            _require(
                isinstance(result, QualifierOutputSchema),
                "Expected QualifierOutputSchema result",
            )
            _require(isinstance(result.qualified, bool), "qualified must be bool")
            if not result.qualified:
                _require(
                    result.why_not_qualified is None
                    or isinstance(result.why_not_qualified, str),
                    "why_not_qualified must be str or None",
                )
            # Expected outcomes derived from business rules in schemas.py
            _require(
                result.qualified == expected,
                f"Expected {expected}, got {result.qualified}",
            )
            print("Qualifier:", result.model_dump_json(indent=2))

    async def test_zip_inference_user_info() -> None:
        """Verify that ZIP-only inputs infer the correct state."""
        samples = [
            ("21201", {"maryland", "md"}),
            ("22102", {"virginia", "va"}),
        ]
        for zip_code, expected_states in samples:
            text = f"My ZIP is {zip_code}."
            result = await user_info_collection_chain.ainvoke(text)
            _require(
                isinstance(result, UserInfoOutputSchema),
                "Expected UserInfoOutputSchema result",
            )
            _require(
                isinstance(result.zip_code, str) and result.zip_code == zip_code,
                "ZIP must be echoed and match input",
            )
            _require(
                isinstance(result.state, str)
                and result.state.lower() in expected_states,
                f"State {result.state!r} not inferred from ZIP {zip_code}",
            )

    async def test_zip_conflict_and_boundary_in_qualifier() -> None:
        """Check ZIP vs text conflict resolution and age boundaries."""
        # Conflict: claims CA but MD ZIP → prefer ZIP (MD), qualified
        text_conflict = "I live in CA, ZIP 21201. I'm 25."
        res_conflict = await qualifier_chain.ainvoke(text_conflict)
        _require(isinstance(res_conflict, QualifierOutputSchema), "bad schema")
        _require(res_conflict.qualified is True, "Should qualify via MD ZIP")

        # Boundary: 18 in Maryland → qualified
        text_boundary_ok = "I'm 18 in Maryland."
        res_boundary_ok = await qualifier_chain.ainvoke(text_boundary_ok)
        _require(isinstance(res_boundary_ok, QualifierOutputSchema), "bad schema")
        _require(res_boundary_ok.qualified is True, "18yo in MD should qualify")

        # Boundary: 17 in MD → not qualified
        text_boundary_bad = "I'm 17 in MD."
        res_boundary_bad = await qualifier_chain.ainvoke(text_boundary_bad)
        _require(isinstance(res_boundary_bad, QualifierOutputSchema), "bad schema")
        _require(
            res_boundary_bad.qualified is False,
            "17yo should not qualify",
        )

    async def main() -> None:
        """Run minimal demos for both chains to validate core behavior."""
        await demo_user_info_collection_chain()
        await demo_qualifier_chain()
        await test_zip_inference_user_info()
        await test_zip_conflict_and_boundary_in_qualifier()

    asyncio.run(main())
