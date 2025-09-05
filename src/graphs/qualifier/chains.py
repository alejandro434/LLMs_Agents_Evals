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
    Ranges (inclusive) are exact and targeted to MD/VA/DC job qualification:
      - Washington D.C.: 20001–20099, 20201–20599
      - Maryland: 20600–21999
      - Virginia: 20100–20200, 22000–24699
      - California: 90000–96199
      - New York: 10000–14999
      - Texas: 75000–79999, 88500–88599
    """
    # Washington D.C. - two separate ranges
    if 20001 <= zip5 <= 20099 or 20201 <= zip5 <= 20599:
        return "District of Columbia"
    # Maryland
    if 20600 <= zip5 <= 21999:
        return "Maryland"
    # Virginia - note: 20100-20200 range excludes D.C.'s 20201
    if 20100 <= zip5 <= 20200 or 22000 <= zip5 <= 24699:
        return "Virginia"
    # California
    if 90000 <= zip5 <= 96199:
        return "California"
    # New York
    if 10000 <= zip5 <= 14999:
        return "New York"
    # Texas
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

    # Test result tracking
    test_results = {"passed": 0, "failed": 0, "errors": []}

    def _assert_with_tracking(condition: bool, message: str, test_name: str) -> None:
        """Enhanced assertion with test result tracking."""
        if not condition:
            error_msg = f"[{test_name}] {message}"
            test_results["errors"].append(error_msg)
            test_results["failed"] += 1
            print(f"✗ {error_msg}")
            raise AssertionError(error_msg)
        test_results["passed"] += 1

    async def test_user_info_collection_comprehensive() -> None:
        """Comprehensive user info collection tests with all edge cases."""
        print("\n" + "=" * 60)
        print("Testing User Info Collection")
        print("=" * 60)

        test_cases = [
            # Basic cases
            (
                "I'm 25 years old and I live in Maryland. My zip code is 21201.",
                {"age": 25, "state": "Maryland", "zip_code": "21201"},
                "MD adult with all info",
            ),
            (
                "I'm 17 and from Virginia.",
                {"age": 17, "state": "Virginia", "zip_code": None},
                "VA minor without ZIP",
            ),
            (
                "I'm 30, living in California, zip 94016.",
                {"age": 30, "state": "California", "zip_code": "94016"},
                "CA adult with ZIP",
            ),
            (
                "I live in VA, zip 22102.",
                {"age": None, "state": "Virginia", "zip_code": "22102"},
                "VA with ZIP, no age",
            ),
            # D.C. cases
            (
                "I'm 22, ZIP 20001.",
                {"age": 22, "state": "District of Columbia", "zip_code": "20001"},
                "D.C. first range",
            ),
            (
                "Age 28, zip code 20500",
                {"age": 28, "state": "District of Columbia", "zip_code": "20500"},
                "D.C. second range",
            ),
            (
                "20250 is my ZIP",
                {"age": None, "state": "District of Columbia", "zip_code": "20250"},
                "D.C. ZIP only",
            ),
            # ZIP format variations
            (
                "ZIP: 22102-1234",
                {"age": None, "state": "Virginia", "zip_code": "22102-1234"},
                "ZIP+4 format",
            ),
            (
                "My postal code is 21201",
                {"age": None, "state": "Maryland", "zip_code": "21201"},
                "Alternative phrasing",
            ),
            # Conflict resolution
            (
                "I'm from Texas but my ZIP is 20001",
                {"age": None, "state": "District of Columbia", "zip_code": "20001"},
                "TX→DC override",
            ),
            (
                "Living in New York, 45, but ZIP 21201",
                {"age": 45, "state": "Maryland", "zip_code": "21201"},
                "NY→MD override",
            ),
            # Edge cases
            (
                "I'm 18 years old",
                {"age": 18, "state": None, "zip_code": None},
                "Age only",
            ),
            (
                "District of Columbia",
                {"age": None, "state": "District of Columbia", "zip_code": None},
                "State only",
            ),
            ("99999", {"age": None, "state": None, "zip_code": "99999"}, "Unknown ZIP"),
        ]

        for text, expected, description in test_cases:
            try:
                result = await user_info_collection_chain.ainvoke(text)

                # Type validation
                _assert_with_tracking(
                    isinstance(result, UserInfoOutputSchema),
                    f"Expected UserInfoOutputSchema, got {type(result)}",
                    description,
                )

                # Field validation
                if expected["age"] is not None:
                    _assert_with_tracking(
                        result.age == expected["age"],
                        f"Age mismatch: got {result.age}, expected {expected['age']}",
                        description,
                    )

                if expected["state"] is not None:
                    _assert_with_tracking(
                        result.state == expected["state"],
                        f"State mismatch: got {result.state}, expected {expected['state']}",
                        description,
                    )

                if expected["zip_code"] is not None:
                    # Handle ZIP+4 format - just check the 5-digit part matches
                    result_zip = result.zip_code[:5] if result.zip_code else None
                    expected_zip = expected["zip_code"][:5]
                    _assert_with_tracking(
                        result_zip == expected_zip,
                        f"ZIP mismatch: got {result.zip_code}, expected {expected['zip_code']}",
                        description,
                    )

                print(f"✓ {description}")

            except AssertionError:
                pass  # Already handled in _assert_with_tracking
            except Exception as e:
                test_results["failed"] += 1
                error_msg = f"[{description}] Unexpected error: {e}"
                test_results["errors"].append(error_msg)
                print(f"✗ {error_msg}")

    async def test_qualifier_comprehensive() -> None:
        """Comprehensive qualifier tests including D.C. and all edge cases."""
        print("\n" + "=" * 60)
        print("Testing Qualifier Logic")
        print("=" * 60)

        test_cases = [
            # Qualified cases - MD/VA/DC adults
            ("I'm 25 in Maryland. Zip 21201.", True, "MD adult"),
            ("I'm 20 in VA, zip 22102.", True, "VA adult"),
            ("I'm 30 in Washington D.C., ZIP 20001.", True, "D.C. adult"),
            ("Just turned 18, Maryland", True, "MD exactly 18"),
            ("Age 50, ZIP 20500", True, "D.C. second range"),
            # Not qualified - age
            ("I'm 17 in Virginia.", False, "VA minor"),
            ("16 years old in Maryland", False, "MD minor"),
            ("My child is 10, we're in D.C.", False, "D.C. minor"),
            # Not qualified - location
            ("I'm 30 in California.", False, "CA adult"),
            ("25 years old, New York", False, "NY adult"),
            ("Age 40, Texas", False, "TX adult"),
            # ZIP override cases
            ("I live in CA, ZIP 21201. I'm 25.", True, "CA→MD override"),
            ("From Texas, 30, but ZIP 20001", True, "TX→DC override"),
            ("New York resident, 22, ZIP 22102", True, "NY→VA override"),
            # Boundary cases
            ("I'm 18 in MD, ZIP 20600", True, "MD boundary ZIP"),
            ("Age 18, Virginia, 24699", True, "VA boundary ZIP"),
            ("18 years old, 20099", True, "D.C. boundary ZIP"),
            # Edge cases
            ("I'm 17 in CA", False, "Minor + wrong state"),
            ("30 years old, ZIP 99999", False, "Unknown ZIP adult"),
        ]

        for text, expected_qualified, description in test_cases:
            try:
                result = await qualifier_chain.ainvoke(text)

                # Type validation
                _assert_with_tracking(
                    isinstance(result, QualifierOutputSchema),
                    f"Expected QualifierOutputSchema, got {type(result)}",
                    description,
                )

                # Qualification validation
                _assert_with_tracking(
                    isinstance(result.qualified, bool),
                    f"qualified must be bool, got {type(result.qualified)}",
                    description,
                )

                _assert_with_tracking(
                    result.qualified == expected_qualified,
                    f"Qualification mismatch: got {result.qualified}, expected {expected_qualified}",
                    description,
                )

                # Reason validation for not qualified
                if not result.qualified:
                    _assert_with_tracking(
                        result.why_not_qualified is not None
                        and len(result.why_not_qualified) > 0,
                        "Missing or empty why_not_qualified for non-qualified result",
                        description,
                    )
                else:
                    _assert_with_tracking(
                        result.why_not_qualified is None,
                        f"why_not_qualified should be None for qualified, got: {result.why_not_qualified}",
                        description,
                    )

                print(f"✓ {description}: qualified={result.qualified}")

            except AssertionError:
                pass  # Already handled in _assert_with_tracking
            except Exception as e:
                test_results["failed"] += 1
                error_msg = f"[{description}] Unexpected error: {e}"
                test_results["errors"].append(error_msg)
                print(f"✗ {error_msg}")

    async def test_zip_inference_comprehensive() -> None:
        """Test ZIP code to state inference for all supported states."""
        print("\n" + "=" * 60)
        print("Testing ZIP to State Inference")
        print("=" * 60)

        samples = [
            # D.C. ranges
            ("20001", "District of Columbia", "D.C. first range start"),
            ("20050", "District of Columbia", "D.C. first range middle"),
            ("20099", "District of Columbia", "D.C. first range end"),
            ("20201", "District of Columbia", "D.C. second range start"),
            ("20400", "District of Columbia", "D.C. second range middle"),
            ("20599", "District of Columbia", "D.C. second range end"),
            # Maryland
            ("20600", "Maryland", "MD start"),
            ("21000", "Maryland", "MD middle"),
            ("21999", "Maryland", "MD end"),
            # Virginia
            ("20100", "Virginia", "VA north start"),
            ("20150", "Virginia", "VA north middle"),
            ("20200", "Virginia", "VA north end"),
            ("22000", "Virginia", "VA south start"),
            ("23000", "Virginia", "VA south middle"),
            ("24699", "Virginia", "VA south end"),
            # Other states
            ("90210", "California", "CA Beverly Hills"),
            ("10001", "New York", "NY Manhattan"),
            ("75001", "Texas", "TX Dallas"),
            # Unknown
            ("99999", None, "Unknown ZIP"),
            ("00000", None, "Invalid ZIP"),
        ]

        for zip_code, expected_state, description in samples:
            try:
                text = f"My ZIP is {zip_code}"
                result = await user_info_collection_chain.ainvoke(text)

                _assert_with_tracking(
                    isinstance(result, UserInfoOutputSchema),
                    f"Expected UserInfoOutputSchema, got {type(result)}",
                    description,
                )

                _assert_with_tracking(
                    result.zip_code == zip_code,
                    f"ZIP not preserved: got {result.zip_code}, expected {zip_code}",
                    description,
                )

                if expected_state:
                    _assert_with_tracking(
                        result.state == expected_state,
                        f"State inference wrong: got {result.state}, expected {expected_state}",
                        description,
                    )
                else:
                    _assert_with_tracking(
                        result.state is None,
                        f"Should not infer state for unknown ZIP, got {result.state}",
                        description,
                    )

                print(f"✓ {description}: {zip_code} → {result.state}")

            except AssertionError:
                pass  # Already handled in _assert_with_tracking
            except Exception as e:
                test_results["failed"] += 1
                error_msg = f"[{description}] Unexpected error: {e}"
                test_results["errors"].append(error_msg)
                print(f"✗ {error_msg}")

    async def test_helper_functions() -> None:
        """Test the helper functions directly."""
        print("\n" + "=" * 60)
        print("Testing Helper Functions")
        print("=" * 60)

        # Test _extract_zip5
        zip_tests = [
            ("21201", 21201, "Basic 5-digit"),
            ("21201-1234", 21201, "ZIP+4"),
            ("ZIP: 21201", 21201, "With prefix"),
            ("My zip is 21201 thanks", 21201, "In sentence"),
            ("123", None, "Too short"),
            ("abcde", None, "No digits"),
            (None, None, "None input"),
            ("", None, "Empty string"),
        ]

        for input_val, expected, description in zip_tests:
            result = _extract_zip5(input_val)
            try:
                _assert_with_tracking(
                    result == expected,
                    f"_extract_zip5({input_val!r}) = {result}, expected {expected}",
                    f"extract_zip5: {description}",
                )
                print(f"✓ _extract_zip5: {description}")
            except AssertionError:
                pass

        # Test _infer_state_from_zip
        state_tests = [
            (20001, "District of Columbia", "D.C. first range"),
            (20500, "District of Columbia", "D.C. second range"),
            (20600, "Maryland", "MD start"),
            (21999, "Maryland", "MD end"),
            (20100, "Virginia", "VA north"),
            (24699, "Virginia", "VA south end"),
            (90210, "California", "CA"),
            (10001, "New York", "NY"),
            (75001, "Texas", "TX"),
            (99999, None, "Unknown"),
            (20000, None, "Before D.C."),
            (20200, "Virginia", "VA 20200"),
            (20201, "District of Columbia", "D.C. 20201"),
        ]

        for zip5, expected, description in state_tests:
            result = _infer_state_from_zip(zip5)
            try:
                _assert_with_tracking(
                    result == expected,
                    f"_infer_state_from_zip({zip5}) = {result}, expected {expected}",
                    f"infer_state: {description}",
                )
                print(f"✓ _infer_state_from_zip: {description}")
            except AssertionError:
                pass

    async def main() -> None:
        """Run all comprehensive tests and report results."""
        print("\n" + "=" * 60)
        print("QUALIFIER CHAINS COMPREHENSIVE TEST SUITE")
        print("=" * 60)

        # Run all test suites
        await test_helper_functions()
        await test_user_info_collection_comprehensive()
        await test_qualifier_comprehensive()
        await test_zip_inference_comprehensive()

        # Report results
        print("\n" + "=" * 60)
        print("TEST RESULTS SUMMARY")
        print("=" * 60)
        print(f"✓ Passed: {test_results['passed']}")
        print(f"✗ Failed: {test_results['failed']}")
        print(f"Total: {test_results['passed'] + test_results['failed']}")

        if test_results["errors"]:
            print("\nErrors encountered:")
            for error in test_results["errors"][:10]:  # Show first 10 errors
                print(f"  - {error}")
            if len(test_results["errors"]) > 10:
                print(f"  ... and {len(test_results['errors']) - 10} more")

        if test_results["failed"] == 0:
            print("\n🎉 All tests passed successfully!")
        else:
            print(f"\n⚠️  {test_results['failed']} tests failed. Review errors above.")
            exit(1)

    asyncio.run(main())
