"""Unit tests for qualifier helper functions and postprocess logic."""

# %%

from src.graphs.qualifier.chains import (
    _enforce_state_from_zip_user_info,
    _extract_zip5,
    _infer_state_from_zip,
)
from src.graphs.qualifier.schemas import UserInfoOutputSchema


def test_extract_zip5_variants() -> None:
    """ZIP extraction handles normal, ZIP+4, noisy, and invalid cases."""
    assert _extract_zip5("22102") == 22102
    assert _extract_zip5("22102-1234") == 22102
    assert _extract_zip5("ZIP: 21201") == 21201
    assert _extract_zip5("2120") is None
    assert _extract_zip5("") is None
    assert _extract_zip5(None) is None


def test_infer_state_boundaries() -> None:
    """State mapping covers targeted ranges and boundaries."""
    # Maryland
    assert _infer_state_from_zip(20600) == "Maryland"
    assert _infer_state_from_zip(21999) == "Maryland"
    # Virginia
    assert _infer_state_from_zip(20100) == "Virginia"
    assert _infer_state_from_zip(20199) == "Virginia"
    assert _infer_state_from_zip(22000) == "Virginia"
    assert _infer_state_from_zip(24699) == "Virginia"
    # California
    assert _infer_state_from_zip(90000) == "California"
    assert _infer_state_from_zip(96199) == "California"
    # New York
    assert _infer_state_from_zip(10000) == "New York"
    assert _infer_state_from_zip(14999) == "New York"
    # Texas
    assert _infer_state_from_zip(75000) == "Texas"
    assert _infer_state_from_zip(79999) == "Texas"
    assert _infer_state_from_zip(88500) == "Texas"
    # Unknown
    assert _infer_state_from_zip(20599) is None


def test_enforce_state_from_zip_user_info_sets_state() -> None:
    """ZIP-derived state should overwrite conflicting textual state."""
    ui = UserInfoOutputSchema(age=30, state="California", zip_code="21201-1234")
    fixed = _enforce_state_from_zip_user_info(ui)
    assert fixed.state == "Maryland"


def test_enforce_state_from_zip_user_info_no_zip() -> None:
    """No ZIP leads to no change in state."""
    ui = UserInfoOutputSchema(age=30, state="California", zip_code=None)
    fixed = _enforce_state_from_zip_user_info(ui)
    assert fixed.state == "California"


def test_enforce_state_from_zip_user_info_unknown_zip() -> None:
    """Unknown ZIP leaves state unchanged."""
    ui = UserInfoOutputSchema(age=30, state="California", zip_code="20599")
    fixed = _enforce_state_from_zip_user_info(ui)
    assert fixed.state == "California"


if __name__ == "__main__":
    # Simple demonstration / test
    print("test_qualifier_helpers: ready")
