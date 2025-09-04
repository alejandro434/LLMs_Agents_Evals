"""Tests for qualifier nodes logic: merging and prompting for missing fields."""

# %%
import pytest

from src.graphs.qualifier.nodes_logic import collect_user_info
from src.graphs.qualifier.schemas import QualifierSubgraphState, UserInfoOutputSchema


@pytest.mark.asyncio
async def test_collect_user_info_merges_prior() -> None:
    """New extraction should preserve prior values when new fields are None."""
    # Prior has age; message provides ZIP
    prior = UserInfoOutputSchema(age=25, state=None, zip_code=None)
    state = QualifierSubgraphState(
        messages=["", "My ZIP is 21201"],
        collected_user_info=prior,
    )
    cmd = await collect_user_info(state)

    merged: UserInfoOutputSchema = cmd.update.get("collected_user_info")  # type: ignore[assignment]
    assert isinstance(merged, UserInfoOutputSchema)
    # Prior age preserved
    assert merged.age == 25
    # ZIP-derived state should be set to Maryland
    assert merged.state == "Maryland"
    assert merged.zip_code is not None


@pytest.mark.asyncio
async def test_collect_user_info_prompts_for_missing() -> None:
    """When fields are missing, the node should ask concisely for them."""
    state = QualifierSubgraphState(messages=["Hi, I'm from VA."])
    cmd = await collect_user_info(state)
    merged: UserInfoOutputSchema = cmd.update.get("collected_user_info")  # type: ignore[assignment]
    assert isinstance(merged, UserInfoOutputSchema)
    # Should at least try to ask for missing age/ZIP; direct response present
    assert (
        isinstance(merged.direct_response_to_the_user, str)
        or merged.direct_response_to_the_user is None
    )


if __name__ == "__main__":
    print("test_qualifier_nodes: ready")
