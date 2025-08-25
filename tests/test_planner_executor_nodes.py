"""Comprehensive test suite for planner_executor_subgraph nodes.

uv run -m pytest tests/test_planner_executor_nodes.py -v
"""

# %%
import asyncio
from unittest.mock import patch

import pytest
from langgraph.graph import END
from langgraph.types import Command

from src.graphs.planner_executor_subgraph.config import (
    available_subgraphs_for_planner_executor_subgraph,
)
from src.graphs.planner_executor_subgraph.nodes_logic import (
    executor_node,
    planner_node,
)
from src.graphs.planner_executor_subgraph.schemas import (
    Plan,
    PlannerExecutorSubgraphState,
    Step,
)


class TestPlannerNode:
    """Test suite for planner_node function."""

    @pytest.mark.asyncio
    async def test_planner_node_basic_functionality(self):
        """Test that planner_node generates a plan and returns correct command."""
        # Create mock plan
        mock_plan = Plan(
            goal="Test goal",
            steps=[
                Step(
                    instruction="Step 1",
                    suggested_agent="rag_subgraph",
                    reasoning="Test reasoning",
                    result="",
                    is_complete=False,
                )
            ],
        )

        # Mock the chain.ainvoke to return our mock plan
        with patch(
            "src.graphs.planner_executor_subgraph.nodes_logic.chain.ainvoke"
        ) as mock_chain:
            mock_chain.return_value = mock_plan

            state = PlannerExecutorSubgraphState(handoff_input="Test input")
            result = await planner_node(state)

            # Verify the chain was called with correct input
            mock_chain.assert_called_once_with("Test input")

            # Verify the command structure
            assert isinstance(result, Command)
            assert result.goto == "executor_node"
            assert "plan" in result.update
            assert result.update["plan"] == mock_plan

    @pytest.mark.asyncio
    async def test_planner_node_with_complex_input(self):
        """Test planner_node with complex multi-step plan."""
        complex_plan = Plan(
            goal="Complex multi-step goal",
            steps=[
                Step(
                    instruction=f"Step {i}",
                    suggested_agent=available_subgraphs_for_planner_executor_subgraph[
                        i % 3
                    ],
                    reasoning=f"Reasoning for step {i}",
                    result="",
                    is_complete=False,
                )
                for i in range(5)
            ],
        )

        with patch(
            "src.graphs.planner_executor_subgraph.nodes_logic.chain.ainvoke"
        ) as mock_chain:
            mock_chain.return_value = complex_plan

            state = PlannerExecutorSubgraphState(
                handoff_input="Complex task requiring multiple steps"
            )
            result = await planner_node(state)

            assert result.update["plan"] == complex_plan
            assert len(result.update["plan"].steps) == 5

    @pytest.mark.asyncio
    async def test_planner_node_error_handling(self):
        """Test planner_node handles errors gracefully."""
        with patch(
            "src.graphs.planner_executor_subgraph.nodes_logic.chain.ainvoke"
        ) as mock_chain:
            mock_chain.side_effect = Exception("Chain invocation failed")

            state = PlannerExecutorSubgraphState(handoff_input="Test input")

            with pytest.raises(Exception, match="Chain invocation failed"):
                await planner_node(state)


class TestExecutorNode:
    """Test suite for executor_node function."""

    @pytest.mark.asyncio
    async def test_executor_with_incomplete_plan(self):
        """Test executor_node with plan containing incomplete steps."""
        plan = Plan(
            goal="Test goal",
            steps=[
                Step(
                    instruction="Completed step",
                    suggested_agent="rag_subgraph",
                    reasoning="Already done",
                    result="Done",
                    is_complete=True,
                ),
                Step(
                    instruction="Incomplete step",
                    suggested_agent="react_subgraph",
                    reasoning="Need to do this",
                    result="",
                    is_complete=False,
                ),
                Step(
                    instruction="Another incomplete",
                    suggested_agent="reasoner_subgraph",
                    reasoning="Also need this",
                    result="",
                    is_complete=False,
                ),
            ],
        )

        state = PlannerExecutorSubgraphState(handoff_input="test", plan=plan)
        result = await executor_node(state)

        # Should go to the first incomplete step's agent
        assert result.goto == "react_subgraph"
        assert result.update["current_step"] == plan.steps[1]

    @pytest.mark.asyncio
    async def test_executor_with_completed_plan(self):
        """Test executor_node returns END when all steps are complete."""
        plan = Plan(
            goal="Completed goal",
            steps=[
                Step(
                    instruction="Step 1",
                    suggested_agent="rag_subgraph",
                    reasoning="Done",
                    result="Result 1",
                    is_complete=True,
                ),
                Step(
                    instruction="Step 2",
                    suggested_agent="react_subgraph",
                    reasoning="Also done",
                    result="Result 2",
                    is_complete=True,
                ),
            ],
        )

        state = PlannerExecutorSubgraphState(handoff_input="test", plan=plan)
        result = await executor_node(state)

        assert result.goto == END

    @pytest.mark.asyncio
    async def test_executor_with_invalid_agent_fallback(self):
        """Test executor_node falls back to first available agent for invalid agent."""
        plan = Plan(
            goal="Test with invalid agent",
            steps=[
                Step(
                    instruction="Invalid agent step",
                    suggested_agent="non_existent_agent",
                    reasoning="Test fallback",
                    result="",
                    is_complete=False,
                )
            ],
        )

        state = PlannerExecutorSubgraphState(handoff_input="test", plan=plan)

        # Capture print output to verify warning
        with patch("builtins.print") as mock_print:
            result = await executor_node(state)

            # Should fall back to first available subgraph
            assert result.goto == available_subgraphs_for_planner_executor_subgraph[0]
            assert result.update["current_step"] == plan.steps[0]

            # Verify warning was printed
            mock_print.assert_called()
            warning_call = str(mock_print.call_args_list[0])
            assert "Unknown agent" in warning_call
            assert "non_existent_agent" in warning_call

    @pytest.mark.asyncio
    async def test_executor_with_all_valid_agents(self):
        """Test executor_node correctly routes to each valid agent."""
        for agent in available_subgraphs_for_planner_executor_subgraph:
            plan = Plan(
                goal=f"Test with {agent}",
                steps=[
                    Step(
                        instruction=f"Use {agent}",
                        suggested_agent=agent,
                        reasoning=f"Testing {agent}",
                        result="",
                        is_complete=False,
                    )
                ],
            )

            state = PlannerExecutorSubgraphState(handoff_input="test", plan=plan)
            result = await executor_node(state)

            assert result.goto == agent
            assert result.update["current_step"].suggested_agent == agent

    @pytest.mark.asyncio
    async def test_executor_edge_case_empty_plan(self):
        """Test executor_node with empty plan."""
        plan = Plan(goal="Empty plan", steps=[])

        state = PlannerExecutorSubgraphState(handoff_input="test", plan=plan)
        result = await executor_node(state)

        # Empty plan should be considered complete
        assert result.goto == END

    @pytest.mark.asyncio
    async def test_executor_sequential_step_processing(self):
        """Test that executor processes steps in order."""
        steps = [
            Step(
                instruction=f"Step {i}",
                suggested_agent=available_subgraphs_for_planner_executor_subgraph[
                    i % 3
                ],
                reasoning=f"Reasoning {i}",
                result="" if i >= 2 else f"Result {i}",
                is_complete=i < 2,  # First 2 steps complete
            )
            for i in range(5)
        ]

        plan = Plan(goal="Sequential processing", steps=steps)
        state = PlannerExecutorSubgraphState(handoff_input="test", plan=plan)
        result = await executor_node(state)

        # Should pick the third step (index 2) as first incomplete
        assert result.update["current_step"] == steps[2]
        assert result.goto == available_subgraphs_for_planner_executor_subgraph[2 % 3]


class TestIntegration:
    """Integration tests for planner and executor nodes working together."""

    @pytest.mark.asyncio
    async def test_planner_to_executor_flow(self):
        """Test the complete flow from planner to executor."""
        mock_plan = Plan(
            goal="Integration test goal",
            steps=[
                Step(
                    instruction="First step",
                    suggested_agent="rag_subgraph",
                    reasoning="Start here",
                    result="",
                    is_complete=False,
                ),
                Step(
                    instruction="Second step",
                    suggested_agent="react_subgraph",
                    reasoning="Then this",
                    result="",
                    is_complete=False,
                ),
            ],
        )

        with patch(
            "src.graphs.planner_executor_subgraph.nodes_logic.chain.ainvoke"
        ) as mock_chain:
            mock_chain.return_value = mock_plan

            # Step 1: Run planner
            state = PlannerExecutorSubgraphState(
                handoff_input="Create integration test plan"
            )
            planner_result = await planner_node(state)

            assert planner_result.goto == "executor_node"
            generated_plan = planner_result.update["plan"]

            # Step 2: Use planner output in executor
            state_with_plan = PlannerExecutorSubgraphState(
                handoff_input="test", plan=generated_plan
            )
            executor_result = await executor_node(state_with_plan)

            # Verify executor picks first step
            assert executor_result.goto == "rag_subgraph"
            assert executor_result.update["current_step"] == mock_plan.steps[0]

    @pytest.mark.asyncio
    async def test_state_transitions(self):
        """Test various state transitions through the nodes."""
        transitions = []

        # Initial incomplete plan
        plan = Plan(
            goal="State transition test",
            steps=[
                Step(
                    instruction="Step 1",
                    suggested_agent="rag_subgraph",
                    reasoning="First",
                    result="",
                    is_complete=False,
                ),
                Step(
                    instruction="Step 2",
                    suggested_agent="react_subgraph",
                    reasoning="Second",
                    result="",
                    is_complete=False,
                ),
            ],
        )

        # First execution - should go to first step
        state = PlannerExecutorSubgraphState(handoff_input="test", plan=plan)
        result = await executor_node(state)
        transitions.append(("start", result.goto))
        assert result.goto == "rag_subgraph"

        # Mark first step complete
        plan.steps[0].is_complete = True
        plan.steps[0].result = "Step 1 completed"

        # Second execution - should go to second step
        state = PlannerExecutorSubgraphState(handoff_input="test", plan=plan)
        result = await executor_node(state)
        transitions.append((transitions[-1][1], result.goto))
        assert result.goto == "react_subgraph"

        # Mark second step complete
        plan.steps[1].is_complete = True
        plan.steps[1].result = "Step 2 completed"

        # Third execution - should end
        state = PlannerExecutorSubgraphState(handoff_input="test", plan=plan)
        result = await executor_node(state)
        transitions.append((transitions[-1][1], result.goto))
        assert result.goto == END

        # Verify complete transition path
        expected_transitions = [
            ("start", "rag_subgraph"),
            ("rag_subgraph", "react_subgraph"),
            ("react_subgraph", END),
        ]
        assert transitions == expected_transitions


class TestRobustness:
    """Tests for robustness and edge cases."""

    @pytest.mark.asyncio
    async def test_concurrent_execution(self):
        """Test nodes can handle concurrent executions."""
        mock_plan = Plan(
            goal="Concurrent test",
            steps=[
                Step(
                    instruction="Concurrent step",
                    suggested_agent="rag_subgraph",
                    reasoning="Test",
                    result="",
                    is_complete=False,
                )
            ],
        )

        with patch(
            "src.graphs.planner_executor_subgraph.nodes_logic.chain.ainvoke"
        ) as mock_chain:
            mock_chain.return_value = mock_plan

            # Create multiple concurrent tasks
            tasks = []
            for i in range(10):
                state = PlannerExecutorSubgraphState(
                    handoff_input=f"Concurrent test {i}"
                )
                tasks.append(planner_node(state))

            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks)

            # All should succeed
            assert len(results) == 10
            for result in results:
                assert result.goto == "executor_node"

    @pytest.mark.asyncio
    async def test_malformed_plan_handling(self):
        """Test handling of malformed or unexpected plan structures."""
        # Test with None steps
        with pytest.raises(AttributeError):
            plan = Plan(goal="Malformed", steps=None)

        # Test with plan missing required fields
        with pytest.raises(Exception):
            plan = Plan()  # Missing required 'goal' and 'steps'

    @pytest.mark.asyncio
    async def test_very_long_plan(self):
        """Test handling of very long plans with many steps."""
        # Create a plan with 100 steps
        steps = [
            Step(
                instruction=f"Step {i}",
                suggested_agent=available_subgraphs_for_planner_executor_subgraph[
                    i % 3
                ],
                reasoning=f"Reasoning {i}",
                result="" if i == 0 else f"Result {i}",
                is_complete=i > 0,  # Only first step incomplete
            )
            for i in range(100)
        ]

        plan = Plan(goal="Very long plan", steps=steps)
        state = PlannerExecutorSubgraphState(handoff_input="test", plan=plan)
        result = await executor_node(state)

        # Should still correctly find the first incomplete step
        assert result.update["current_step"] == steps[0]
        assert result.goto == "rag_subgraph"


if __name__ == "__main__":
    # Run basic smoke test

    print("Running smoke tests...")

    async def smoke_test():
        """Quick smoke test."""
        # Test planner
        with patch(
            "src.graphs.planner_executor_subgraph.nodes_logic.chain.ainvoke"
        ) as mock:
            mock.return_value = Plan(
                goal="Smoke test",
                steps=[
                    Step(
                        instruction="Test",
                        suggested_agent="rag_subgraph",
                        reasoning="Test",
                        result="",
                        is_complete=False,
                    )
                ],
            )

            state = PlannerExecutorSubgraphState(handoff_input="smoke test")
            result = await planner_node(state)
            assert result.goto == "executor_node"
            print("✓ Planner node smoke test passed")

        # Test executor
        plan = Plan(
            goal="Test",
            steps=[
                Step(
                    instruction="Test",
                    suggested_agent="rag_subgraph",
                    reasoning="Test",
                    result="",
                    is_complete=False,
                )
            ],
        )
        state = PlannerExecutorSubgraphState(handoff_input="test", plan=plan)
        result = await executor_node(state)
        assert result.goto == "rag_subgraph"
        print("✓ Executor node smoke test passed")

        print(
            "\nAll smoke tests passed! Run 'uv run -m pytest tests/test_planner_executor_nodes.py -v' for full test suite."
        )

    asyncio.run(smoke_test())
