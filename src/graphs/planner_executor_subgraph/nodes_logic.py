"""Nodes logic.

uv run -m src.graphs.planner_executor_subgraph.nodes_logic

"""

# %%

from typing import Literal

from langgraph.graph import END
from langgraph.types import Command

from src.graphs.planner_executor_subgraph.chains import chain
from src.graphs.planner_executor_subgraph.config import (
    available_subgraphs_for_planner_executor_subgraph,
)
from src.graphs.planner_executor_subgraph.schemas import (
    Plan,
    PlannerExecutorSubgraphState,
    Step,
)


async def planner_node(
    state: PlannerExecutorSubgraphState,
) -> Command[Literal["executor_node"]]:
    """Router node."""
    generated_plan = await chain.ainvoke(state["handoff_input"])
    print(f"Generated plan: {generated_plan.model_dump_json(indent=2)}")

    return Command(goto="executor_node", update={"plan": generated_plan})


async def executor_node(
    state: PlannerExecutorSubgraphState,
) -> Command[Literal["rag_subgraph", "react_subgraph", "reasoner_subgraph", END]]:
    """Executor node."""
    is_complete = state["plan"].is_complete
    if is_complete:
        return Command(goto=END)

    current_step = None
    for step in state["plan"].steps:
        if not step.is_complete:
            current_step = step
            break

    if current_step is None:
        # No incomplete steps found, plan is complete
        return Command(goto=END)

    suggested_agent = current_step.suggested_agent

    if suggested_agent in available_subgraphs_for_planner_executor_subgraph:
        next_subgraph = suggested_agent
    else:
        print(
            f"Warning: Unknown agent '{suggested_agent}', falling back to '{available_subgraphs_for_planner_executor_subgraph[0]}'"
        )
        next_subgraph = available_subgraphs_for_planner_executor_subgraph[0]

    return Command(goto=next_subgraph, update={"current_step": current_step})


if __name__ == "__main__":
    import asyncio

    async def main():
        """Main function."""
        print("=" * 50)
        print("Testing planner_node")
        print("=" * 50)

        # Test planner_node
        state = PlannerExecutorSubgraphState(
            handoff_input="make a plan to Descripción/resumen de los proyectos de la región de antofagasta"
        )
        response = await planner_node(state)
        print(f"Planner response: {response}")

        # Test executor_node with the actual plan from planner_node
        print("\n" + "=" * 50)
        print("Testing executor_node with planner-generated plan")
        print("=" * 50)

        # Create state with the plan from planner_node
        state_with_real_plan = PlannerExecutorSubgraphState(
            handoff_input="test input", plan=response.update["plan"]
        )

        executor_response_real = await executor_node(state_with_real_plan)
        print(f"Executor response (real plan): {executor_response_real}")

        print("\n" + "=" * 50)
        print("Testing executor_node with manual test cases")
        print("=" * 50)

        # Test executor_node with incomplete plan
        test_plan = Plan(
            goal="Test goal for evaluation",
            steps=[
                Step(
                    instruction="Search for information about Antofagasta projects",
                    suggested_agent=available_subgraphs_for_planner_executor_subgraph[
                        0
                    ],
                    reasoning="Need to retrieve relevant documents",
                    result="",
                    is_complete=False,
                ),
                Step(
                    instruction="Analyze the retrieved information",
                    suggested_agent=available_subgraphs_for_planner_executor_subgraph[
                        1
                    ],
                    reasoning="Need to process and analyze the data",
                    result="",
                    is_complete=False,
                ),
            ],
        )

        state_with_plan = PlannerExecutorSubgraphState(
            handoff_input="test input", plan=test_plan
        )

        print("Testing executor_node with incomplete plan:")
        executor_response = await executor_node(state_with_plan)
        print(f"Executor response: {executor_response}")

        # Test executor_node with completed plan
        completed_plan = Plan(
            goal="Test goal for evaluation",
            steps=[
                Step(
                    instruction="Search for information",
                    suggested_agent=available_subgraphs_for_planner_executor_subgraph[
                        0
                    ],
                    reasoning="Need to retrieve documents",
                    result="Information retrieved successfully",
                    is_complete=True,
                ),
                Step(
                    instruction="Analyze the information",
                    suggested_agent=available_subgraphs_for_planner_executor_subgraph[
                        1
                    ],
                    reasoning="Need to process data",
                    result="Analysis completed",
                    is_complete=True,
                ),
            ],
        )

        state_completed = PlannerExecutorSubgraphState(
            handoff_input="test input", plan=completed_plan
        )

        print("\nTesting executor_node with completed plan:")
        executor_response_completed = await executor_node(state_completed)
        print(f"Executor response (completed): {executor_response_completed}")

        # Test executor_node with invalid agent
        print("\nTesting executor_node with invalid agent:")
        invalid_plan = Plan(
            goal="Test goal with invalid agent",
            steps=[
                Step(
                    instruction="Invalid step",
                    suggested_agent="invalid_agent",
                    reasoning="Testing error handling",
                    result="",
                    is_complete=False,
                )
            ],
        )

        state_invalid = PlannerExecutorSubgraphState(
            handoff_input="test input", plan=invalid_plan
        )

        try:
            executor_response_invalid = await executor_node(state_invalid)
            print(f"Executor response (invalid): {executor_response_invalid}")
        except ValueError as e:
            print(f"Expected error caught: {e}")

    asyncio.run(main())

# %%
