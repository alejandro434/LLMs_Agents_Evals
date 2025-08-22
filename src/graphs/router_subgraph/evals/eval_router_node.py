"""Evaluate the router node.

uv run -m src.graphs.router_subgraph.evals.eval_router_node
"""

# %%

import asyncio
from pathlib import Path

import yaml
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_correctness

from src.graphs.router_subgraph.chains import chain
from src.graphs.router_subgraph.nodes_logic import router_node
from src.graphs.router_subgraph.schemas import RouterSubgraphState


async def main() -> None:
    """Run router node over all inputs and compute answer correctness.

    This script does two things:
    - Calls the router node to show its control flow decision per input
    - Uses the structured chain output to build a dataset and evaluates
      answer correctness against the fewshots ground truth
    """
    yaml_path = Path("src/graphs/router_subgraph/fewshots.yml")
    with yaml_path.open(encoding="utf-8") as f:
        fewshots = yaml.safe_load(f)

    examples = fewshots.get("Routing_examples", [])

    questions: list[str] = []
    answers: list[str] = []
    ground_truths: list[str] = []

    for example in examples:
        user_text = example.get("input", "")
        expected_output = example.get("output", {})
        expected_text: str = (
            expected_output.get("direct_response_to_the_user")
            or expected_output.get("next_subgraph")
            or ""
        )

        # Show router control decision (Command)
        print(f"Input: {user_text!r}")
        state = RouterSubgraphState(user_input=user_text)
        control = await router_node(state)
        print(control)
        print("---")

        # Prefer extracting prediction from the router Command
        # If the router decided to hand off, use the subgraph name (goto)
        # Otherwise, fetch the direct response from the structured output
        predicted_text: str
        if getattr(control, "goto", None):
            predicted_text = str(control.goto)
        else:
            structured = await chain.ainvoke(user_text)
            predicted_text = (
                structured.direct_response_to_the_user or structured.next_subgraph or ""
            )

        questions.append(user_text)
        answers.append(predicted_text)
        ground_truths.append(expected_text)

    dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "ground_truth": ground_truths,
        }
    )

    result = evaluate(dataset, metrics=[answer_correctness])
    df = result.to_pandas()

    print("Ragas answer correctness results:")
    # Avoid very wide prints; rely on pandas default formatting
    print(df)


if __name__ == "__main__":
    asyncio.run(main())
