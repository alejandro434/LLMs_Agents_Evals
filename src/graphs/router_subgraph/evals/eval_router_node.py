"""Evaluate the router node.

uv run -m src.graphs.router_subgraph.evals.eval_router_node
"""

# %%

import asyncio
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import yaml
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_correctness
from sklearn.metrics import average_precision_score, precision_recall_curve

from src.graphs.llm_chains_factory.assembling import build_structured_chain
from src.graphs.router_subgraph.chains import SYSTEM_PROMPT, chain
from src.graphs.router_subgraph.nodes_logic import router_node
from src.graphs.router_subgraph.schemas import RouterOutputSchema, RouterSubgraphState


def _compute_classification_metrics(
    *,
    y_true: list[str],
    y_pred: list[str],
    classes: list[str],
) -> tuple[float, list[dict[str, float | int | None]]]:
    """Compute accuracy and per-class precision/recall/F1.

    Returns a tuple of (accuracy, per_class_rows).
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must be same length, got {len(y_true)} and {len(y_pred)}"
        )
    total = len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred, strict=False) if t == p)
    accuracy = correct / total if total else 0.0

    rows: list[dict[str, float | int | None]] = []
    for cls in classes:
        tp = sum(
            1 for t, p in zip(y_true, y_pred, strict=False) if t == cls and p == cls
        )
        fp = sum(
            1 for t, p in zip(y_true, y_pred, strict=False) if t != cls and p == cls
        )
        fn = sum(
            1 for t, p in zip(y_true, y_pred, strict=False) if t == cls and p != cls
        )

        precision = tp / (tp + fp) if (tp + fp) > 0 else None
        recall = tp / (tp + fn) if (tp + fn) > 0 else None
        f1: float | None
        if precision is None or recall is None or (precision + recall) == 0:
            f1 = None
        else:
            f1 = 2 * precision * recall / (precision + recall)

        rows.append(
            {
                "class": cls,
                "support": sum(1 for t in y_true if t == cls),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return accuracy, rows


async def main() -> None:
    """Run router node evaluation.

    - Prints router control decision per input
    - Computes routing classification metrics for subgraph selections
    - Computes Ragas answer correctness for direct responses only
    """
    yaml_path = Path("src/graphs/router_subgraph/fewshots.yml")
    with yaml_path.open(encoding="utf-8") as f:
        fewshots = yaml.safe_load(f)

    examples = fewshots.get("Routing_examples", [])

    # For Ragas (only direct-response tasks)
    questions: list[str] = []
    answers: list[str] = []
    ground_truths: list[str] = []

    # For routing classification metrics
    y_true: list[str] = []
    y_pred: list[str] = []

    for example in examples:
        user_text = example.get("input", "")
        expected_output = example.get("output", {})
        expected_direct = expected_output.get("direct_response_to_the_user")
        expected_next = expected_output.get("next_subgraph")

        # Show router control decision (Command)
        print(f"Input: {user_text!r}")
        state = RouterSubgraphState(user_input=user_text)
        control = await router_node(state)
        print(control)
        print("---")

        # Classification target if expected_next is present
        if expected_next is not None:
            predicted_label = (
                str(control.goto) if getattr(control, "goto", None) else "NONE"
            )
            y_true.append(str(expected_next))
            y_pred.append(predicted_label)

        # Ragas only for direct-response tasks
        if expected_next is None and expected_direct is not None:
            structured = await chain.ainvoke(user_text)
            predicted_direct = (
                structured.direct_response_to_the_user or structured.next_subgraph or ""
            )
            questions.append(user_text)
            answers.append(predicted_direct)
            ground_truths.append(expected_direct)

    # --- Classification metrics for routing decisions ---
    # Load classes from handoff-subgraphs.yml
    with Path("src/graphs/router_subgraph/handoff-subgraphs.yml").open(
        encoding="utf-8"
    ) as f:
        subgraphs = yaml.safe_load(f)["handoff_subgraphs"]
    classes = [sg["name"] for sg in subgraphs]

    if y_true:
        accuracy, per_class = _compute_classification_metrics(
            y_true=y_true, y_pred=y_pred, classes=classes
        )
        print("Router classification metrics (subgraphs):")
        print(f"  Samples: {len(y_true)}  Accuracy: {accuracy:.4f}")
        for row in per_class:
            print(
                "  {class}: support={support} tp={tp} fp={fp} fn={fn} "
                "precision={precision} recall={recall} f1={f1}".format(
                    **{
                        "class": row["class"],
                        "support": row["support"],
                        "tp": row["tp"],
                        "fp": row["fp"],
                        "fn": row["fn"],
                        "precision": (
                            f"{row['precision']:.4f}"
                            if row["precision"] is not None
                            else "None"
                        ),
                        "recall": (
                            f"{row['recall']:.4f}"
                            if row["recall"] is not None
                            else "None"
                        ),
                        "f1": (f"{row['f1']:.4f}" if row["f1"] is not None else "None"),
                    }
                )
            )
        print("---")

        # --- Precision-Recall curves via sampling-based class scores ---
        # Build a higher-temperature chain for diversity
        n_samples = int(os.getenv("ROUTER_PR_SAMPLES", "80"))
        sample_temperature = float(os.getenv("ROUTER_PR_TEMPERATURE", "0.9"))
        jitter_eps = float(os.getenv("ROUTER_PR_JITTER", "1e-06"))
        random.seed(42)
        sampling_chain = build_structured_chain(
            system_prompt=SYSTEM_PROMPT,
            output_schema=RouterOutputSchema,
            k=5,
            temperature=sample_temperature,
            postprocess=None,
            group="Routing_examples",
            yaml_path=Path("src/graphs/router_subgraph/fewshots.yml"),
        )

        # Collect texts corresponding to routing (exclude direct-response)
        routing_texts: list[str] = []
        routing_true: list[str] = []
        for example in examples:
            expected_output = example.get("output", {})
            if expected_output.get("next_subgraph") is not None:
                routing_texts.append(example.get("input", ""))
                routing_true.append(str(expected_output.get("next_subgraph")))

        # For each routing example, estimate scores per class
        class_to_scores: dict[str, list[float]] = {cls: [] for cls in classes}

        for text in routing_texts:
            # Run samples asynchronously
            tasks = [sampling_chain.ainvoke(text) for _ in range(n_samples)]
            results = await asyncio.gather(*tasks)
            counts: dict[str, int] = dict.fromkeys(classes, 0)
            for r in results:
                if r.next_subgraph in classes:
                    counts[str(r.next_subgraph)] += 1
            for cls in classes:
                score = counts[cls] / float(n_samples)
                if jitter_eps > 0:
                    score = min(1.0, max(0.0, score + random.uniform(0, jitter_eps)))
                class_to_scores[cls].append(score)

        # Plot per-class PR curves
        plt.figure(figsize=(9, 6))
        all_true_bins: list[int] = []
        all_scores: list[float] = []
        for cls in classes:
            y_true_bin = [1 if t == cls else 0 for t in routing_true]
            y_scores = class_to_scores[cls]
            if sum(y_true_bin) == 0:
                continue
            precision, recall, _ = precision_recall_curve(y_true_bin, y_scores)
            ap = average_precision_score(y_true_bin, y_scores)
            plt.step(recall, precision, where="post", label=f"{cls} (AP={ap:.2f})")
            all_true_bins.extend(y_true_bin)
            all_scores.extend(y_scores)

        # Micro-averaged PR curve across classes
        if all_true_bins:
            m_precision, m_recall, _ = precision_recall_curve(all_true_bins, all_scores)
            m_ap = average_precision_score(all_true_bins, all_scores)
            plt.step(
                m_recall,
                m_precision,
                where="post",
                linewidth=2.0,
                color="black",
                label=f"micro-average (AP={m_ap:.2f})",
            )

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Router Subgraph Precision-Recall Curves")
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        out_path = Path("router_pr_curve.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        print(f"Saved precision-recall curves to {out_path.resolve()}")

    # --- Ragas for direct responses ---
    if questions:
        dataset = Dataset.from_dict(
            {
                "question": questions,
                "answer": answers,
                "ground_truth": ground_truths,
            }
        )

        result = evaluate(dataset, metrics=[answer_correctness])
        df = result.to_pandas()
        print("Ragas answer correctness results (direct responses only):")
        print(df)

    # If there were no direct-response tasks, make it clear
    if not questions:
        print("No direct-response tasks to score with Ragas.")


if __name__ == "__main__":
    asyncio.run(main())
