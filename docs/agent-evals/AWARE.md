# Agentic Workflow Analysis and Reliability Evaluation (AWARE)

## Table of Contents

1. [Introduction](#introduction)
2. [Multi-Agent System Architecture Overview](#multi-agent-system-architecture-overview)
3. [RAGAS Evaluation Framework for Multi-Agent Systems](#ragas-evaluation-framework-for-multi-agent-systems)
4. [Three-Tier Evaluation Strategy](#three-tier-evaluation-strategy)
5. [Agent-Specific Evaluation Methodologies](#agent-specific-evaluation-methodologies)
6. [Custom RAGAS Metrics for Multi-Agent Systems](#custom-ragas-metrics-for-multi-agent-systems)
7. [Implementation Guidelines](#implementation-guidelines)
8. [Operational Considerations](#operational-considerations)
9. [Conclusion](#conclusion)

---

## Introduction

The evaluation of LLM-based multi-agent systems presents unique challenges that extend beyond traditional single-model assessment. Unlike monolithic AI applications, multi-agent workflows involve complex interactions between specialized agents, each with distinct capabilities and failure modes. This document presents **AWARE** (Agentic Workflow Analysis and Reliability Evaluation), a comprehensive evaluation strategy specifically designed for LangGraph-based multi-agent systems.

Building upon the RAGAS (Retrieval-Augmented Generation Assessment) framework, AWARE addresses the critical need for systematic, quantifiable, and actionable evaluation across three complementary dimensions:

- **Unit-level evaluation**: Individual agent performance assessment in isolation
- **Integration-level evaluation**: Agent interaction and handoff quality measurement
- **System-level evaluation**: End-to-end workflow effectiveness verification

This methodology transforms multi-agent evaluation from subjective assessments to a data-driven engineering discipline, enabling developers to iterate with precision and confidence.


## Multi-Agent System Architecture Overview

Our multi-agent system comprises five specialized LangGraph-based agents:

### 1. Router Agent (`router_subgraph`)
**Purpose**: Entry point for user queries, determining optimal routing decisions

- **Core Functionality**: Classifies incoming queries and routes to appropriate subgraphs
- **Decision Types**: Direct response vs. subgraph handoff
- **Output Schema**: `RouterOutputSchema` with `direct_response_to_the_user` or `next_subgraph`
- **Key Challenge**: Accurate classification across diverse query types

**Example Routing Decisions**:
```yaml
- Input: "Hi there!"
  Output: {direct_response_to_the_user: "Hello! How can I help you today?", next_subgraph: null}
- Input: "what is the meaning of life?"
  Output: {direct_response_to_the_user: null, next_subgraph: "reasoner_subgraph"}
- Input: "what is the weather in SF and how is the traffic?"
  Output: {direct_response_to_the_user: null, next_subgraph: "ReAct_subgraph"}
```

### 2. Reasoner Agent (`reasoner_subgraph`)
**Purpose**: Deep analytical thinking for complex philosophical and abstract questions

- **Use Cases**: "What is the meaning of life?", "Is free will an illusion?"
- **Capabilities**: Multi-step reasoning, reflection, and in-depth analysis
- **Evaluation Focus**: Logical consistency, analytical depth, argument coherence

### 3. ReAct Agent (`ReAct_subgraph`)
**Purpose**: Tool-calling and external API interactions

- **Use Cases**: Weather queries, traffic information, external data retrieval
- **Capabilities**: Reasoning + Acting paradigm with tool orchestration
- **Evaluation Focus**: Tool selection accuracy, sequence logic, error recovery

### 4. RAG Agent (`rag_agent`)
**Purpose**: Knowledge base retrieval and document-based question answering

- **Use Cases**: "Summarise all the certificates", document extraction
- **Capabilities**: Context retrieval, generation, and factual grounding
- **Evaluation Focus**: Retrieval quality, generation faithfulness, factual accuracy

### 5. Planner Agent (`planner_subgraph`)
**Purpose**: Strategic planning and step-by-step task decomposition

- **Use Cases**: "Strategy to improve data quality"
- **Capabilities**: Multi-step planning, task breakdown, execution coordination
- **Evaluation Focus**: Plan completeness, logical sequencing, feasibility assessment

### Workflow Integration
Agents are orchestrated through a `workflow_graph.py` that manages state transitions, handoffs, and coordination between specialized subgraphs.

---

## RAGAS Evaluation Framework for Multi-Agent Systems

### Core RAGAS Data Structures

#### SingleTurnSample
For individual agent evaluation:
```python
from ragas import SingleTurnSample  # Correct import path

SingleTurnSample(
    user_input="What is the weather in SF?",
    response="Current weather in San Francisco is 68°F and sunny",
    retrieved_contexts=["Weather API data: SF temp 68°F"],  # Optional for non-RAG agents
    reference="The weather in San Francisco is currently 68°F with sunny skies"  # Optional, required for reference-based metrics
)
```

#### MultiTurnSample
For conversational and multi-step agent evaluation:
```python
from ragas import MultiTurnSample  # Correct import path

MultiTurnSample(
    messages=[
        {"role": "user", "content": "I need help planning a data strategy"},
        {"role": "assistant", "content": "I'll route this to our planning agent"},
        {"role": "user", "content": "Focus on data quality improvements"},
        {"role": "assistant", "content": "Here's a comprehensive 5-step plan..."}
    ],
    reference="Optional ground truth for the entire conversation",
    reference_tool_calls=[...],  # Optional for tool-using agents
    reference_topics=[...]  # Optional for topic adherence evaluation
)
```

### RAGAS Metric Categories for Multi-Agent Systems

#### 1. RAG-Specific Metrics (for RAG Agent)
- **Context Precision**: Measures the signal-to-noise ratio of retrieved contexts (proportion of relevant vs. irrelevant information)
- **Context Recall**: Evaluates completeness of information retrieval (whether all necessary information was retrieved)
- **Context Entity Recall**: Calculates the proportion of named entities from reference that appear in retrieved contexts
- **Noise Sensitivity**: Measures system robustness when irrelevant information is present (lower is better)
- **Faithfulness**: Assesses factual consistency between generated response and provided contexts (prevents hallucination)
- **Response Relevancy**: Evaluates how directly the response addresses the user's query

#### 2. Agentic System Metrics
- **Tool Call Accuracy**: Evaluates correct tool selection, argument passing, and sequencing (specific to ReAct Agent)
- **Topic Adherence**: Measures agent's ability to stay within designated conversational boundaries (precision/recall/F1)
- **Agent Goal Accuracy**: Assesses whether the agent successfully accomplished the user's underlying objective

#### 3. Traditional & General-Purpose Metrics
- **Exact Match**: Binary metric (0/1) for character-perfect matches, ideal for structured outputs like routing decisions
- **Semantic Similarity**: Cosine similarity between embeddings to measure meaning alignment (0-1 scale)
- **AspectCritic**: Binary LLM-based evaluation (pass/fail) using natural language criteria
- **RubricsScore**: Multi-level LLM-based scoring using detailed rubrics (typically 1-5 scale)

### RAGAS Metric Selection Matrix

| Agent Type | Primary Metrics | Custom Metrics | Cost Level | Rationale |
|------------|----------------|----------------|------------|------------|
| Router | ExactMatch, AspectCritic | RoutingConfidence | Low | Fast deterministic evaluation for classification tasks |
| RAG | Faithfulness, ContextPrecision, ContextRecall | KnowledgeAlignment | High | Comprehensive retrieval and generation quality assessment |
| ReAct | ToolCallAccuracy, AspectCritic | ToolOptimality | Medium | Tool usage validation with reasoning assessment |
| Reasoner | SemanticSimilarity, AspectCritic | ArgumentCoherence | Medium | Semantic understanding with logical structure evaluation |
| Planner | AgentGoalAccuracy, RubricsScore | PlanCompleteness | Medium | Goal achievement with structured quality assessment |

---

## Three-Tier Evaluation Strategy

### Tier 1: Unit-Level Agent Evaluation

#### Purpose
Isolate and measure individual agent performance to identify component-specific issues and optimize each agent independently.

#### Methodology
Each agent is evaluated in isolation using:
- **Test Data**: Agent-specific datasets with ground truth annotations
- **Metrics**: Carefully selected RAGAS metrics aligned with agent capabilities
- **Baseline**: Performance thresholds based on acceptable quality levels
- **Statistical Significance**: Multiple evaluation runs to ensure reliability

#### Router Agent Unit Testing
```python
from ragas.metrics import ExactMatch, AspectCritic
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# Binary evaluation for routing correctness
routing_metric = AspectCritic(
    name="routing_correctness",
    definition="Return 1 if the routing decision correctly matches the expected subgraph for the given query type"
)

# Comprehensive classification metrics
def evaluate_router_classification(y_true, y_pred, classes):
    """Compute detailed classification metrics for router performance."""
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, average=None
    )

    return {
        "accuracy": accuracy,
        "per_class_metrics": {
            cls: {"precision": p, "recall": r, "f1": f, "support": s}
            for cls, p, r, f, s in zip(classes, precision, recall, f1, support)
        }
    }
```

#### RAG Agent Unit Testing
```python
from ragas.metrics import (
    ContextPrecision, ContextRecall, Faithfulness,
    ResponseRelevancy, FactualCorrectness, NoiseSensitivity
)

rag_metrics = [
    # Retrieval Stage Metrics
    ContextPrecision(),           # Measures precision@k for retrieved contexts
    ContextRecall(),              # Measures recall of necessary information

    # Generation Stage Metrics
    Faithfulness(),               # Prevents hallucination (score: 0-1)
    ResponseRelevancy(),          # Ensures response addresses the query

    # End-to-End Metrics
    FactualCorrectness(atomicity="high", coverage="high"),  # Comprehensive fact checking
    NoiseSensitivity()            # Robustness to irrelevant context (lower is better)
]
```

#### ReAct Agent Unit Testing
```python
from ragas.metrics import ToolCallAccuracy

react_metrics = [
    ToolCallAccuracy(),           # Tool usage correctness
    AspectCritic(
        name="tool_sequence_logic",
        definition="Return 1 if the tool calling sequence follows logical reasoning"
    )
]
```

### Tier 2: Integration-Level Evaluation

#### Purpose
Assess the quality of agent interactions, handoffs, and state management across agent boundaries.

#### Handoff Quality Assessment
```python
# Custom RAGAS metric for handoff evaluation
from ragas.metrics.base import MetricWithLLM, SingleTurnMetric
from typing import Dict, Any

class HandoffQuality(MetricWithLLM, SingleTurnMetric):
    name = "handoff_quality"
    _required_columns = ["conversation_history", "handoff_context", "receiving_agent_response"]

    async def _single_turn_ascore(self, sample: Dict[str, Any]) -> float:
        """Evaluate the quality of agent handoffs.

        Returns:
            float: Score between 0-1, where 1 indicates perfect handoff.
        """
        prompt = f"""
        Evaluate the quality of this agent handoff:

        Previous Context: {sample.handoff_context}
        Receiving Agent Response: {sample.receiving_agent_response}

        Scoring Criteria (equal weight):
        1. Context Preservation: Are all relevant details maintained?
        2. Information Completeness: Is critical information transferred?
        3. Response Continuity: Does the response logically follow?

        Return a score between 0 and 1.
        """

        response = await self.llm.generate(prompt)
        return float(response.strip())
```

#### State Consistency Evaluation
```python
from ragas.metrics.base import MetricWithLLM, MultiTurnMetric
from typing import Dict, Any, List

class StateConsistency(MetricWithLLM, MultiTurnMetric):
    name = "state_consistency"
    _required_columns = ["messages", "state_transitions"]

    async def _multi_turn_ascore(self, sample: Dict[str, Any]) -> float:
        """Verify state consistency across agent interactions.

        Returns:
            float: Consistency score (0-1), where 1 is perfectly consistent.
        """
        contradictions = await self._detect_contradictions(sample.messages)
        state_preservation = await self._check_state_preservation(sample.state_transitions)
        context_maintenance = await self._assess_context_maintenance(sample.messages)

        # Weighted average with heavier penalty for contradictions
        consistency_score = (
            (1 - contradictions) * 0.5 +  # 50% weight on no contradictions
            state_preservation * 0.3 +      # 30% weight on state preservation
            context_maintenance * 0.2        # 20% weight on context maintenance
        )

        return consistency_score
```

### Tier 3: System-Level End-to-End Evaluation

#### Purpose
Evaluate complete workflow effectiveness from user input to final output, measuring overall system performance and user experience.

#### Comprehensive Workflow Assessment
```python
from ragas.metrics import AgentGoalAccuracy, TopicAdherence, SemanticSimilarity

# Core system-level metrics
system_metrics = [
    # RAGAS Built-in Metrics
    AgentGoalAccuracy(),          # Binary: Did the system achieve the user's goal?
    TopicAdherence(),             # Precision/Recall/F1 for staying on topic
    SemanticSimilarity(),         # Cosine similarity with reference (0-1)

    # Custom End-to-End Metrics (defined separately)
    WorkflowEfficiency(),         # Ratio: optimal_steps / actual_steps
    UserSatisfaction(),           # Rubric-based user experience score (1-5)
    TaskComplexityHandling()      # Success rate for multi-step tasks
]
```

#### Multi-Turn Conversation Evaluation
```python
from ragas.metrics import AspectCritic, RubricsScore

conversation_metrics = [
    # Binary coherence check
    AspectCritic(
        name="conversation_coherence",
        definition="""Return 1 if ALL of the following are true:
        - The conversation maintains logical flow between turns
        - Context from previous turns is properly referenced
        - No contradictory statements exist between agents
        - The narrative progression is natural
        Otherwise return 0."""
    ),

    # Binary task completion check
    AspectCritic(
        name="task_completion",
        definition="""Return 1 if the agent workflow successfully completes
        the user's requested task with all requirements met. Return 0 if
        any requirement is missing or the task is incomplete."""
    ),

    # Granular quality assessment
    RubricsScore(
        name="response_quality",
        rubric={
            1: "Poor - Response is incorrect, incomplete, or irrelevant",
            2: "Fair - Partially addresses user needs with significant gaps",
            3: "Good - Adequately fulfills requirements with minor issues",
            4: "Very Good - Comprehensive and accurate with good depth",
            5: "Excellent - Exceeds expectations with exceptional insight and clarity"
        }
    )
]
```

---

## Theoretical Foundations

### Evaluation Principles

1. **Single-Aspect Focus**: Each metric isolates one specific performance dimension to ensure actionable insights
2. **Statistical Significance**: Multiple evaluation runs with confidence intervals to ensure reliability
3. **Baseline Comparisons**: Performance measured against both absolute thresholds and relative baselines
4. **Cost-Benefit Analysis**: Balancing evaluation thoroughness with computational expense

### Metric Validity Considerations

- **Construct Validity**: Metrics must measure what they claim to measure
- **Content Validity**: Evaluation covers all relevant aspects of agent performance
- **Criterion Validity**: Correlation with human judgments and real-world outcomes
- **Reliability**: Consistent results across multiple evaluation runs

### Trade-offs in Evaluation Design

| Dimension | LLM-based Metrics | Traditional Metrics |
|-----------|------------------|---------------------|
| **Accuracy** | High semantic understanding | Limited to surface patterns |
| **Cost** | $0.01-0.20 per evaluation | Near zero |
| **Speed** | 1-5 seconds per sample | Milliseconds |
| **Determinism** | Probabilistic (~95% consistency) | 100% deterministic |
| **Explainability** | Detailed reasoning traces | Simple numerical scores |

---

## Agent-Specific Evaluation Methodologies

### Router Agent Evaluation

#### Core Metrics
1. **Classification Accuracy**: Overall routing decision correctness (threshold: ≥ 90%)
2. **Per-Class Metrics**: Precision, recall, F1 for each subgraph (balanced F1 ≥ 0.85)
3. **Confidence Calibration**: Routing decision confidence assessment (ECE ≤ 0.1)
4. **Fallback Handling**: Correctness validation and retry logic (recovery rate ≥ 95%)

#### Implementation Example
```python
# Based on existing eval_router_node.py
import asyncio
from pathlib import Path
from ragas import evaluate
from ragas.metrics import answer_correctness
from sklearn.metrics import confusion_matrix, classification_report

async def evaluate_router():
    \"\"\"Comprehensive router agent evaluation.\"\"\"\n    # Load test cases
    yaml_path = Path("src/graphs/router_subgraph/fewshots.yml")
    with yaml_path.open(encoding="utf-8") as f:
        examples = yaml.safe_load(f)["Routing_examples"]

    # Separate routing and direct-response cases
    routing_cases = [e for e in examples if e["output"].get("next_subgraph")]
    direct_cases = [e for e in examples if e["output"].get("direct_response_to_the_user")]

    # Evaluate routing accuracy
    routing_metrics = await evaluate_routing_classification(routing_cases)

    # Evaluate direct responses with RAGAS
    if direct_cases:
        dataset = create_dataset_from_examples(direct_cases)
        ragas_result = evaluate(dataset, metrics=[answer_correctness])

    return {
        "routing_accuracy": routing_metrics["accuracy"],
        "per_class_f1": routing_metrics["f1_scores"],
        "confusion_matrix": routing_metrics["confusion_matrix"],
        "direct_response_quality": ragas_result.scores() if direct_cases else None
    }
```

#### Custom Router Metrics
```python
from ragas.metrics.base import MetricWithLLM, SingleTurnMetric
import numpy as np

class RoutingConfidenceCalibration(MetricWithLLM, SingleTurnMetric):
    \"\"\"Evaluates if routing confidence scores are well-calibrated.\"\"\"

    name = "routing_confidence_calibration"
    _required_columns = ["user_input", "predicted_route", "confidence_score", "actual_route"]

    async def _single_turn_ascore(self, sample) -> float:
        \"\"\"Calculate Expected Calibration Error (ECE).

        Lower ECE indicates better calibration (confidence matches accuracy).
        \"\"\"
        # Group predictions by confidence bins
        bins = np.linspace(0, 1, 11)
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []

        for i in range(len(bins) - 1):
            mask = (sample.confidence_score >= bins[i]) & (sample.confidence_score < bins[i+1])
            if mask.sum() > 0:
                bin_accuracy = (sample.predicted_route[mask] == sample.actual_route[mask]).mean()
                bin_confidence = sample.confidence_score[mask].mean()
                bin_count = mask.sum()

                bin_accuracies.append(bin_accuracy)
                bin_confidences.append(bin_confidence)
                bin_counts.append(bin_count)

        # Calculate ECE
        total_samples = sum(bin_counts)
        ece = sum(
            (count / total_samples) * abs(acc - conf)
            for acc, conf, count in zip(bin_accuracies, bin_confidences, bin_counts)
        )

        # Return inverted ECE (1 - ECE) so higher is better
        return 1.0 - ece
```

---

## Best Practices and Guidelines

### 1. Test Dataset Construction

- **Diversity**: Include edge cases, adversarial examples, and typical use cases
- **Balance**: Ensure balanced representation across all agent types and decision paths
- **Versioning**: Maintain versioned test datasets with clear changelog
- **Size**: Minimum 100 examples per agent for statistical significance

### 2. Metric Selection Strategy

```python
def select_evaluation_metrics(agent_type: str, evaluation_phase: str) -> List[Metric]:
    \"\"\"Select appropriate metrics based on agent type and evaluation phase.\"\"\"

    if evaluation_phase == "development":
        # Use detailed, explainable metrics during development
        return get_explainable_metrics(agent_type)
    elif evaluation_phase == "ci_cd":
        # Use fast, deterministic metrics for CI/CD
        return get_fast_metrics(agent_type)
    elif evaluation_phase == "production":
        # Use lightweight, sample-based metrics for production
        return get_lightweight_metrics(agent_type)
```

### 3. Performance Thresholds

| Agent Type | Metric | Development | Staging | Production |
|------------|--------|-------------|---------|------------|
| Router | Accuracy | ≥ 85% | ≥ 90% | ≥ 95% |
| RAG | Faithfulness | ≥ 0.80 | ≥ 0.85 | ≥ 0.90 |
| ReAct | Tool Accuracy | ≥ 0.85 | ≥ 0.90 | ≥ 0.95 |
| Reasoner | Coherence | ≥ 0.75 | ≥ 0.80 | ≥ 0.85 |
| Planner | Goal Success | ≥ 0.80 | ≥ 0.85 | ≥ 0.90 |

### 4. Continuous Improvement Process

1. **Baseline Establishment**: Run initial evaluation to establish performance baselines
2. **Regular Evaluation**: Schedule automated evaluations (daily for development, per-commit for staging)
3. **Trend Analysis**: Monitor metric trends over time to detect degradation
4. **Root Cause Analysis**: Use detailed metrics to identify failure patterns
5. **Iterative Refinement**: Update agents based on evaluation insights

---

## Conclusion

The AWARE framework provides a comprehensive, theoretically grounded approach to evaluating LLM-based multi-agent systems. By leveraging RAGAS metrics across three evaluation tiers and incorporating custom metrics for multi-agent-specific challenges, it enables:

### Key Achievements

1. **Systematic Evaluation**: Transforms ad-hoc testing into rigorous, repeatable assessment
2. **Component Isolation**: Identifies specific failure points in complex workflows
3. **Quantifiable Quality**: Provides numerical metrics for objective comparison
4. **Cost-Aware Design**: Balances evaluation thoroughness with computational expense
5. **Production Readiness**: Supports continuous monitoring and quality assurance

### Critical Success Factors

- **Metric Validity**: Ensure metrics measure intended constructs
- **Statistical Rigor**: Use sufficient sample sizes and confidence intervals
- **Balanced Coverage**: Evaluate all aspects of system performance
- **Continuous Calibration**: Regularly update thresholds based on real-world performance
- **Stakeholder Alignment**: Ensure metrics align with business objectives

### Future Directions

1. **Automated Test Generation**: Use LLMs to generate diverse test cases
2. **Online Learning**: Incorporate production feedback into evaluation datasets
3. **Adversarial Testing**: Develop robustness metrics for edge cases
4. **Cross-Agent Dependencies**: Deeper evaluation of agent interaction patterns
5. **Human-in-the-Loop**: Integrate human evaluation for subjective quality aspects

The AWARE framework positions multi-agent systems for reliable, production-grade deployment while maintaining the flexibility to evolve through systematic, metric-driven iteration.

---

*AWARE Framework v1.0 - A RAGAS-based evaluation methodology for LangGraph multi-agent systems*
*For updates and contributions, see: [Repository Link]*
