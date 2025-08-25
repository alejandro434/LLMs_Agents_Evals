# AWARE Framework - Part 2: Custom Metrics and Implementation

## Custom RAGAS Metrics for Multi-Agent Systems

### 1. Workflow Orchestration Metrics

#### Agent Coordination Quality
```python
from ragas.metrics.base import MetricWithLLM, MultiTurnMetric

class AgentCoordination(MetricWithLLM, MultiTurnMetric):
    name = "agent_coordination"
    _required_columns = ["messages", "agent_transitions", "shared_context"]

    async def _multi_turn_ascore(self, sample: MultiTurnSample) -> float:
        """
        Evaluates the quality of coordination between agents in a multi-turn workflow.

        Assesses:
        - Smooth handoffs between agents
        - Context preservation across transitions
        - Appropriate agent selection for tasks
        - Collaborative problem-solving effectiveness
        """

        transition_quality = await self._evaluate_transitions(sample.agent_transitions)
        context_preservation = await self._evaluate_context_preservation(sample.shared_context)
        agent_selection = await self._evaluate_agent_selection(sample.messages, sample.agent_transitions)

        return (transition_quality + context_preservation + agent_selection) / 3
```

#### Workflow Efficiency
```python
from ragas.metrics.base import Metric, MultiTurnMetric

class WorkflowEfficiency(Metric, MultiTurnMetric):
    name = "workflow_efficiency"
    _required_columns = ["messages", "execution_path", "optimal_path"]

    async def _multi_turn_ascore(self, sample: MultiTurnSample) -> float:
        """
        Measures the efficiency of the chosen workflow path.
        """
        actual_steps = len(sample.execution_path)
        optimal_steps = len(sample.optimal_path)

        # Calculate efficiency score
        efficiency = optimal_steps / actual_steps if actual_steps > 0 else 0.0

        # Penalize redundant invocations
        redundant_calls = self._count_redundant_calls(sample.execution_path)
        redundancy_penalty = redundant_calls * 0.1

        return max(0, efficiency - redundancy_penalty)
```

### 2. Multi-Agent Communication Metrics

#### Context Preservation
```python
class ContextPreservation(MetricWithLLM, MultiTurnMetric):
    name = "context_preservation"
    _required_columns = ["messages", "context_handoffs"]

    async def _multi_turn_ascore(self, sample: MultiTurnSample) -> float:
        """
        Evaluates how well context is preserved across agent handoffs.
        """
        preservation_scores = []

        for handoff in sample.context_handoffs:
            source_context = handoff.get("source_context", "")
            target_context = handoff.get("target_context", "")

            prompt = f"""
            Evaluate context preservation in this handoff:
            Source Context: {source_context}
            Target Context: {target_context}

            Score (0-1) based on:
            - Information completeness
            - Relevance maintenance
            - Critical detail retention
            """

            score = await self.llm.generate(prompt)
            preservation_scores.append(float(score.strip()))

        return sum(preservation_scores) / len(preservation_scores) if preservation_scores else 0
```

### 3. Task Completion Metrics

#### Multi-Step Task Success
```python
class MultiStepTaskSuccess(MetricWithLLM, MultiTurnMetric):
    name = "multi_step_task_success"
    _required_columns = ["messages", "task_requirements", "final_outcome"]

    async def _multi_turn_ascore(self, sample: MultiTurnSample) -> float:
        """
        Evaluates success in completing complex, multi-step tasks.
        """
        requirements = sample.task_requirements
        completed_requirements = []

        for req in requirements:
            if await self._is_requirement_fulfilled(req, sample.final_outcome):
                completed_requirements.append(req)

        completion_rate = len(completed_requirements) / len(requirements) if requirements else 0

        # Assess quality of final deliverable
        quality_prompt = f"""
        Evaluate the quality of this task outcome:
        Requirements: {requirements}
        Final Outcome: {sample.final_outcome}

        Score quality (0-1):
        """

        quality_score = await self.llm.generate(quality_prompt)
        quality = float(quality_score.strip())

        return (completion_rate * 0.6) + (quality * 0.4)
```

## Implementation Guidelines

### 1. Evaluation Infrastructure Setup

#### Project Structure
```
evals/
├── agents/
│   ├── router/
│   │   ├── unit_tests.py
│   │   ├── test_data.yml
│   │   └── metrics_config.py
│   ├── rag/
│   │   ├── unit_tests.py
│   │   ├── test_datasets/
│   │   └── custom_metrics.py
│   ├── react/
│   │   ├── unit_tests.py
│   │   └── tool_scenarios.yml
│   ├── reasoner/
│   │   ├── unit_tests.py
│   │   └── reasoning_benchmarks.yml
│   └── planner/
│       ├── unit_tests.py
│       └── planning_scenarios.yml
├── integration/
│   ├── handoff_tests.py
│   ├── state_consistency_tests.py
│   └── coordination_metrics.py
├── system/
│   ├── end_to_end_tests.py
│   ├── workflow_evaluation.py
│   └── user_scenarios.yml
└── common/
    ├── custom_ragas_metrics.py
    ├── evaluation_utils.py
    └── reporting.py
```

#### Environment Configuration
```python
# config/evaluation_config.py
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class EvaluationConfig:
    # Model Configuration
    judge_llm: str = "gpt-4o"
    embedding_model: str = "text-embedding-ada-002"
    temperature: float = 0.0

    # Cost Management
    max_token_budget: int = 1000000
    enable_cost_tracking: bool = True

    # Evaluation Scope
    unit_tests_enabled: bool = True
    integration_tests_enabled: bool = True
    system_tests_enabled: bool = True

    # Metric Selection
    core_metrics: List[str] = None
    custom_metrics: List[str] = None

    # Paths
    router_test_path: str = "src/graphs/router_subgraph/fewshots.yml"
    knowledge_base_path: str = "src/graphs/rag_agent/knowledge_base"

    def __post_init__(self):
        if self.core_metrics is None:
            self.core_metrics = [
                "faithfulness", "context_precision", "context_recall",
                "response_relevancy", "tool_call_accuracy", "agent_goal_accuracy"
            ]
        if self.custom_metrics is None:
            self.custom_metrics = [
                "workflow_efficiency", "context_preservation",
                "agent_coordination", "multi_step_task_success"
            ]
```

### 2. Unit Test Implementation

#### Router Agent Unit Tests
```python
# evals/agents/router/unit_tests.py
import asyncio
from pathlib import Path
from typing import List, Dict, Any

from ragas import evaluate
from ragas.metrics import ExactMatch, AspectCritic
from datasets import Dataset

from src.graphs.router_subgraph.chains import chain
from src.graphs.router_subgraph.nodes_logic import router_node
from src.graphs.router_subgraph.schemas import RouterSubgraphState
from evals.common.custom_ragas_metrics import RoutingConfidence

class RouterEvaluator:
    def __init__(self, config_path: Path):
        self.test_cases = self._load_test_cases(config_path)
        self.metrics = [
            ExactMatch(),
            AspectCritic(
                name="routing_logic",
                definition="Return 1 if the routing decision demonstrates sound logical reasoning"
            ),
            RoutingConfidence()
        ]

    async def run_evaluation(self) -> Dict[str, Any]:
        """Run comprehensive router evaluation."""
        # Classification evaluation
        classification_results = await self._evaluate_classification()

        # RAGAS evaluation for direct responses
        ragas_results = await self._evaluate_direct_responses()

        # Confidence calibration assessment
        confidence_results = await self._evaluate_confidence()

        return {
            "classification": classification_results,
            "direct_responses": ragas_results,
            "confidence": confidence_results,
            "overall_score": self._compute_overall_score(
                classification_results, ragas_results, confidence_results
            )
        }

    async def _evaluate_classification(self) -> Dict[str, float]:
        """Evaluate routing classification accuracy."""
        y_true, y_pred = [], []

        for test_case in self.test_cases:
            if test_case.get("expected_subgraph"):
                state = RouterSubgraphState(user_input=test_case["input"])
                command = await router_node(state)

                y_true.append(test_case["expected_subgraph"])
                y_pred.append(str(command.goto))

        return self._compute_classification_metrics(y_true, y_pred)
```

#### RAG Agent Unit Tests
```python
# evals/agents/rag/unit_tests.py
from ragas.metrics import (
    ContextPrecision, ContextRecall, Faithfulness,
    ResponseRelevancy, FactualCorrectness
)
from evals.common.custom_ragas_metrics import (
    KnowledgeBaseAlignment, CitationAccuracy
)

class RAGEvaluator:
    def __init__(self, knowledge_base_path: Path):
        self.knowledge_base = self._load_knowledge_base(knowledge_base_path)
        self.metrics = [
            ContextPrecision(),
            ContextRecall(),
            Faithfulness(),
            ResponseRelevancy(),
            FactualCorrectness(atomicity="high", coverage="high"),
            KnowledgeBaseAlignment(),
            CitationAccuracy()
        ]

    async def run_evaluation(self) -> Dict[str, Any]:
        """Run comprehensive RAG evaluation."""
        # Retrieval quality assessment
        retrieval_results = await self._evaluate_retrieval()

        # Generation quality assessment
        generation_results = await self._evaluate_generation()

        # End-to-end assessment
        e2e_results = await self._evaluate_end_to_end()

        return {
            "retrieval": retrieval_results,
            "generation": generation_results,
            "end_to_end": e2e_results,
            "overall_score": self._compute_rag_score(
                retrieval_results, generation_results, e2e_results
            )
        }
```

### 3. Integration Test Implementation

#### Agent Handoff Evaluation
```python
# evals/integration/handoff_tests.py
from evals.common.custom_ragas_metrics import (
    ContextPreservation, InformationCoherence, AgentCoordination
)

class HandoffEvaluator:
    def __init__(self):
        self.metrics = [
            ContextPreservation(),
            InformationCoherence(),
            AgentCoordination()
        ]

    async def evaluate_handoff_sequence(
        self,
        handoff_scenario: Dict[str, Any]
    ) -> Dict[str, float]:
        """Evaluate a specific agent handoff sequence."""

        # Create MultiTurnSample from handoff scenario
        sample = MultiTurnSample(
            messages=handoff_scenario["conversation"],
            agent_transitions=handoff_scenario["transitions"],
            shared_context=handoff_scenario["context"]
        )

        results = {}
        for metric in self.metrics:
            score = await metric._multi_turn_ascore(sample)
            results[metric.name] = score

        return results

    async def evaluate_all_handoff_patterns(self) -> Dict[str, Any]:
        """Evaluate all possible agent handoff patterns."""
        handoff_patterns = [
            ("router", "reasoner"),
            ("router", "rag_agent"),
            ("router", "ReAct_subgraph"),
            ("router", "planner_subgraph"),
            ("planner_subgraph", "executor_subgraph"),
            # Add more patterns as needed
        ]

        results = {}
        for source, target in handoff_patterns:
            scenario = self._create_handoff_scenario(source, target)
            results[f"{source}_to_{target}"] = await self.evaluate_handoff_sequence(scenario)

        return results
```

### 4. System-Level Test Implementation

#### End-to-End Workflow Evaluation
```python
# evals/system/end_to_end_tests.py
from ragas.metrics import AgentGoalAccuracy, TopicAdherence
from evals.common.custom_ragas_metrics import (
    WorkflowEfficiency, MultiStepTaskSuccess, AdaptiveProblemSolving
)

class SystemEvaluator:
    def __init__(self):
        self.metrics = [
            AgentGoalAccuracy(),
            TopicAdherence(),
            WorkflowEfficiency(),
            MultiStepTaskSuccess(),
            AdaptiveProblemSolving()
        ]

    async def evaluate_user_scenario(
        self,
        scenario: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate complete user interaction scenario."""

        # Execute workflow for scenario
        workflow_result = await self._execute_workflow(scenario)

        # Create evaluation sample
        sample = MultiTurnSample(
            messages=workflow_result["conversation"],
            execution_path=workflow_result["agent_path"],
            final_outcome=workflow_result["result"],
            task_requirements=scenario.get("requirements", [])
        )

        # Run evaluation metrics
        results = {}
        for metric in self.metrics:
            if hasattr(metric, '_multi_turn_ascore'):
                score = await metric._multi_turn_ascore(sample)
            else:
                score = await metric._single_turn_ascore(sample)
            results[metric.name] = score

        return {
            "scenario_id": scenario["id"],
            "scenario_type": scenario["type"],
            "metrics": results,
            "overall_score": sum(results.values()) / len(results),
            "execution_details": workflow_result
        }

    async def _execute_workflow(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the multi-agent workflow for a given scenario."""
        # Import the main workflow graph
        from src.graphs.workflow_graph import workflow_graph

        # Execute the workflow
        result = await workflow_graph.ainvoke(
            {"user_input": scenario["input"]},
            config={"configurable": {"thread_id": scenario["id"]}}
        )

        return {
            "conversation": result.get("messages", []),
            "agent_path": result.get("execution_path", []),
            "result": result.get("final_response", "")
        }
```

### 5. Automated Testing Pipeline

#### CI/CD Integration
```python
# evals/pipeline/automated_evaluation.py
import asyncio
from typing import List, Dict, Any
from datetime import datetime

from evals.agents.router.unit_tests import RouterEvaluator
from evals.agents.rag.unit_tests import RAGEvaluator
from evals.agents.react.unit_tests import ReActEvaluator
from evals.agents.reasoner.unit_tests import ReasonerEvaluator
from evals.agents.planner.unit_tests import PlannerEvaluator
from evals.integration.handoff_tests import HandoffEvaluator
from evals.system.end_to_end_tests import SystemEvaluator

class EvaluationPipeline:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.evaluators = {
            "router": RouterEvaluator(config.router_test_path),
            "rag": RAGEvaluator(config.knowledge_base_path),
            "react": ReActEvaluator(),
            "reasoner": ReasonerEvaluator(),
            "planner": PlannerEvaluator(),
            "handoff": HandoffEvaluator(),
            "system": SystemEvaluator()
        }

    async def run_full_evaluation(self) -> Dict[str, Any]:
        """Run complete evaluation pipeline."""
        start_time = datetime.now()
        results = {}

        # Unit tests
        if self.config.unit_tests_enabled:
            results["unit"] = await self._run_unit_tests()

        # Integration tests
        if self.config.integration_tests_enabled:
            results["integration"] = await self._run_integration_tests()

        # System tests
        if self.config.system_tests_enabled:
            results["system"] = await self._run_system_tests()

        # Generate comprehensive report
        report = self._generate_evaluation_report(
            results,
            start_time,
            datetime.now()
        )

        return report

    async def _run_unit_tests(self) -> Dict[str, Any]:
        """Execute all unit-level evaluations."""
        unit_results = {}

        # Run evaluations in parallel for efficiency
        tasks = [
            self.evaluators["router"].run_evaluation(),
            self.evaluators["rag"].run_evaluation(),
            self.evaluators["react"].run_evaluation(),
            self.evaluators["reasoner"].run_evaluation(),
            self.evaluators["planner"].run_evaluation()
        ]

        results = await asyncio.gather(*tasks)

        unit_results["router"] = results[0]
        unit_results["rag"] = results[1]
        unit_results["react"] = results[2]
        unit_results["reasoner"] = results[3]
        unit_results["planner"] = results[4]

        return unit_results
```

## Operational Considerations

### 1. Cost Management

#### Token Usage Optimization
```python
from ragas.cost import get_token_usage_for_openai

# Configure cost tracking
cost_parser = get_token_usage_for_openai()

# Run evaluation with cost tracking
result = evaluate(
    dataset,
    metrics=selected_metrics,
    token_usage_parser=cost_parser
)

# Calculate and monitor costs
total_cost = result.total_cost(
    cost_per_1k_input_tokens=0.005,
    cost_per_1k_output_tokens=0.015
)

print(f"Evaluation cost: ${total_cost:.2f}")
```

#### Metric Selection Strategy
```python
class CostAwareMetricSelector:
    def __init__(self, budget_limit: float):
        self.budget_limit = budget_limit
        self.metric_costs = {
            "exact_match": 0.0,  # Traditional metric, no LLM cost
            "faithfulness": 0.15,  # High cost, high explainability
            "response_groundedness": 0.08,  # Lower cost, less detailed
            "context_precision": 0.12,
            "agent_goal_accuracy": 0.20
        }

    def select_metrics(self, priority_metrics: List[str]) -> List[str]:
        """Select metrics based on budget and priority."""
        selected = []
        estimated_cost = 0.0

        for metric in sorted(priority_metrics, key=lambda m: self.metric_costs.get(m, 0.1)):
            metric_cost = self.metric_costs.get(metric, 0.1)
            if estimated_cost + metric_cost <= self.budget_limit:
                selected.append(metric)
                estimated_cost += metric_cost

        return selected
```

### 2. Performance Monitoring

#### Production Evaluation Pipeline
```python
class ProductionMonitor:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.metrics_cache = {}

    async def continuous_evaluation(
        self,
        conversation_logs: List[Dict[str, Any]]
    ) -> None:
        """Run continuous evaluation on production conversations."""

        # Sample conversations for evaluation
        sampled_logs = self._sample_conversations(
            conversation_logs,
            sample_rate=0.1
        )

        # Run lightweight evaluation metrics
        lightweight_metrics = [
            "response_groundedness",  # NVIDIA metric - faster
            "context_relevance",      # NVIDIA metric - faster
            "exact_match",           # Traditional - instant
        ]

        for log in sampled_logs:
            sample = self._convert_log_to_sample(log)
            results = await self._evaluate_sample(sample, lightweight_metrics)

            # Store results for trend analysis
            self._store_evaluation_results(log["conversation_id"], results)

            # Alert on quality degradation
            if results["overall_score"] < self.config.quality_threshold:
                await self._send_quality_alert(log, results)
```

## Conclusion

The AWARE evaluation framework provides a comprehensive, systematic approach to evaluating LLM-based multi-agent systems. By leveraging RAGAS metrics across three evaluation tiers - unit, integration, and system-level - development teams can:

1. **Ensure Component Quality**: Individual agent performance through specialized unit tests
2. **Validate Integration**: Agent coordination and handoff quality through integration tests
3. **Verify System Effectiveness**: End-to-end workflow performance through system tests
4. **Enable Continuous Improvement**: Data-driven optimization through comprehensive metrics
5. **Manage Operational Costs**: Strategic metric selection and cost-aware evaluation pipelines

### Key Benefits

- **Objective Assessment**: Transforms subjective "vibe checks" into quantifiable metrics
- **Comprehensive Coverage**: Addresses all aspects of multi-agent system performance
- **Actionable Insights**: Provides specific, component-level feedback for optimization
- **Scalable Implementation**: Supports both development-time and production monitoring
- **Cost-Effective Operations**: Balances evaluation quality with computational efficiency

### Implementation Roadmap

1. **Phase 1**: Implement unit-level evaluations for each agent type
2. **Phase 2**: Develop integration-level handoff and coordination metrics
3. **Phase 3**: Deploy system-level end-to-end evaluation capabilities
4. **Phase 4**: Integrate continuous monitoring and automated quality assurance
5. **Phase 5**: Optimize cost-performance trade-offs and scale evaluation infrastructure

This framework positions multi-agent systems for reliable, production-grade deployment while maintaining the flexibility to evolve and improve through systematic, metric-driven iteration.

---

*AWARE framework v1.0 - Built on RAGAS evaluation methodology for LangGraph-based multi-agent systems*
