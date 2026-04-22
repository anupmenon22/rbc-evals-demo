"""
Agent Behavior Evaluator — evaluates the LLMEvaluatorAgent's behavior using Strands Evals.

While tools.py evaluates LLM *output quality* (correctness, fluency, etc.),
this module evaluates the *agent's behavior* — did it pick the right tools,
use the right parameters, follow the right trajectory, and produce helpful output?

Evaluation categories:
  1. Output quality   — Is the agent's final response accurate and helpful?
  2. Trajectory       — Did the agent call the right tools in the right order?
  3. Tool selection   — Did the agent pick the correct tools for the task?
  4. Tool parameters  — Did the agent pass correct arguments to tools?
  5. Helpfulness      — Was the agent genuinely helpful (trace-based)?
  6. Faithfulness     — Did the agent stay faithful to tool outputs?
  7. Goal success     — Did the agent achieve the user's stated goal?

Results are logged to MLflow alongside the LLM evaluation metrics for a
unified view of both output quality and agent behavior.

Usage:
    python agent_behavior_eval.py
"""

import json
import os
import sys

# Suppress harmless multiprocess ResourceTracker error on Python 3.12 shutdown
try:
    from multiprocess.resource_tracker import ResourceTracker
    ResourceTracker.__del__ = lambda self: None
except Exception:
    pass

import mlflow
from strands import Agent
from strands.models import BedrockModel
from strands_evals import Case, Experiment
from strands_evals.evaluators import (
    FaithfulnessEvaluator,
    HelpfulnessEvaluator,
    OutputEvaluator,
    ToolSelectionAccuracyEvaluator,
    TrajectoryEvaluator,
)
from strands_evals.extractors import tools_use_extractor
from strands_evals.telemetry import StrandsEvalsTelemetry
from strands_evals.mappers import StrandsInMemorySessionMapper

from src.tools import (
    load_evaluation_dataset,
    run_bedrock_evaluation,
    run_all_evaluations,
    get_experiment_summary,
)
from src import config


# =============================================================================
# Agent factory
# =============================================================================

TOOLS = [load_evaluation_dataset, run_bedrock_evaluation, run_all_evaluations, get_experiment_summary]

def _create_agent(**kwargs):
    """Create a fresh LLMEvaluatorAgent instance for each test case."""
    model = BedrockModel(
        model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        boto_session=config.get_boto_session(),
    )
    return Agent(
        model=model,
        name="LLMEvaluatorAgent",
        tools=TOOLS,
        system_prompt=(
            "You are an LLM Evaluation Agent. You help users evaluate language models "
            "using standardized datasets and MLflow tracking."
        ),
        callback_handler=None,  # Suppress console output during eval
        **kwargs,
    )


# =============================================================================
# Test cases
# =============================================================================

test_cases = [
    Case[str, str](
        name="load-dataset",
        input="Load 5 samples from the dataset",
        expected_output="Loaded 5 samples",
        expected_trajectory=["load_evaluation_dataset"],
        metadata={"category": "dataset"},
    ),
    Case[str, str](
        name="eval-single-model",
        input="Load 3 samples and evaluate claude-sonnet-4-5",
        expected_output="Evaluation complete",
        expected_trajectory=["load_evaluation_dataset", "run_bedrock_evaluation"],
        metadata={"category": "evaluation"},
    ),
    Case[str, str](
        name="get-summary",
        input="Show me a summary of the experiment results",
        expected_output="Experiment summary",
        expected_trajectory=["get_experiment_summary"],
        metadata={"category": "summary"},
    ),
]


# =============================================================================
# Task functions (called by Strands Evals for each test case)
# =============================================================================

# Shared telemetry for trace-based evaluators
telemetry = StrandsEvalsTelemetry().setup_in_memory_exporter()


def trajectory_task(case: Case) -> dict:
    """Run the agent and return output + tool trajectory."""
    agent = _create_agent()
    response = agent(case.input)
    trajectory = tools_use_extractor.extract_agent_tools_used_from_messages(agent.messages)
    return {"output": str(response), "trajectory": trajectory}


def trace_task(case: Case) -> dict:
    """Run the agent with telemetry and return output + session traces."""
    telemetry.in_memory_exporter.clear()
    agent = _create_agent(
        trace_attributes={
            "gen_ai.conversation.id": case.session_id,
            "session.id": case.session_id,
        },
    )
    response = agent(case.input)
    finished_spans = telemetry.in_memory_exporter.get_finished_spans()
    mapper = StrandsInMemorySessionMapper()
    session = mapper.map_to_session(finished_spans, session_id=case.session_id)
    return {"output": str(response), "trajectory": session}


# =============================================================================
# Evaluator definitions
# =============================================================================

# Judge model for all Strands Evals evaluators (same Bedrock model as the agent)
judge_model = BedrockModel(
    model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    boto_session=config.get_boto_session(),
)

# Get tool descriptions for trajectory evaluator
sample_agent = _create_agent()
tool_descriptions = tools_use_extractor.extract_tools_description(sample_agent, is_short=True)

output_evaluator = OutputEvaluator(
    model=judge_model,
    rubric="""
    Evaluate the agent's response:
    1. Did it correctly perform the requested action (load data, run eval, show summary)?
    2. Did it report results clearly with relevant metrics or confirmation?
    3. Was the response well-structured and informative?
    Score 1.0 if excellent, 0.5 if partially correct, 0.0 if wrong or unhelpful.
    """,
    include_inputs=True,
)

trajectory_evaluator = TrajectoryEvaluator(
    model=judge_model,
    rubric="""
    Evaluate the tool usage:
    1. Were the correct tools selected for the task?
    2. Were they called in the right order?
    3. Were unnecessary tools avoided?
    Score 1.0 if optimal, 0.5 if correct but suboptimal, 0.0 if wrong tools used.
    """,
    include_inputs=True,
)
trajectory_evaluator.update_trajectory_description(tool_descriptions)

tool_selection_evaluator = ToolSelectionAccuracyEvaluator(model=judge_model)
helpfulness_evaluator = HelpfulnessEvaluator(model=judge_model)
faithfulness_evaluator = FaithfulnessEvaluator(model=judge_model)


# =============================================================================
# Run evaluations and log to MLflow
# =============================================================================

def _print_report(report):
    """Print evaluation report without interactive prompts."""
    print(f"\n  📊 {report.evaluator_name}")
    print(f"     Overall Score: {report.overall_score:.2f}  |  Pass Rate: {sum(report.test_passes)}/{len(report.test_passes)}")
    for i, (score, passed, reason) in enumerate(zip(report.scores, report.test_passes, report.reasons)):
        status = "✅" if passed else "❌"
        print(f"     {status} {test_cases[i].name}: {score:.2f} — {reason[:100]}")


def run_all():
    """Run all Strands Evals evaluations and log results to MLflow."""
    from tools import _ensure_aws_env_vars
    _ensure_aws_env_vars()

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment("agent-behavior-evaluation")

    print("=" * 60)
    print("Agent Behavior Evaluation (Strands Evals)")
    print("=" * 60)

    all_results = {}

    # --- Pass 1: Output + Trajectory (single agent run per case) ---
    print("\n[1/2] Running output + trajectory evaluation...")
    exp = Experiment[str, str](
        cases=test_cases,
        evaluators=[output_evaluator, trajectory_evaluator, tool_selection_evaluator],
    )
    reports = exp.run_evaluations(trajectory_task)
    for report in reports:
        _print_report(report)
        all_results[report.evaluator_name] = {
            test_cases[i].name: {"score": report.scores[i], "passed": report.test_passes[i]}
            for i in range(len(test_cases))
        }

    # --- Pass 2: Trace-based (helpfulness + faithfulness) ---
    print("\n[2/2] Running trace-based evaluation...")
    trace_exp = Experiment[str, str](
        cases=test_cases,
        evaluators=[helpfulness_evaluator, faithfulness_evaluator],
    )
    trace_reports = trace_exp.run_evaluations(trace_task)
    for report in trace_reports:
        _print_report(report)
        all_results[report.evaluator_name] = {
            test_cases[i].name: {"score": report.scores[i], "passed": report.test_passes[i]}
            for i in range(len(test_cases))
        }

    # --- Log to MLflow ---
    print("\n[✓] Logging results to MLflow...")
    mlflow.set_experiment("agent-behavior-evaluation")  # Re-set after tools may have changed it
    with mlflow.start_run(run_name="agent-behavior-eval"):
        mlflow.log_param("num_test_cases", len(test_cases))
        mlflow.log_param("evaluators", list(all_results.keys()))

        for eval_name, cases in all_results.items():
            scores = [c["score"] for c in cases.values() if c["score"] is not None]
            if scores:
                mlflow.log_metric(f"{eval_name}/mean_score", sum(scores) / len(scores))
                passed = sum(1 for c in cases.values() if c["passed"])
                mlflow.log_metric(f"{eval_name}/pass_rate", passed / len(cases))

        with open("/tmp/agent_behavior_results.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        mlflow.log_artifact("/tmp/agent_behavior_results.json")

    print("\n✅ All agent behavior evaluations complete. Results logged to MLflow.")

    # --- Generate EMRM report ---
    print("[✓] Generating EMRM evaluation report...")
    from src.report_generator import generate_agent_behavior_report
    filepath = generate_agent_behavior_report(all_results, len(test_cases))
    print(f"    Report saved: {filepath}")


if __name__ == "__main__":
    run_all()
