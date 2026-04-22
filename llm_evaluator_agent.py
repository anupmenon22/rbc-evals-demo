"""
LLM Evaluator Agent — entry point.

This module creates and runs the LLMEvaluatorAgent, a Strands-based AI agent
that orchestrates LLM evaluation workflows through natural language commands.

Architecture:
    User prompt → Strands Agent (Claude Sonnet 4.5 on Bedrock)
                      ↓ decides which tools to call
                  Tool calls → load dataset, run evals, get summaries
                      ↓ results fed back to agent
                  Agent → formats and presents results to user

The agent uses Claude Sonnet 4.5 as its own reasoning model (via Bedrock
cross-region inference profile) and has access to four tools defined in
tools.py:

    - load_evaluation_dataset: Pulls samples from HuggingFace
    - run_bedrock_evaluation:  Evaluates a single Bedrock model with 18 scorers
    - run_all_evaluations:     Iterates over all configured models
    - get_experiment_summary:  Queries MLflow for run metrics

Usage:
    # Default: loads dataset, evaluates all models, summarizes
    python llm_evaluator.py

    # Custom prompt — the agent interprets and executes accordingly
    python llm_evaluator.py "Load 10 samples and evaluate only claude-haiku-4-5"
"""

import sys
import warnings
import atexit

# Suppress harmless multiprocess ResourceTracker error on Python 3.12 shutdown
def _suppress_resource_tracker():
    try:
        from multiprocess.resource_tracker import ResourceTracker
        ResourceTracker.__del__ = lambda self: None
    except Exception:
        pass

_suppress_resource_tracker()

from strands import Agent
from strands.models import BedrockModel
from src.tools import (
    load_evaluation_dataset,
    run_bedrock_evaluation,
    run_all_evaluations,
    get_experiment_summary,
    generate_eval_report,
)
from src import config

# System prompt that defines the agent's persona and capabilities.
# The agent uses this to understand what tools are available and how
# to structure its workflow (load → evaluate → summarize).
SYSTEM_PROMPT = """You are an LLM Evaluation Agent. You help users evaluate language models using standardized datasets and MLflow tracking.

Your capabilities:
1. Load evaluation datasets from HuggingFace (explodinggradients/ragas-wikiqa)
2. Run evaluations against multiple Bedrock models (Claude Sonnet 4.5, Claude Haiku 4.5)
3. Track all results in MLflow with metrics, parameters, and artifacts
4. Provide experiment summaries and comparisons

Workflow:
- First load the dataset
- Then run evaluations (individually or all at once)
- Finally summarize results

Always explain what you're doing and report results clearly."""


def create_agent():
    """Create the LLMEvaluatorAgent with Bedrock model and evaluation tools.

    The agent uses Claude Sonnet 4.5 via a US cross-region inference profile
    as its reasoning model. This is the model that interprets user prompts
    and decides which tools to call — it is NOT the model being evaluated.
    The models being evaluated are defined in config.BEDROCK_MODELS.
    """
    model = BedrockModel(
        model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        boto_session=config.get_boto_session(),
    )
    return Agent(
        model=model,
        name="LLMEvaluatorAgent",
        tools=[load_evaluation_dataset, run_bedrock_evaluation, run_all_evaluations, get_experiment_summary, generate_eval_report],
        system_prompt=SYSTEM_PROMPT,
    )


def main():
    """Run the agent with a user-provided prompt or the default evaluation workflow."""
    agent = create_agent()
    if len(sys.argv) > 1:
        # User provided a custom prompt via CLI args
        agent(" ".join(sys.argv[1:]))
    else:
        # Default: full evaluation pipeline
        agent(
            "Load the evaluation dataset, then run evaluations against all configured Bedrock models, "
            "and finally give me a summary of the results."
        )


if __name__ == "__main__":
    main()
