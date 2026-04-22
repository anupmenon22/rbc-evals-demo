"""
Evaluation tools for the LLM Evaluator Agent.

This module defines the Strands agent tools and evaluation scorers used to
evaluate Bedrock LLMs against the ragas-wikiqa dataset. All evaluation results
are tracked in MLflow via the serverless MLflow App on SageMaker.

The evaluation pipeline uses three categories of scorers (18 total):

1. MLflow Built-in LLM-as-Judge Scorers (5):
   - RelevanceToQuery: Does the response address the user's question?
   - Equivalence: Is the response semantically equivalent to the ground truth?
   - Fluency: Is the response grammatically correct and naturally flowing?
   - Guidelines("answer_groundedness"): Is the answer grounded in context?
   - Guidelines("answer_completeness"): Does the answer fully address the question?

2. Custom LLM-as-Judge Scorers via make_judge (3):
   - factual_consistency: Is the response consistent with the provided context?
   - professionalism: Does the response use formal, professional language?
   - correctness: Is the response factually correct vs the expected answer?

3. DeepEval Scorers via MLflow integration (6):
   - Faithfulness: Are all claims in the response supported by the context?
     (Uses claim decomposition — more rigorous than Guidelines-based checks)
   - AnswerRelevancy: Is the answer relevant? (Uses RAGAS-style claim decomposition)
   - ContextualRelevancy: Is the retrieved context relevant to the question?
   - Hallucination: Does the response contain hallucinated information?
   - Toxicity: Does the response contain toxic or harmful content?
   - Bias: Does the response exhibit gender, racial, or other biases?

4. Code-based Scorers (4):
   - exact_match: Exact string match between output and ground truth
   - is_concise: Is the response under 100 words?
   - word_overlap: Token overlap ratio between output and ground truth
   - response_length: Word count of the response

All LLM-as-Judge scorers use Bedrock Claude Sonnet 4.5 as the judge model
(configured in config.JUDGE_MODEL). No OpenAI API key is required.
"""

import json
import os
import re
import pandas as pd
import mlflow
import boto3
from datasets import load_dataset
from strands import tool
from mlflow.genai import scorer
from mlflow.genai.scorers import (
    Equivalence,
    Fluency,
    Guidelines,
    RelevanceToQuery,
)
from mlflow.genai import make_judge
from mlflow.genai.scorers.deepeval import DeepEvalScorer
from src import config


# =============================================================================
# Helper functions
# =============================================================================

def _ensure_aws_env_vars():
    """Export AWS credentials as environment variables for MLflow's Bedrock adapter.

    MLflow's built-in Bedrock judge adapter (used by scorers like Correctness,
    RelevanceToQuery, etc.) requires explicit AWS_ACCESS_KEY_ID and
    AWS_SECRET_ACCESS_KEY environment variables. It does not use the default
    boto3 credential chain. This function bridges that gap by reading
    credentials from the agent's boto3 session (which may have assumed the
    least-privilege agent-execution role) and exporting them.
    """
    session = config.get_boto_session()
    creds = session.get_credentials().get_frozen_credentials()
    os.environ["AWS_ACCESS_KEY_ID"] = creds.access_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = creds.secret_key
    os.environ["AWS_DEFAULT_REGION"] = config.AWS_REGION
    if creds.token:
        os.environ["AWS_SESSION_TOKEN"] = creds.token


def _extract_facts(text: str) -> list[str]:
    """Extract short factual claims from a ground truth answer.

    Splits the text on sentence boundaries and returns up to 5 sentences,
    each truncated to 200 characters. This format is required by MLflow's
    Correctness scorer which evaluates each fact individually.

    Args:
        text: The full ground truth answer text.

    Returns:
        A list of up to 5 short factual claim strings.
    """
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 10]
    return [s[:200] for s in sentences[:5]]


# =============================================================================
# Code-based scorers (heuristic)
#
# These scorers use deterministic logic — no LLM calls. They run instantly
# and provide objective, reproducible metrics. Decorated with @scorer so
# MLflow tracks them alongside LLM-as-Judge metrics in the same run.
# =============================================================================

@scorer
def exact_match(outputs: str, expectations: dict) -> bool:
    """Check if the model output exactly matches the expected response.

    Case-insensitive, whitespace-trimmed comparison. Returns True/False.
    Aggregated as a ratio (0.0 to 1.0) across all samples by MLflow.
    """
    return outputs.strip().lower() == expectations["expected_response"].strip().lower()


@scorer
def is_concise(outputs: str) -> bool:
    """Check if the response is concise (100 words or fewer).

    Useful for detecting overly verbose model outputs. Returns True/False.
    """
    return len(outputs.split()) <= 100


@scorer
def word_overlap(outputs: str, expectations: dict) -> float:
    """Compute the word overlap ratio between the output and expected response.

    Calculates: |output_words ∩ expected_words| / |expected_words|
    A value of 1.0 means every word in the expected response appears in the output.
    A value of 0.0 means no words overlap. This is a simple lexical similarity metric.
    """
    output_words = set(outputs.lower().split())
    expected_words = set(expectations["expected_response"].lower().split())
    if not expected_words:
        return 0.0
    return len(output_words & expected_words) / len(expected_words)


@scorer
def response_length(outputs: str) -> int:
    """Return the word count of the model's response.

    Aggregated as mean/variance across samples by MLflow. Useful for
    comparing verbosity across different models.
    """
    return len(outputs.split())


# =============================================================================
# Custom LLM-as-Judge scorers (via make_judge)
#
# These use MLflow's make_judge API to create custom evaluation criteria
# with Jinja2 prompt templates. The judge model (Bedrock Claude) scores
# each response as yes/no. Template variables:
#   {{ outputs }}      — the model's response
#   {{ expectations }} — the full expectations dict as string
# =============================================================================

def _build_custom_judges(model: str) -> list:
    """Build custom LLM-as-Judge scorers using MLflow's make_judge API.

    Args:
        model: The judge model URI (e.g. "bedrock:/us.anthropic.claude-...").

    Returns:
        A list of three custom judge scorers:
        - factual_consistency: Checks if the response stays within the context
        - professionalism: Evaluates tone and formality
        - correctness: Compares response against expected answer
    """
    factual_consistency = make_judge(
        name="factual_consistency",
        model=model,
        instructions=(
            "Evaluate whether the response is factually consistent with the provided context. "
            "The response should not introduce facts that contradict or are absent from the context.\n\n"
            "Response: {{ outputs }}"
        ),
    )

    professionalism = make_judge(
        name="professionalism",
        model=model,
        instructions=(
            "Evaluate the professionalism of the response. A professional response uses "
            "formal language, avoids slang, is well-structured, and maintains a neutral tone.\n\n"
            "Response: {{ outputs }}"
        ),
    )

    correctness = make_judge(
        name="correctness",
        model=model,
        instructions=(
            "Evaluate whether the response is factually correct based on the expected answer.\n\n"
            "Response: {{ outputs }}\n\n"
            "Expected answer: {{ expectations }}"
        ),
    )

    return [factual_consistency, professionalism, correctness]


# =============================================================================
# DeepEval scorers (using Bedrock as judge via MLflow integration)
#
# DeepEval provides advanced evaluation metrics, particularly for RAG systems.
# MLflow's DeepEvalScorer wrapper integrates them into mlflow.genai.evaluate()
# so all results appear in a single MLflow run. The model= parameter tells
# DeepEval to use Bedrock (via litellm) instead of OpenAI.
#
# Key advantage over MLflow's built-in scorers: DeepEval decomposes responses
# into individual claims and verifies each one, providing more granular
# evaluation than single yes/no judgments.
# =============================================================================

def _build_deepeval_scorers() -> list:
    """Build DeepEval scorers wrapped for MLflow, using Bedrock as the judge model.

    Uses the same judge model URI as MLflow scorers (config.JUDGE_MODEL).
    DeepEval resolves the "bedrock:/" prefix via litellm to make Bedrock API calls.

    Returns:
        A list of six DeepEvalScorer instances:
        - Faithfulness: Claim-level verification against context
        - AnswerRelevancy: RAGAS-style relevancy scoring
        - ContextualRelevancy: Evaluates retrieval quality
        - Hallucination: Detects fabricated information
        - Toxicity: Flags harmful or toxic content
        - Bias: Detects gender, racial, or other biases
    """
    judge_model = config.JUDGE_MODEL

    metric_names = [
        "Faithfulness",
        "AnswerRelevancy",
        "ContextualRelevancy",
        "Hallucination",
        "Toxicity",
        "Bias",
    ]

    return [DeepEvalScorer(metric_name=name, model=judge_model) for name in metric_names]


# =============================================================================
# Agent tools
#
# These are Strands @tool-decorated functions that the LLMEvaluatorAgent can
# call. The agent decides which tools to invoke based on the user's prompt.
# Each tool returns a string result that the agent incorporates into its response.
# =============================================================================

@tool
def load_evaluation_dataset(sample_size: int = 0) -> str:
    """Load the RAG evaluation dataset from HuggingFace.

    Downloads the explodinggradients/ragas-wikiqa dataset and extracts
    question, correct_answer, and context columns. Saves to a temporary
    JSON file that subsequent evaluation tools read from.

    Args:
        sample_size: Number of samples to load. 0 means use config default (20).
    """
    size = sample_size if sample_size > 0 else config.SAMPLE_SIZE
    ds = load_dataset(config.DATASET_NAME, split=config.DATASET_SPLIT)
    df = ds.to_pandas().head(size)
    eval_df = pd.DataFrame({
        "inputs": df["question"].tolist(),
        "ground_truth": df["correct_answer"].tolist(),
        "context": df["context"].tolist(),
    })
    eval_df.to_json("/tmp/eval_dataset.json", orient="records", indent=2)
    return f"Loaded {len(eval_df)} samples from {config.DATASET_NAME}. Columns: {list(eval_df.columns)}. Saved to /tmp/eval_dataset.json"


@tool
def run_bedrock_evaluation(model_key: str) -> str:
    """Run a comprehensive LLM evaluation using a Bedrock model as the generator.

    This is the core evaluation tool. It:
    1. Reads the dataset from /tmp/eval_dataset.json (loaded by load_evaluation_dataset)
    2. Creates a predict_fn that calls the specified Bedrock model via the Converse API
    3. Assembles 18 scorers across 4 categories (MLflow built-in, custom judges,
       DeepEval, and code-based)
    4. Runs mlflow.genai.evaluate() which:
       - Calls predict_fn for each sample to generate model responses
       - Runs all 18 scorers against each response
       - Logs metrics, traces, and assessments to MLflow
    5. Logs CSV and JSON artifacts with evaluation results

    Args:
        model_key: Key from config.BEDROCK_MODELS (e.g. 'claude-sonnet-4-5', 'claude-haiku-4-5').
    """
    if model_key not in config.BEDROCK_MODELS:
        return f"Unknown model_key '{model_key}'. Available: {list(config.BEDROCK_MODELS.keys())}"

    _ensure_aws_env_vars()

    model_id = config.BEDROCK_MODELS[model_key]
    eval_df = pd.read_json("/tmp/eval_dataset.json")
    bedrock = config.get_boto_session().client("bedrock-runtime")
    os.environ["AWS_DEFAULT_REGION"] = config.AWS_REGION

    def predict_fn(question: str, context: str = "") -> str:
        """Call the Bedrock model to generate an answer for a given question and context.

        This function is passed to mlflow.genai.evaluate() which calls it once per
        sample in the dataset. The parameter names (question, context) must match
        the keys in eval_data[*]["inputs"].
        """
        prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer concisely."
        try:
            response = bedrock.converse(
                modelId=model_id,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": 512, "temperature": 0.1},
            )
            return response["output"]["message"]["content"][0]["text"]
        except Exception as e:
            return f"ERROR: {e}"

    # Build eval dataset in the format required by mlflow.genai.evaluate():
    #   inputs: dict matching predict_fn parameter names
    #   expectations: dict consumed by scorers (expected_response for Equivalence,
    #                 exact_match, word_overlap; expected_facts for Correctness)
    eval_data = [
        {
            "inputs": {"question": row["inputs"], "context": row.get("context", "")},
            "expectations": {
                "expected_response": row["ground_truth"],
                "expected_facts": _extract_facts(row["ground_truth"]),
            },
        }
        for _, row in eval_df.iterrows()
    ]

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.EXPERIMENT_NAME)

    judge_model = config.JUDGE_MODEL
    scorers_list = [
        # === MLflow Built-in LLM-as-Judge ===
        RelevanceToQuery(model=judge_model),
        Equivalence(model=judge_model),
        Fluency(model=judge_model),
        # === MLflow Guidelines-based ===
        Guidelines(
            name="answer_groundedness",
            guidelines="The answer must be grounded in the provided context and not hallucinate facts.",
            model=judge_model,
        ),
        Guidelines(
            name="answer_completeness",
            guidelines="The answer must fully address the question without omitting key information.",
            model=judge_model,
        ),
        # === Custom LLM-as-Judge (via make_judge) ===
        *_build_custom_judges(judge_model),
        # === DeepEval scorers (using Bedrock via litellm) ===
        *_build_deepeval_scorers(),
        # === Code-based scorers ===
        exact_match,
        is_concise,
        word_overlap,
        response_length,
    ]

    with mlflow.start_run(run_name=f"eval-{model_key}"):
        # Log parameters for reproducibility and filtering in MLflow UI
        mlflow.log_param("model_id", model_id)
        mlflow.log_param("model_key", model_key)
        mlflow.log_param("dataset", config.DATASET_NAME)
        mlflow.log_param("sample_size", len(eval_df))

        # Run the evaluation — this calls predict_fn for each sample,
        # then runs all 18 scorers against each response
        results = mlflow.genai.evaluate(
            data=eval_data,
            predict_fn=predict_fn,
            scorers=scorers_list,
        )

        # Log evaluation results as a CSV artifact for easy download
        results_df = pd.DataFrame([
            {
                "question": d["inputs"]["question"],
                "ground_truth": d["expectations"]["expected_response"],
                "context": d["inputs"].get("context", "")[:200],
            }
            for d in eval_data
        ])
        traces = mlflow.search_traces(
            experiment_ids=[mlflow.get_experiment_by_name(config.EXPERIMENT_NAME).experiment_id],
            filter_string=f"run_id = '{mlflow.active_run().info.run_id}'",
        )
        if not traces.empty and "response" in traces.columns:
            results_df["prediction"] = traces["response"].tolist()[:len(results_df)]
        results_df.to_csv("/tmp/eval_results.csv", index=False)
        mlflow.log_artifact("/tmp/eval_results.csv", artifact_path="eval_tables")

        # Log aggregated metrics as a JSON artifact
        metrics = {k: round(v, 4) if isinstance(v, float) else v for k, v in results.metrics.items()}
        with open("/tmp/metrics_summary.json", "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        mlflow.log_artifact("/tmp/metrics_summary.json", artifact_path="eval_tables")

        run_id = mlflow.active_run().info.run_id
        metrics_str = json.dumps(metrics, indent=2, default=str)
        return f"Evaluation complete for {model_key} (run_id={run_id}).\nMetrics:\n{metrics_str}"


@tool
def run_all_evaluations() -> str:
    """Run evaluations across all Bedrock models defined in config.BEDROCK_MODELS.

    Iterates over each model key and calls run_bedrock_evaluation for each one.
    Each model gets its own MLflow run, making it easy to compare results
    side-by-side in the MLflow UI.
    """
    results = {}
    for model_key in config.BEDROCK_MODELS:
        result = run_bedrock_evaluation.tool_function(model_key=model_key)
        results[model_key] = result
    return f"All evaluations complete. Models evaluated: {list(config.BEDROCK_MODELS.keys())}"


@tool
def get_experiment_summary() -> str:
    """Query MLflow for a summary of all evaluation runs in the experiment.

    Returns a table of all runs with their metrics and parameters, ordered
    by most recent first. Useful for comparing model performance across runs.
    """
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    experiment = mlflow.get_experiment_by_name(config.EXPERIMENT_NAME)
    if not experiment:
        return "No experiment found. Run evaluations first."

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"]
    )
    if runs.empty:
        return "No runs found in the experiment."

    summary_cols = [
        c for c in runs.columns
        if c.startswith("metrics.") or c.startswith("params.") or c in ["run_id", "status", "start_time"]
    ]
    summary = runs[summary_cols].to_string()
    return f"Experiment: {config.EXPERIMENT_NAME}\nTotal runs: {len(runs)}\n\n{summary}"


@tool
def generate_eval_report(model_key: str) -> str:
    """Generate an EMRM evaluation report document for a model's latest evaluation run.

    Reads the latest MLflow run for the given model, extracts all metrics,
    and fills the EMRM template to produce a .docx report in the output/ folder.

    Args:
        model_key: Key from config BEDROCK_MODELS dict (e.g. 'claude-sonnet-4-5').
    """
    from src.report_generator import generate_llm_eval_report

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    experiment = mlflow.get_experiment_by_name(config.EXPERIMENT_NAME)
    if not experiment:
        return "No experiment found. Run evaluations first."

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"params.model_key = '{model_key}'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        return f"No runs found for model_key '{model_key}'."

    run = runs.iloc[0]
    run_id = run["run_id"]
    sample_size = int(run.get("params.sample_size", 0))

    # Extract all metrics
    metrics = {
        col.replace("metrics.", ""): run[col]
        for col in runs.columns if col.startswith("metrics.") and pd.notna(run[col])
    }

    filepath = generate_llm_eval_report(model_key, metrics, sample_size, run_id)
    return f"Report generated: {filepath}"
