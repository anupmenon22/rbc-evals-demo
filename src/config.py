"""
Configuration module for the LLM Evaluation Agent.

This module centralizes all configuration used across the evaluation pipeline.
It handles four concerns:

1. **AWS & Region Settings**: Defaults to ca-central-1. Overridable via AWS_REGION env var.

2. **IAM Role Assumption**: When AGENT_ROLE_ARN env var is set (or auto-discovered from
   SSM), the agent assumes the least-privilege agent-execution role instead of using
   the caller's default credentials. This ensures the agent only has permissions to
   invoke Bedrock models, read SSM parameters, and write to MLflow.

3. **Bedrock Model Definitions**: Maps friendly model keys (e.g. "claude-sonnet-4-5") to
   their US cross-region inference profile IDs. These profiles are available in ca-central-1
   and route requests across US + Canada regions for better throughput.

4. **MLflow Tracking URI Resolution**: Automatically discovers the serverless MLflow App
   ARN from AWS SSM Parameter Store (written by Terraform during infra setup). The ARN
   is used directly as the tracking URI via the sagemaker-mlflow plugin, which handles
   SigV4 authentication transparently. Resolution order:
     - MLFLOW_TRACKING_URI env var (manual override)
     - SSM parameter /{prefix}/mlflow/app-arn (auto-discovery)
     - http://localhost:5000 (fallback for local development)
"""

import os
import boto3

# ---------------------------------------------------------------------------
# AWS region and SSM prefix (matches Terraform defaults in infra/)
# ---------------------------------------------------------------------------
AWS_REGION = os.environ.get("AWS_REGION", "ca-central-1")
SSM_PREFIX = os.environ.get("SSM_PREFIX", "/llm-eval")

# ---------------------------------------------------------------------------
# MLflow experiment settings
# ---------------------------------------------------------------------------
EXPERIMENT_NAME = "llm-evaluation"

# ---------------------------------------------------------------------------
# HuggingFace dataset: explodinggradients/ragas-wikiqa
# Contains: question, correct_answer, context, generated_with_rag, generated_without_rag
# ---------------------------------------------------------------------------
DATASET_NAME = "explodinggradients/ragas-wikiqa"
DATASET_SPLIT = "train"
SAMPLE_SIZE = int(os.environ.get("SAMPLE_SIZE", "20"))

# ---------------------------------------------------------------------------
# Bedrock models to evaluate
# Keys are friendly names used by the agent; values are inference profile IDs.
# US cross-region profiles (us.*) route across us-east-1, us-east-2, us-west-2,
# and ca-central-1 for higher throughput and availability.
# ---------------------------------------------------------------------------
BEDROCK_MODELS = {
    "claude-sonnet-4-5": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "claude-haiku-4-5": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
}

# ---------------------------------------------------------------------------
# Judge model for LLM-as-Judge evaluation metrics
# Used by MLflow built-in scorers, custom make_judge scorers, and DeepEval.
# The "bedrock:/" prefix tells MLflow/DeepEval to use the Bedrock adapter.
# ---------------------------------------------------------------------------
JUDGE_MODEL = "bedrock:/us.anthropic.claude-sonnet-4-5-20250929-v1:0"
JUDGE_PARAMS = {"temperature": 0, "max_tokens": 512, "anthropic_version": "bedrock-2023-05-31"}


def _get_ssm_param(name):
    """Fetch a single SSM parameter value. Returns None on any error."""
    ssm = boto3.client("ssm", region_name=AWS_REGION)
    try:
        return ssm.get_parameter(Name=name)["Parameter"]["Value"]
    except Exception:
        return None


def get_boto_session():
    """Get a boto3 session, optionally assuming the agent execution role.

    If AGENT_ROLE_ARN is set as an env var, assumes that role via STS.
    Otherwise tries to discover the role ARN from SSM, then falls back
    to the default credential chain (for local dev or when running as admin).
    """
    role_arn = os.environ.get("AGENT_ROLE_ARN") or _get_ssm_param(f"{SSM_PREFIX}/security/agent-role-arn")

    if role_arn:
        sts = boto3.client("sts", region_name=AWS_REGION)
        try:
            creds = sts.assume_role(
                RoleArn=role_arn,
                RoleSessionName="llm-evaluator-agent",
            )["Credentials"]
            return boto3.Session(
                aws_access_key_id=creds["AccessKeyId"],
                aws_secret_access_key=creds["SecretAccessKey"],
                aws_session_token=creds["SessionToken"],
                region_name=AWS_REGION,
            )
        except Exception:
            pass  # Fall through to default session

    return boto3.Session(region_name=AWS_REGION)


def get_mlflow_tracking_uri():
    """Resolve the MLflow tracking URI using a three-tier fallback strategy.

    1. MLFLOW_TRACKING_URI env var — for manual override or CI/CD pipelines.
    2. SSM parameter — reads the MLflow App ARN stored by Terraform. The
       sagemaker-mlflow plugin accepts the ARN directly as a tracking URI
       and handles authentication via AWS SigV4.
    3. localhost:5000 — fallback for local MLflow server during development.
    """
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if uri:
        return uri

    arn = _get_ssm_param(f"{SSM_PREFIX}/mlflow/app-arn")
    if arn:
        return arn

    return "http://localhost:5000"


# Resolved at import time so all modules share the same URI
MLFLOW_TRACKING_URI = get_mlflow_tracking_uri()
