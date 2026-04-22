# LLM Evaluation Agent

An agent-based LLM evaluation system built with [Strands Agents SDK](https://strandsagents.com/) that evaluates multiple Bedrock models using open-source datasets and tracks everything in MLflow on SageMaker.

## Architecture

```
HuggingFace Dataset ──> Strands Agent ──> Bedrock Models (Claude Sonnet 4.5, Haiku 4.5)
                            │
                            ▼
                     MLflow App on SageMaker (Serverless)
                     (Metrics, Artifacts, Dashboard)
```

## Project Structure

```
rbc-evals-demo/
├── infra/                          # Terraform (KMS, S3, IAM, SageMaker, MLflow App, SSM)
│   ├── main.tf
│   ├── variables.tf
│   └── outputs.tf
├── src/                            # Python source modules
│   ├── config.py                   # Configuration (models, dataset, MLflow URI, IAM)
│   ├── tools.py                    # Agent tools + 18 evaluation scorers
│   └── report_generator.py         # EMRM template → .docx report generator
├── notebooks/                      # Jupyter notebooks (walkthrough / demo)
│   ├── llm_evaluator_agent.ipynb   # LLM output evaluation notebook
│   └── agent_behavior_eval.ipynb   # Agent behavior evaluation notebook
├── templates/                      # Document templates
│   └── emrm_eval_template.docx     # EMRM evaluation report template
├── output/                         # Generated evaluation reports (.docx)
├── llm_evaluator_agent.py          # CLI: Strands evaluator agent entry point
├── agent_behavior_eval.py          # CLI: Strands Evals agent behavior evaluation
├── requirements.txt
└── README.md
```

## Prerequisites

- AWS account with Bedrock model access (Claude, Titan)
- Terraform >= 1.5
- Python >= 3.10
- AWS CLI configured

## Setup

### 1. Deploy Infrastructure

Terraform creates an S3 artifact bucket, IAM role, serverless MLflow App, and stores all details as SSM parameters.

```bash
cd infra
terraform init
terraform plan
terraform apply
```

All infra values are stored under the SSM prefix `/llm-eval/` — the agent reads these automatically.

### 2. Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
export AWS_REGION="ca-central-1"
export SSM_PREFIX="/llm-eval"   # optional, matches Terraform default
export SAMPLE_SIZE=20           # optional, default 20
```

The agent auto-discovers the MLflow App ARN from SSM and uses it as the tracking URI via the `sagemaker-mlflow` plugin. You can override it:

```bash
export MLFLOW_TRACKING_URI="<mlflow-app-arn-or-url>"
```

### 4. Run the Agent

```bash
# Default: loads dataset, evaluates all models, summarizes
python llm_evaluator_agent.py

# Custom prompt
python llm_evaluator_agent.py "Load 10 samples and evaluate only claude-sonnet-4-5"

# With report generation
python llm_evaluator_agent.py "Load 5 samples, evaluate claude-sonnet-4-5, and generate the evaluation report"
```

### 5. Run Agent Behavior Evaluation

```bash
python agent_behavior_eval.py
```

### 6. Run Notebooks

```bash
jupyter lab notebooks/
```

## Dataset

Uses [explodinggradients/ragas-wikiqa](https://huggingface.co/datasets/explodinggradients/ragas-wikiqa) from HuggingFace, which contains:
- `question` - Input questions
- `correct_answer` - Ground truth answers
- `context` - Retrieved context passages
- `generated_with_rag` / `generated_without_rag` - Pre-generated predictions

## Evaluation Metrics (18 scorers)

**MLflow Built-in LLM-as-Judge (5):** RelevanceToQuery, Equivalence, Fluency, Groundedness, Completeness

**Custom LLM-as-Judge via make_judge (3):** Correctness, Factual Consistency, Professionalism

**DeepEval via Bedrock (6):** Faithfulness, AnswerRelevancy, ContextualRelevancy, Hallucination, Toxicity, Bias

**Code-based (4):** Exact Match, Conciseness, Word Overlap, Response Length

## Agent Behavior Evaluation (Strands Evals)

**OutputEvaluator** — Is the agent's response accurate and helpful?
**TrajectoryEvaluator** — Did the agent call the right tools in the right order?
**ToolSelectionAccuracy** — Did the agent pick the correct tools?
**HelpfulnessEvaluator** — Was the agent genuinely helpful? (trace-based)
**FaithfulnessEvaluator** — Did the agent stay faithful to tool outputs? (trace-based)

## MLflow Dashboard

Access the MLflow UI through SageMaker Studio or via a presigned URL:

```bash
aws sagemaker create-presigned-mlflow-tracking-server-url \
  --tracking-server-name llm-eval-mlflow \
  --region ca-central-1
```

## Cleanup

```bash
cd infra
terraform destroy
```
