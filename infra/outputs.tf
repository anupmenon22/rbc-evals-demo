output "mlflow_app_arn" {
  description = "MLflow App ARN (use as tracking URI) — actual value stored in SSM after creation"
  value       = aws_ssm_parameter.mlflow_app_arn.value
  sensitive   = true
}

output "mlflow_artifact_bucket" {
  description = "S3 bucket for MLflow artifacts"
  value       = aws_s3_bucket.mlflow_artifacts.bucket
}

output "kms_key_arn" {
  description = "KMS CMK ARN used for S3 encryption"
  value       = aws_kms_key.s3.arn
}

output "mlflow_execution_role_arn" {
  description = "IAM role ARN for MLflow App (SageMaker service role)"
  value       = aws_iam_role.mlflow_execution.arn
}

output "agent_execution_role_arn" {
  description = "IAM role ARN for the evaluation agent (assume this to run evaluations)"
  value       = aws_iam_role.agent_execution.arn
}

output "deployer_policy_arn" {
  description = "IAM policy ARN to attach to the Terraform deployer user/role (replaces admin)"
  value       = aws_iam_policy.deployer.arn
}

output "sagemaker_domain_id" {
  description = "SageMaker domain ID (for Studio access)"
  value       = aws_sagemaker_domain.this.id
}

output "ssm_parameter_prefix" {
  description = "SSM parameter prefix for all infra values"
  value       = "/${var.project_name}"
}
