variable "project_name" {
  description = "Project name used for resource naming"
  type        = string
  default     = "llm-eval"
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "ca-central-1"
}
