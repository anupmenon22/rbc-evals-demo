terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

data "aws_caller_identity" "current" {}
data "aws_availability_zones" "available" { state = "available" }

locals {
  account_id      = data.aws_caller_identity.current.account_id
  mlflow_app_name = "${var.project_name}-mlflow"
  bucket_name     = "${var.project_name}-mlflow-artifacts-${local.account_id}"
}

# =============================================================================
# KMS CMK for S3 encryption
# =============================================================================

resource "aws_kms_key" "s3" {
  description             = "CMK for ${var.project_name} MLflow artifact bucket"
  deletion_window_in_days = 7
  enable_key_rotation     = true

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid       = "RootAccountFullAccess"
        Effect    = "Allow"
        Principal = { AWS = "arn:aws:iam::${local.account_id}:root" }
        Action    = "kms:*"
        Resource  = "*"
      },
      {
        Sid    = "AllowSageMakerMLflowUse"
        Effect = "Allow"
        Principal = {
          AWS = [
            aws_iam_role.mlflow_execution.arn,
            aws_iam_role.agent_execution.arn,
          ]
        }
        Action = [
          "kms:Decrypt",
          "kms:GenerateDataKey",
          "kms:DescribeKey",
        ]
        Resource = "*"
      }
    ]
  })

  tags = { Project = var.project_name }
}

resource "aws_kms_alias" "s3" {
  name          = "alias/${var.project_name}-s3"
  target_key_id = aws_kms_key.s3.key_id
}

# =============================================================================
# Minimal VPC for SageMaker Domain (PublicInternetOnly — no NAT needed)
# =============================================================================

resource "aws_vpc" "this" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = { Name = "${var.project_name}-vpc", Project = var.project_name }
}

resource "aws_subnet" "private" {
  vpc_id            = aws_vpc.this.id
  cidr_block        = "10.0.1.0/24"
  availability_zone = data.aws_availability_zones.available.names[0]

  tags = { Name = "${var.project_name}-subnet", Project = var.project_name }
}

# =============================================================================
# S3 bucket for MLflow artifacts — encrypted with KMS CMK
# =============================================================================

resource "aws_s3_bucket" "mlflow_artifacts" {
  bucket        = local.bucket_name
  force_destroy = true

  tags = { Project = var.project_name }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "mlflow_artifacts" {
  bucket = aws_s3_bucket.mlflow_artifacts.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = "aws:kms"
      kms_master_key_id = aws_kms_key.s3.arn
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_public_access_block" "mlflow_artifacts" {
  bucket = aws_s3_bucket.mlflow_artifacts.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# =============================================================================
# IAM Role: MLflow App execution (used by SageMaker MLflow App + Domain)
# Least privilege: only S3 artifact access + KMS + minimal SageMaker/logs
# =============================================================================

resource "aws_iam_role" "mlflow_execution" {
  name = "${var.project_name}-mlflow-execution"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "sagemaker.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })

  tags = { Project = var.project_name }
}

resource "aws_iam_role_policy" "mlflow_execution" {
  name = "mlflow-execution-policy"
  role = aws_iam_role.mlflow_execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "S3ArtifactAccess"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket",
          "s3:GetBucketLocation",
        ]
        Resource = [
          aws_s3_bucket.mlflow_artifacts.arn,
          "${aws_s3_bucket.mlflow_artifacts.arn}/*",
        ]
      },
      {
        Sid    = "KMSDecryptEncrypt"
        Effect = "Allow"
        Action = [
          "kms:Decrypt",
          "kms:GenerateDataKey",
          "kms:DescribeKey",
        ]
        Resource = [aws_kms_key.s3.arn]
      },
      {
        Sid    = "CloudWatchLogs"
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
        ]
        Resource = "arn:aws:logs:${var.aws_region}:${local.account_id}:log-group:/aws/sagemaker/*"
      },
      {
        Sid    = "SageMakerStudioAccess"
        Effect = "Allow"
        Action = [
          "sagemaker:CreateApp",
          "sagemaker:DeleteApp",
          "sagemaker:DescribeApp",
          "sagemaker:DescribeDomain",
          "sagemaker:DescribeUserProfile",
          "sagemaker:DescribeMlflowApp",
          "sagemaker:ListMlflowApps",
          "sagemaker:ListApps",
          "sagemaker:ListSpaces",
          "sagemaker:ListDomains",
          "sagemaker:ListUserProfiles",
          "sagemaker:ListTags",
          "sagemaker:DescribeSpace",
          "sagemaker:CreatePresignedDomainUrl",
          "sagemaker:CreatePresignedMlflowTrackingServerUrl",
          "sagemaker:CreatePresignedMlflowAppUrl",
          "sagemaker-mlflow:*",
        ]
        Resource = "*"
      },
    ]
  })
}

# =============================================================================
# IAM Role: Agent execution (used by the Python evaluation agent at runtime)
# Least privilege: Bedrock invoke, SSM read, MLflow tracking, S3 artifacts
# =============================================================================

resource "aws_iam_role" "agent_execution" {
  name = "${var.project_name}-agent-execution"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { AWS = "arn:aws:iam::${local.account_id}:root" }
      Action    = "sts:AssumeRole"
    }]
  })

  tags = { Project = var.project_name }
}

resource "aws_iam_role_policy" "agent_execution" {
  name = "agent-execution-policy"
  role = aws_iam_role.agent_execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "BedrockInvoke"
        Effect = "Allow"
        Action = [
          "bedrock:InvokeModel",
          "bedrock:InvokeModelWithResponseStream",
        ]
        Resource = [
          "arn:aws:bedrock:*::foundation-model/*",
          "arn:aws:bedrock:*:${local.account_id}:inference-profile/*",
        ]
      },
      {
        Sid    = "SSMReadParameters"
        Effect = "Allow"
        Action = [
          "ssm:GetParameter",
          "ssm:GetParametersByPath",
        ]
        Resource = "arn:aws:ssm:${var.aws_region}:${local.account_id}:parameter/${var.project_name}/*"
      },
      {
        Sid    = "SageMakerMLflowTracking"
        Effect = "Allow"
        Action = [
          "sagemaker-mlflow:*",
          "sagemaker:*",
        ]
        Resource = "arn:aws:sagemaker:${var.aws_region}:${local.account_id}:mlflow-app/*"
      },
      {
        Sid    = "S3ArtifactAccess"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket",
          "s3:GetBucketLocation",
        ]
        Resource = [
          aws_s3_bucket.mlflow_artifacts.arn,
          "${aws_s3_bucket.mlflow_artifacts.arn}/*",
        ]
      },
      {
        Sid    = "KMSDecryptEncrypt"
        Effect = "Allow"
        Action = [
          "kms:Decrypt",
          "kms:GenerateDataKey",
          "kms:DescribeKey",
        ]
        Resource = [aws_kms_key.s3.arn]
      },
    ]
  })
}

# =============================================================================
# IAM Policy: Terraform deployer (attach to the admin user/role running TF)
# Least privilege for deploying this specific infrastructure
# =============================================================================

resource "aws_iam_policy" "deployer" {
  name        = "${var.project_name}-deployer"
  description = "Least-privilege policy for deploying ${var.project_name} infrastructure via Terraform"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "VPCAndNetworking"
        Effect = "Allow"
        Action = [
          "ec2:CreateVpc", "ec2:DeleteVpc", "ec2:DescribeVpcs", "ec2:ModifyVpcAttribute",
          "ec2:CreateSubnet", "ec2:DeleteSubnet", "ec2:DescribeSubnets",
          "ec2:DescribeAvailabilityZones",
          "ec2:DescribeNetworkInterfaces",
          "ec2:CreateTags", "ec2:DeleteTags", "ec2:DescribeTags",
        ]
        Resource = "*"
      },
      {
        Sid    = "S3BucketManagement"
        Effect = "Allow"
        Action = [
          "s3:CreateBucket", "s3:DeleteBucket", "s3:ListBucket",
          "s3:GetBucketLocation", "s3:GetBucketPolicy", "s3:PutBucketPolicy",
          "s3:GetBucketTagging", "s3:PutBucketTagging",
          "s3:GetEncryptionConfiguration", "s3:PutEncryptionConfiguration",
          "s3:GetBucketPublicAccessBlock", "s3:PutBucketPublicAccessBlock",
          "s3:GetBucketVersioning", "s3:GetBucketAcl",
          "s3:GetAccelerateConfiguration", "s3:GetBucketCORS",
          "s3:GetBucketLogging", "s3:GetBucketObjectLockConfiguration",
          "s3:GetBucketRequestPayment", "s3:GetBucketWebsite",
          "s3:GetLifecycleConfiguration", "s3:GetReplicationConfiguration",
          "s3:ListBucketVersions",
          "s3:DeleteObject", "s3:DeleteObjectVersion",
        ]
        Resource = [
          "arn:aws:s3:::${var.project_name}-mlflow-artifacts-*",
          "arn:aws:s3:::${var.project_name}-mlflow-artifacts-*/*",
        ]
      },
      {
        Sid    = "KMSKeyManagement"
        Effect = "Allow"
        Action = [
          "kms:CreateKey", "kms:DescribeKey", "kms:GetKeyPolicy", "kms:GetKeyRotationStatus",
          "kms:ListResourceTags", "kms:TagResource", "kms:UntagResource",
          "kms:EnableKeyRotation", "kms:ScheduleKeyDeletion",
          "kms:CreateAlias", "kms:DeleteAlias", "kms:ListAliases",
          "kms:PutKeyPolicy",
        ]
        Resource = "*"
      },
      {
        Sid    = "IAMRoleManagement"
        Effect = "Allow"
        Action = [
          "iam:CreateRole", "iam:DeleteRole", "iam:GetRole", "iam:ListRolePolicies",
          "iam:ListAttachedRolePolicies", "iam:ListInstanceProfilesForRole",
          "iam:PutRolePolicy", "iam:DeleteRolePolicy", "iam:GetRolePolicy",
          "iam:TagRole", "iam:UntagRole",
          "iam:PassRole",
          "iam:CreatePolicy", "iam:DeletePolicy", "iam:GetPolicy",
          "iam:GetPolicyVersion", "iam:ListPolicyVersions",
          "iam:CreatePolicyVersion", "iam:DeletePolicyVersion",
          "iam:TagPolicy", "iam:UntagPolicy",
        ]
        Resource = [
          "arn:aws:iam::${local.account_id}:role/${var.project_name}-*",
          "arn:aws:iam::${local.account_id}:policy/${var.project_name}-*",
        ]
      },
      {
        Sid    = "SageMakerDomainAndMLflow"
        Effect = "Allow"
        Action = [
          "sagemaker:CreateDomain", "sagemaker:DeleteDomain", "sagemaker:DescribeDomain",
          "sagemaker:UpdateDomain", "sagemaker:ListDomains",
          "sagemaker:CreateUserProfile", "sagemaker:DeleteUserProfile",
          "sagemaker:DescribeUserProfile", "sagemaker:UpdateUserProfile",
          "sagemaker:CreateMlflowApp", "sagemaker:DeleteMlflowApp",
          "sagemaker:DescribeMlflowApp", "sagemaker:ListMlflowApps",
          "sagemaker:AddTags", "sagemaker:ListTags", "sagemaker:DeleteTags",
        ]
        Resource = [
          "arn:aws:sagemaker:${var.aws_region}:${local.account_id}:domain/*",
          "arn:aws:sagemaker:${var.aws_region}:${local.account_id}:user-profile/*/*",
          "arn:aws:sagemaker:${var.aws_region}:${local.account_id}:mlflow-app/*",
        ]
      },
      {
        Sid    = "SSMParameterManagement"
        Effect = "Allow"
        Action = [
          "ssm:PutParameter", "ssm:GetParameter", "ssm:GetParameters",
          "ssm:GetParametersByPath", "ssm:DeleteParameter",
          "ssm:AddTagsToResource", "ssm:RemoveTagsFromResource",
          "ssm:ListTagsForResource",
        ]
        Resource = "arn:aws:ssm:${var.aws_region}:${local.account_id}:parameter/${var.project_name}/*"
      },
      {
        Sid    = "EFSCleanup"
        Effect = "Allow"
        Action = [
          "elasticfilesystem:DescribeMountTargets",
          "elasticfilesystem:DeleteMountTarget",
          "elasticfilesystem:DeleteFileSystem",
          "elasticfilesystem:DescribeFileSystems",
        ]
        Resource = "*"
      },
      {
        Sid      = "STSGetCallerIdentity"
        Effect   = "Allow"
        Action   = ["sts:GetCallerIdentity"]
        Resource = "*"
      },
    ]
  })

  tags = { Project = var.project_name }
}

# =============================================================================
# MLflow App (serverless) — no native TF resource, use AWS CLI via provisioner
# =============================================================================

resource "terraform_data" "mlflow_app" {
  depends_on = [aws_iam_role_policy.mlflow_execution, aws_sagemaker_domain.this]

  input = local.mlflow_app_name

  provisioner "local-exec" {
    command = <<-EOT
      APP_ARN=$(aws sagemaker create-mlflow-app \
        --name ${local.mlflow_app_name} \
        --artifact-store-uri s3://${aws_s3_bucket.mlflow_artifacts.bucket}/mlflow \
        --role-arn ${aws_iam_role.mlflow_execution.arn} \
        --default-domain-id-list ${aws_sagemaker_domain.this.id} \
        --tags Key=Project,Value=${var.project_name} \
        --region ${var.aws_region} \
        --query 'Arn' --output text)
      echo "MLflow App ARN: $APP_ARN"
      aws ssm put-parameter \
        --name "/${var.project_name}/mlflow/app-arn" \
        --value "$APP_ARN" \
        --type String --overwrite \
        --region ${var.aws_region}
    EOT
  }

  provisioner "local-exec" {
    when    = destroy
    command = <<-EOT
      APP_ARN=$(aws sagemaker list-mlflow-apps \
        --region ca-central-1 \
        --query 'Summaries[?Name==`'${self.input}'` && Status!=`Deleted`].Arn' \
        --output text 2>/dev/null)
      if [ -n "$APP_ARN" ] && [ "$APP_ARN" != "None" ]; then
        echo "Deleting MLflow App: $APP_ARN"
        aws sagemaker delete-mlflow-app --arn "$APP_ARN" --region ca-central-1 || true
      fi
    EOT
  }
}

# =============================================================================
# SageMaker Domain + User Profile (for Studio UI access to MLflow)
# =============================================================================

resource "aws_sagemaker_domain" "this" {
  domain_name = var.project_name
  auth_mode   = "IAM"
  vpc_id      = aws_vpc.this.id
  subnet_ids  = [aws_subnet.private.id]

  default_user_settings {
    execution_role = aws_iam_role.mlflow_execution.arn
  }

  provisioner "local-exec" {
    when    = destroy
    command = <<-EOT
      echo "Cleaning up EFS mount targets for ${self.home_efs_file_system_id}..."
      MTS=$(aws efs describe-mount-targets \
        --file-system-id ${self.home_efs_file_system_id} \
        --query 'MountTargets[].MountTargetId' --output text \
        --region ${self.region} 2>/dev/null)
      for MT in $MTS; do
        echo "Deleting mount target $MT"
        aws efs delete-mount-target --mount-target-id $MT --region ${self.region} 2>/dev/null || true
      done
      if [ -n "$MTS" ]; then
        echo "Waiting for ENIs to detach..."
        sleep 90
      fi
      echo "Deleting EFS filesystem ${self.home_efs_file_system_id}..."
      aws efs delete-file-system --file-system-id ${self.home_efs_file_system_id} --region ${self.region} 2>/dev/null || true
      echo "Cleaning up NFS security groups..."
      for SG in $(aws ec2 describe-security-groups \
        --filters "Name=vpc-id,Values=${self.vpc_id}" "Name=group-name,Values=security-group-for-*-nfs-*" \
        --query 'SecurityGroups[].GroupId' --output text --region ${self.region} 2>/dev/null); do
        # Revoke all ingress/egress rules first
        for RULE in $(aws ec2 describe-security-group-rules --filters "Name=group-id,Values=$SG" \
          --query 'SecurityGroupRules[?!IsEgress].SecurityGroupRuleId' --output text --region ${self.region} 2>/dev/null); do
          aws ec2 revoke-security-group-ingress --group-id $SG --security-group-rule-ids $RULE --region ${self.region} 2>/dev/null || true
        done
        for RULE in $(aws ec2 describe-security-group-rules --filters "Name=group-id,Values=$SG" \
          --query 'SecurityGroupRules[?IsEgress].SecurityGroupRuleId' --output text --region ${self.region} 2>/dev/null); do
          aws ec2 revoke-security-group-egress --group-id $SG --security-group-rule-ids $RULE --region ${self.region} 2>/dev/null || true
        done
      done
      for SG in $(aws ec2 describe-security-groups \
        --filters "Name=vpc-id,Values=${self.vpc_id}" "Name=group-name,Values=security-group-for-*-nfs-*" \
        --query 'SecurityGroups[].GroupId' --output text --region ${self.region} 2>/dev/null); do
        echo "Deleting security group $SG"
        aws ec2 delete-security-group --group-id $SG --region ${self.region} 2>/dev/null || true
      done
      echo "EFS cleanup complete"
    EOT
  }

  tags = { Project = var.project_name }
}

resource "aws_sagemaker_user_profile" "default" {
  domain_id         = aws_sagemaker_domain.this.id
  user_profile_name = "${var.project_name}-user"

  user_settings {
    execution_role = aws_iam_role.mlflow_execution.arn
  }

  tags = { Project = var.project_name }
}

# =============================================================================
# SSM Parameters — infra details for the agent to consume
# =============================================================================

resource "aws_ssm_parameter" "mlflow_app_arn" {
  name  = "/${var.project_name}/mlflow/app-arn"
  type  = "String"
  value = "pending-mlflow-app-creation"

  tags = { Project = var.project_name }

  lifecycle {
    ignore_changes = [value]
  }
}

resource "aws_ssm_parameter" "mlflow_app_name" {
  name  = "/${var.project_name}/mlflow/app-name"
  type  = "String"
  value = local.mlflow_app_name

  tags = { Project = var.project_name }
}

resource "aws_ssm_parameter" "mlflow_artifact_bucket" {
  name  = "/${var.project_name}/mlflow/artifact-bucket"
  type  = "String"
  value = aws_s3_bucket.mlflow_artifacts.bucket

  tags = { Project = var.project_name }
}

resource "aws_ssm_parameter" "kms_key_arn" {
  name  = "/${var.project_name}/security/kms-key-arn"
  type  = "String"
  value = aws_kms_key.s3.arn

  tags = { Project = var.project_name }
}

resource "aws_ssm_parameter" "kms_key_alias" {
  name  = "/${var.project_name}/security/kms-key-alias"
  type  = "String"
  value = aws_kms_alias.s3.name

  tags = { Project = var.project_name }
}

resource "aws_ssm_parameter" "aws_region" {
  name  = "/${var.project_name}/config/aws-region"
  type  = "String"
  value = var.aws_region

  tags = { Project = var.project_name }
}

resource "aws_ssm_parameter" "sagemaker_domain_id" {
  name  = "/${var.project_name}/sagemaker/domain-id"
  type  = "String"
  value = aws_sagemaker_domain.this.id

  tags = { Project = var.project_name }
}

resource "aws_ssm_parameter" "agent_role_arn" {
  name  = "/${var.project_name}/security/agent-role-arn"
  type  = "String"
  value = aws_iam_role.agent_execution.arn

  tags = { Project = var.project_name }
}
