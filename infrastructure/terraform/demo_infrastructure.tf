# Cost-Optimized Demo Infrastructure for Portfolio
# Estimated monthly cost: $10-20 (mostly Free Tier)
# Capable of demonstrating production-ready architecture

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# ===== VARIABLES =====
variable "aws_region" {
  default = "us-east-1"  # Cheapest region
}

variable "project_name" {
  default = "ecommerce-analytics-demo"
}

variable "environment" {
  default = "demo"
}

# ===== S3 STATIC WEBSITE HOSTING (Nearly Free) =====
resource "aws_s3_bucket" "portfolio_frontend" {
  bucket = "${var.project_name}-frontend-${var.environment}"
  
  tags = {
    Name        = "${var.project_name}-frontend"
    Environment = var.environment
    Cost        = "minimal"
  }
}

resource "aws_s3_bucket_website_configuration" "portfolio" {
  bucket = aws_s3_bucket.portfolio_frontend.id
  
  index_document {
    suffix = "index.html"
  }
  
  error_document {
    key = "error.html"
  }
}

resource "aws_s3_bucket_public_access_block" "portfolio" {
  bucket = aws_s3_bucket.portfolio_frontend.id
  
  block_public_acls       = false
  block_public_policy     = false
  ignore_public_acls      = false
  restrict_public_buckets = false
}

resource "aws_s3_bucket_policy" "portfolio" {
  bucket = aws_s3_bucket.portfolio_frontend.id
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid       = "PublicReadGetObject"
        Effect    = "Allow"
        Principal = "*"
        Action    = "s3:GetObject"
        Resource  = "${aws_s3_bucket.portfolio_frontend.arn}/*"
      }
    ]
  })
}

# ===== LAMBDA FOR BACKEND API (Pay per request) =====
resource "aws_lambda_function" "api_handler" {
  filename         = "lambda_deployment.zip"
  function_name    = "${var.project_name}-api"
  role            = aws_iam_role.lambda_execution.arn
  handler         = "lambda_handler.main"
  runtime         = "python3.9"
  timeout         = 30
  memory_size     = 512  # Enough for ML inference
  
  environment {
    variables = {
      ENVIRONMENT     = var.environment
      MODEL_BUCKET    = aws_s3_bucket.ml_models.id
      DEMO_MODE      = "true"
      CORS_ORIGIN    = "https://${aws_cloudfront_distribution.portfolio.domain_name}"
    }
  }
  
  layers = [aws_lambda_layer_version.ml_dependencies.arn]
}

# Lambda Layer for ML Dependencies (Reusable)
resource "aws_lambda_layer_version" "ml_dependencies" {
  filename            = "ml_layer.zip"
  layer_name          = "${var.project_name}-ml-deps"
  compatible_runtimes = ["python3.9"]
  description         = "TensorFlow, NumPy, and other ML dependencies"
}

# API Gateway for Lambda (HTTP API - Cheaper than REST)
resource "aws_apigatewayv2_api" "main" {
  name          = "${var.project_name}-api"
  protocol_type = "HTTP"
  
  cors_configuration {
    allow_origins = ["*"]  # Configure properly for production
    allow_methods = ["GET", "POST", "OPTIONS"]
    allow_headers = ["content-type", "x-api-key"]
    max_age       = 300
  }
}

resource "aws_apigatewayv2_integration" "lambda" {
  api_id             = aws_apigatewayv2_api.main.id
  integration_type   = "AWS_PROXY"
  integration_uri    = aws_lambda_function.api_handler.invoke_arn
  integration_method = "POST"
}

resource "aws_apigatewayv2_route" "main" {
  api_id    = aws_apigatewayv2_api.main.id
  route_key = "$default"
  target    = "integrations/${aws_apigatewayv2_integration.lambda.id}"
}

resource "aws_apigatewayv2_stage" "main" {
  api_id      = aws_apigatewayv2_api.main.id
  name        = "$default"
  auto_deploy = true
}

# ===== S3 FOR ML MODELS (Minimal Storage) =====
resource "aws_s3_bucket" "ml_models" {
  bucket = "${var.project_name}-models-${var.environment}"
  
  tags = {
    Name        = "${var.project_name}-models"
    Environment = var.environment
  }
}

# Lifecycle rule to keep costs down
resource "aws_s3_bucket_lifecycle_configuration" "ml_models" {
  bucket = aws_s3_bucket.ml_models.id
  
  rule {
    id     = "cleanup-old-versions"
    status = "Enabled"
    
    noncurrent_version_expiration {
      noncurrent_days = 30
    }
  }
}

# ===== DYNAMODB FOR DEMO DATA (On-Demand Pricing) =====
resource "aws_dynamodb_table" "demo_transactions" {
  name         = "${var.project_name}-transactions"
  billing_mode = "PAY_PER_REQUEST"  # Only pay for what you use
  hash_key     = "transaction_id"
  range_key    = "timestamp"
  
  attribute {
    name = "transaction_id"
    type = "S"
  }
  
  attribute {
    name = "timestamp"
    type = "N"
  }
  
  # TTL to auto-delete old demo data
  ttl {
    attribute_name = "expire_at"
    enabled        = true
  }
  
  tags = {
    Name        = "${var.project_name}-transactions"
    Environment = var.environment
  }
}

# ===== CLOUDFRONT CDN (Free Tier includes 1TB) =====
resource "aws_cloudfront_distribution" "portfolio" {
  enabled             = true
  is_ipv6_enabled     = true
  default_root_object = "index.html"
  price_class         = "PriceClass_100"  # Use only NA and EU (cheaper)
  
  origin {
    domain_name = aws_s3_bucket.portfolio_frontend.bucket_regional_domain_name
    origin_id   = "S3-${aws_s3_bucket.portfolio_frontend.id}"
    
    s3_origin_config {
      origin_access_identity = aws_cloudfront_origin_access_identity.portfolio.cloudfront_access_identity_path
    }
  }
  
  default_cache_behavior {
    allowed_methods  = ["GET", "HEAD"]
    cached_methods   = ["GET", "HEAD"]
    target_origin_id = "S3-${aws_s3_bucket.portfolio_frontend.id}"
    
    forwarded_values {
      query_string = false
      cookies {
        forward = "none"
      }
    }
    
    viewer_protocol_policy = "redirect-to-https"
    min_ttl                = 0
    default_ttl            = 3600
    max_ttl                = 86400
  }
  
  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }
  
  viewer_certificate {
    cloudfront_default_certificate = true
  }
}

resource "aws_cloudfront_origin_access_identity" "portfolio" {
  comment = "OAI for ${var.project_name}"
}

# ===== IAM ROLES =====
resource "aws_iam_role" "lambda_execution" {
  name = "${var.project_name}-lambda-execution"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "lambda.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy" "lambda_policy" {
  name = "${var.project_name}-lambda-policy"
  role = aws_iam_role.lambda_execution.id
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:*:*:*"
      },
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject"
        ]
        Resource = "${aws_s3_bucket.ml_models.arn}/*"
      },
      {
        Effect = "Allow"
        Action = [
          "dynamodb:PutItem",
          "dynamodb:GetItem",
          "dynamodb:Query",
          "dynamodb:Scan"
        ]
        Resource = aws_dynamodb_table.demo_transactions.arn
      }
    ]
  })
}

# Lambda permission for API Gateway
resource "aws_lambda_permission" "api_gateway" {
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.api_handler.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.main.execution_arn}/*/*"
}

# ===== CLOUDWATCH FOR METRICS (Free Tier: 10 metrics) =====
resource "aws_cloudwatch_metric_alarm" "api_errors" {
  alarm_name          = "${var.project_name}-api-errors"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "Errors"
  namespace           = "AWS/Lambda"
  period              = "300"
  statistic           = "Sum"
  threshold           = "10"
  alarm_description   = "API error rate monitor"
  treat_missing_data  = "notBreaching"
  
  dimensions = {
    FunctionName = aws_lambda_function.api_handler.function_name
  }
}

# ===== OUTPUTS =====
output "website_url" {
  value = "https://${aws_cloudfront_distribution.portfolio.domain_name}"
  description = "CloudFront URL for the portfolio website"
}

output "api_endpoint" {
  value = aws_apigatewayv2_stage.main.invoke_url
  description = "API Gateway endpoint for backend"
}

output "estimated_monthly_cost" {
  value = <<-EOT
    Estimated Monthly Costs (Demo Mode):
    - S3 Static Hosting: ~$1
    - Lambda (10K requests): ~$2
    - API Gateway (10K requests): ~$3
    - DynamoDB On-Demand: ~$2
    - CloudFront: ~$1 (free tier)
    - CloudWatch: Free (under 10 metrics)
    
    Total: ~$10-15/month
    
    Note: Most services fall under AWS Free Tier for first 12 months
  EOT
}