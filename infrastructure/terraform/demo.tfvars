# Terraform Variables for Demo Deployment
# Low-cost configuration for portfolio demonstration

# Basic Configuration
aws_region   = "us-east-1"  # Cheapest region with all services
project_name = "ecommerce-analytics-demo"
environment  = "demo"

# Cost Optimization Settings
enable_cost_optimization = true
demo_hours              = 12  # Run 12 hours per day

# Lambda Configuration (Serverless = Pay per use)
lambda_memory_size = 512  # Minimum for ML inference
lambda_timeout     = 30   # seconds

# Frontend Configuration
enable_cdn = true  # CloudFront for global distribution

# Optional: Custom Domain (if you have one)
# domain_name = "demo.kaanoram.io"
# create_dns_records = true

# Tags for Resource Management
tags = {
  Project     = "E-commerce Analytics Platform"
  Environment = "Demo"
  Purpose     = "Portfolio"
  CostCenter  = "Personal"
  AutoOff     = "true"  # For automated shutdown scripts
}