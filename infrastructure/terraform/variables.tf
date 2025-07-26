variable "aws_region" {
  description = "AWS region for deployment"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "ecommerce-analytics"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "prod"
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "private_subnets" {
  description = "Private subnet CIDR blocks"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
}

variable "public_subnets" {
  description = "Public subnet CIDR blocks"
  type        = list(string)
  default     = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
}

variable "rds_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.r6g.xlarge"  # 4 vCPUs, 32 GiB RAM for production workload
}

variable "db_username" {
  description = "Database master username"
  type        = string
  default     = "ecommerce_admin"
  sensitive   = true
}

variable "db_password" {
  description = "Database master password"
  type        = string
  sensitive   = true
}

variable "backend_min_tasks" {
  description = "Minimum number of backend ECS tasks"
  type        = number
  default     = 2
}

variable "backend_max_tasks" {
  description = "Maximum number of backend ECS tasks"
  type        = number
  default     = 100  # To handle 2000+ concurrent users
}

variable "frontend_min_tasks" {
  description = "Minimum number of frontend ECS tasks"
  type        = number
  default     = 2
}

variable "frontend_max_tasks" {
  description = "Maximum number of frontend ECS tasks"
  type        = number
  default     = 50
}

variable "backend_cpu" {
  description = "CPU units for backend tasks (1024 = 1 vCPU)"
  type        = number
  default     = 2048  # 2 vCPUs
}

variable "backend_memory" {
  description = "Memory for backend tasks in MB"
  type        = number
  default     = 4096  # 4 GB
}

variable "frontend_cpu" {
  description = "CPU units for frontend tasks"
  type        = number
  default     = 512
}

variable "frontend_memory" {
  description = "Memory for frontend tasks in MB"
  type        = number
  default     = 1024
}