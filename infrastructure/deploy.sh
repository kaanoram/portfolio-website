#!/bin/bash

# Deployment script for E-commerce Analytics Platform
# This script builds and deploys the application to AWS ECS

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
AWS_REGION=${AWS_REGION:-"us-east-1"}
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_BACKEND_REPO="ecommerce-analytics-backend"
ECR_FRONTEND_REPO="ecommerce-analytics-frontend"
ECS_CLUSTER="ecommerce-analytics-cluster"
ECS_SERVICE_BACKEND="ecommerce-analytics-backend"
ECS_SERVICE_FRONTEND="ecommerce-analytics-frontend"

echo -e "${GREEN}Starting deployment process...${NC}"

# Check AWS CLI is configured
if ! aws sts get-caller-identity > /dev/null 2>&1; then
    echo -e "${RED}Error: AWS CLI not configured. Please run 'aws configure'${NC}"
    exit 1
fi

# Login to ECR
echo -e "${YELLOW}Logging into ECR...${NC}"
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Build and push backend
echo -e "${YELLOW}Building backend Docker image...${NC}"
docker build -f Dockerfile.backend -t $ECR_BACKEND_REPO:latest .

echo -e "${YELLOW}Tagging backend image...${NC}"
docker tag $ECR_BACKEND_REPO:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_BACKEND_REPO:latest
docker tag $ECR_BACKEND_REPO:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_BACKEND_REPO:$(git rev-parse --short HEAD)

echo -e "${YELLOW}Pushing backend image to ECR...${NC}"
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_BACKEND_REPO:latest
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_BACKEND_REPO:$(git rev-parse --short HEAD)

# Build and push frontend
echo -e "${YELLOW}Building frontend Docker image...${NC}"
docker build -f Dockerfile.frontend -t $ECR_FRONTEND_REPO:latest .

echo -e "${YELLOW}Tagging frontend image...${NC}"
docker tag $ECR_FRONTEND_REPO:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_FRONTEND_REPO:latest
docker tag $ECR_FRONTEND_REPO:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_FRONTEND_REPO:$(git rev-parse --short HEAD)

echo -e "${YELLOW}Pushing frontend image to ECR...${NC}"
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_FRONTEND_REPO:latest
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_FRONTEND_REPO:$(git rev-parse --short HEAD)

# Update ECS services
echo -e "${YELLOW}Updating ECS services...${NC}"
aws ecs update-service --cluster $ECS_CLUSTER --service $ECS_SERVICE_BACKEND --force-new-deployment --region $AWS_REGION
aws ecs update-service --cluster $ECS_CLUSTER --service $ECS_SERVICE_FRONTEND --force-new-deployment --region $AWS_REGION

echo -e "${YELLOW}Waiting for backend service to stabilize...${NC}"
aws ecs wait services-stable --cluster $ECS_CLUSTER --services $ECS_SERVICE_BACKEND --region $AWS_REGION

echo -e "${YELLOW}Waiting for frontend service to stabilize...${NC}"
aws ecs wait services-stable --cluster $ECS_CLUSTER --services $ECS_SERVICE_FRONTEND --region $AWS_REGION

# Get ALB URL
ALB_URL=$(aws elbv2 describe-load-balancers --names ecommerce-analytics-alb --query 'LoadBalancers[0].DNSName' --output text --region $AWS_REGION)

echo -e "${GREEN}Deployment complete!${NC}"
echo -e "${GREEN}Application URL: http://$ALB_URL${NC}"
echo -e "${GREEN}Health check: http://$ALB_URL/health${NC}"