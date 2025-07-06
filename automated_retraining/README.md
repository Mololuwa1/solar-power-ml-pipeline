# Automated Retraining System for Solar Power Generation Models

This system provides end-to-end automated retraining capabilities for solar power generation models using AWS services.

## Architecture Overview

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│ EventBridge │───▶│ Step Functions│───▶│ SageMaker   │
│ Schedule    │    │ Workflow     │    │ Pipeline    │
└─────────────┘    └──────────────┘    └─────────────┘
       │                   │                   │
       ▼                   ▼                   ▼
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│ Lambda      │    │ CloudWatch   │    │ Model       │
│ Functions   │    │ Monitoring   │    │ Registry    │
└─────────────┘    └──────────────┘    └─────────────┘
```

## Components

### 1. SageMaker Pipeline
- Automated model training and evaluation
- Data preprocessing and validation
- Model registration based on performance thresholds

### 2. Step Functions Workflow
- Orchestrates the entire retraining process
- Checks for new data availability
- Triggers pipeline execution
- Handles notifications

### 3. Lambda Functions
- Data monitoring and detection
- Endpoint updates
- Custom business logic

### 4. CloudWatch Monitoring
- Performance metrics tracking
- Automated alerting
- Dashboard visualization

## Quick Start

1. **Update Configuration:**
   - Replace `YOUR_ACCOUNT` with your AWS account ID
   - Update S3 bucket names
   - Verify IAM role ARNs

2. **Deploy Components:**
   ```bash
   python3 scripts/deploy.py
   ```

3. **Test the System:**
   - Upload test data to S3
   - Trigger workflow manually
   - Monitor execution in AWS console

## Configuration

### Environment Variables
```bash
export AWS_DEFAULT_REGION=us-east-1
export SAGEMAKER_ROLE_ARN=arn:aws:iam::YOUR_ACCOUNT:role/SageMakerExecutionRole
export S3_BUCKET=your-solar-data-bucket
```

### Pipeline Parameters
- `accuracy_threshold`: Minimum R² score for model approval (default: 0.95)
- `instance_type`: Training instance type (default: ml.m5.large)
- `model_type`: Model architecture (default: neural_network)

## Monitoring

### Key Metrics
- Model inference latency
- Prediction accuracy
- Pipeline execution status
- Error rates

### Alerts
- High latency warnings
- Model performance degradation
- Pipeline failures

## Security

- Use IAM roles with least-privilege principle
- Enable encryption at rest and in transit
- Deploy in private subnets when possible
- Enable CloudTrail for audit logging

## Cost Optimization

- Use Spot instances for training jobs
- Implement auto-scaling for endpoints
- Set up S3 lifecycle policies
- Monitor and optimize resource usage

## Troubleshooting

### Common Issues
1. **Pipeline fails to start:** Check IAM permissions and S3 access
2. **Training job failures:** Review CloudWatch logs and data quality
3. **Endpoint update fails:** Verify model artifacts and configuration

### Debugging Commands
```bash
# View pipeline executions
aws sagemaker list-pipeline-executions --pipeline-name solar-power-retraining-pipeline

# Check Step Functions execution
aws stepfunctions list-executions --state-machine-arn YOUR_STATE_MACHINE_ARN

# View Lambda logs
aws logs describe-log-groups --log-group-name-prefix /aws/lambda/solar
```

## Support

For issues and questions:
- Check the troubleshooting section
- Review AWS documentation
- Contact your AWS support team
