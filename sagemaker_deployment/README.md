# SageMaker Deployment for Solar Power Generation Models

This directory contains all the necessary files to deploy the solar power generation models on Amazon SageMaker.

## Directory Structure

```
sagemaker_deployment/
├── code/
│   ├── train.py          # Training script
│   └── inference.py      # Inference script
├── config/
│   ├── training_job.json      # Training job configuration
│   ├── model.json            # Model configuration
│   ├── endpoint_config.json  # Endpoint configuration
│   └── endpoint.json         # Endpoint configuration
├── scripts/
│   ├── deploy.py         # Deployment script
│   └── test_endpoint.py  # Endpoint testing script
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Prerequisites

1. AWS CLI configured with appropriate credentials
2. SageMaker execution role with necessary permissions
3. S3 bucket for storing data and models
4. Python 3.7+ with boto3 installed

## Setup Instructions

### 1. Update Configuration Files

Before deploying, update the following in the configuration files:

- Replace `YOUR_ACCOUNT` with your AWS account ID
- Replace `your-bucket` with your S3 bucket name
- Update the SageMaker execution role ARN

### 2. Prepare Data

Upload your training and validation data to S3:

```bash
aws s3 cp processed_solar_data.csv s3://your-bucket/solar-data/train/
```

### 3. Deploy the Model

Run the deployment script:

```bash
cd sagemaker_deployment
python scripts/deploy.py
```

This will:
1. Create a SageMaker training job
2. Create a model from the trained artifacts
3. Create an endpoint configuration
4. Deploy the endpoint

### 4. Test the Endpoint

Once deployed, test the endpoint:

```bash
python scripts/test_endpoint.py
```

## Model Types

The training script supports three model types:

1. **XGBoost** (`xgboost`)
2. **Random Forest** (`random_forest`)
3. **Neural Network** (`neural_network`) - Default and best performing

## Hyperparameters

You can customize the following hyperparameters in the training job configuration:

- `model-type`: Type of model to train
- `n-estimators`: Number of estimators (for tree-based models)
- `max-depth`: Maximum depth (for tree-based models)
- `learning-rate`: Learning rate (for XGBoost)
- `hidden-layers`: Hidden layer sizes (for neural network, comma-separated)
- `alpha`: Regularization parameter (for neural network)

## Inference

The endpoint accepts data in two formats:

### CSV Format
```
Content-Type: text/csv

hour,month,hour_sin,hour_cos,is_daytime,Irradiance_mean,Temperature_mean,...
12,6,0.0,1.0,1,800.0,25.0,...
```

### JSON Format
```
Content-Type: application/json

{
  "hour": 12,
  "month": 6,
  "hour_sin": 0.0,
  "hour_cos": 1.0,
  "is_daytime": 1,
  "Irradiance_mean": 800.0,
  "Temperature_mean": 25.0
}
```

## Response Format

The endpoint returns predictions in JSON format:

```json
{
  "predictions": [2.5, 3.1, 2.8],
  "model_type": "solar_power_generation",
  "target": "generation(kWh)"
}
```

## Monitoring and Logging

SageMaker automatically provides:
- CloudWatch metrics for endpoint performance
- CloudWatch logs for inference requests
- Model monitoring capabilities

## Cost Optimization

- Use `ml.t2.medium` instances for low-traffic endpoints
- Consider auto-scaling for variable workloads
- Use spot instances for training jobs to reduce costs

## Troubleshooting

### Common Issues

1. **Training job fails**: Check CloudWatch logs for detailed error messages
2. **Endpoint creation fails**: Verify IAM permissions and S3 access
3. **Inference errors**: Ensure input data matches expected feature schema

### Useful Commands

```bash
# List training jobs
aws sagemaker list-training-jobs

# Describe endpoint
aws sagemaker describe-endpoint --endpoint-name solar-power-generation-endpoint

# Delete endpoint (to stop charges)
aws sagemaker delete-endpoint --endpoint-name solar-power-generation-endpoint
```

## Security Considerations

- Use IAM roles with minimal required permissions
- Enable VPC configuration for private deployments
- Use encryption for data at rest and in transit
- Regularly rotate access keys and credentials

## Next Steps

1. Set up automated retraining pipelines
2. Implement A/B testing for model versions
3. Add real-time monitoring and alerting
4. Integrate with your application via API Gateway
