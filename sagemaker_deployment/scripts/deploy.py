#!/usr/bin/env python3
"""
SageMaker Deployment Script
"""

import boto3
import json
import time
import os

def load_config(config_path):
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)

def create_training_job(sagemaker_client, config):
    """Create SageMaker training job"""
    print("Creating training job...")
    
    response = sagemaker_client.create_training_job(**config)
    job_name = config['TrainingJobName']
    
    print(f"Training job created: {job_name}")
    print(f"Job ARN: {response['TrainingJobArn']}")
    
    # Wait for completion
    print("Waiting for training job to complete...")
    waiter = sagemaker_client.get_waiter('training_job_completed_or_stopped')
    waiter.wait(TrainingJobName=job_name)
    
    # Get final status
    response = sagemaker_client.describe_training_job(TrainingJobName=job_name)
    status = response['TrainingJobStatus']
    
    if status == 'Completed':
        print("Training job completed successfully!")
        return response['ModelArtifacts']['S3ModelArtifacts']
    else:
        print(f"Training job failed with status: {status}")
        return None

def create_model(sagemaker_client, config, model_data_url):
    """Create SageMaker model"""
    print("Creating model...")
    
    # Update model data URL
    config['PrimaryContainer']['ModelDataUrl'] = model_data_url
    
    response = sagemaker_client.create_model(**config)
    print(f"Model created: {config['ModelName']}")
    print(f"Model ARN: {response['ModelArn']}")
    
    return response['ModelArn']

def create_endpoint_config(sagemaker_client, config):
    """Create endpoint configuration"""
    print("Creating endpoint configuration...")
    
    response = sagemaker_client.create_endpoint_config(**config)
    print(f"Endpoint config created: {config['EndpointConfigName']}")
    print(f"Config ARN: {response['EndpointConfigArn']}")
    
    return response['EndpointConfigArn']

def create_endpoint(sagemaker_client, config):
    """Create endpoint"""
    print("Creating endpoint...")
    
    response = sagemaker_client.create_endpoint(**config)
    endpoint_name = config['EndpointName']
    
    print(f"Endpoint created: {endpoint_name}")
    print(f"Endpoint ARN: {response['EndpointArn']}")
    
    # Wait for endpoint to be in service
    print("Waiting for endpoint to be in service...")
    waiter = sagemaker_client.get_waiter('endpoint_in_service')
    waiter.wait(EndpointName=endpoint_name)
    
    print("Endpoint is now in service!")
    return response['EndpointArn']

def main():
    """Main deployment function"""
    # Initialize SageMaker client
    sagemaker_client = boto3.client('sagemaker')
    
    # Load configurations
    config_dir = 'config'
    training_config = load_config(os.path.join(config_dir, 'training_job.json'))
    model_config = load_config(os.path.join(config_dir, 'model.json'))
    endpoint_config_config = load_config(os.path.join(config_dir, 'endpoint_config.json'))
    endpoint_config = load_config(os.path.join(config_dir, 'endpoint.json'))
    
    try:
        # Step 1: Create training job
        model_data_url = create_training_job(sagemaker_client, training_config)
        if not model_data_url:
            print("Training job failed. Exiting.")
            return
        
        # Step 2: Create model
        model_arn = create_model(sagemaker_client, model_config, model_data_url)
        
        # Step 3: Create endpoint configuration
        endpoint_config_arn = create_endpoint_config(sagemaker_client, endpoint_config_config)
        
        # Step 4: Create endpoint
        endpoint_arn = create_endpoint(sagemaker_client, endpoint_config)
        
        print("\n=== DEPLOYMENT COMPLETE ===")
        print(f"Endpoint Name: {endpoint_config['EndpointName']}")
        print(f"Endpoint ARN: {endpoint_arn}")
        print("\nYou can now use this endpoint for real-time inference!")
        
    except Exception as e:
        print(f"Deployment failed: {str(e)}")

if __name__ == '__main__':
    main()
