#!/usr/bin/env python3
"""
Model Versioning and Registry System
"""

import json
import boto3
import joblib
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional

class ModelRegistry:
    """Centralized model registry for version management"""
    
    def __init__(self, registry_bucket: str, registry_prefix: str = "model-registry"):
        self.s3_client = boto3.client('s3')
        self.sagemaker_client = boto3.client('sagemaker')
        self.registry_bucket = registry_bucket
        self.registry_prefix = registry_prefix
        self.registry_file = f"{registry_prefix}/registry.json"
    
    def register_model(self, 
                      model_name: str,
                      model_version: str,
                      model_path: str,
                      metadata: Dict,
                      performance_metrics: Dict) -> str:
        """Register a new model version"""
        
        # Load existing registry
        registry = self._load_registry()
        
        # Create model entry
        model_entry = {
            "model_name": model_name,
            "version": model_version,
            "timestamp": datetime.now().isoformat(),
            "model_path": model_path,
            "status": "registered",
            "metadata": metadata,
            "performance_metrics": performance_metrics,
            "deployment_history": [],
            "monitoring_data": {}
        }
        
        # Add to registry
        if model_name not in registry:
            registry[model_name] = {}
        
        registry[model_name][model_version] = model_entry
        
        # Save registry
        self._save_registry(registry)
        
        print(f"Model {model_name} v{model_version} registered successfully")
        return f"{model_name}:{model_version}"
    
    def promote_model(self, model_name: str, version: str, stage: str) -> bool:
        """Promote model to a specific stage (staging, production)"""
        
        registry = self._load_registry()
        
        if model_name in registry and version in registry[model_name]:
            registry[model_name][version]["status"] = stage
            registry[model_name][version]["promoted_at"] = datetime.now().isoformat()
            
            # Demote previous production model
            if stage == "production":
                for v, model_info in registry[model_name].items():
                    if v != version and model_info.get("status") == "production":
                        model_info["status"] = "archived"
                        model_info["archived_at"] = datetime.now().isoformat()
            
            self._save_registry(registry)
            print(f"Model {model_name} v{version} promoted to {stage}")
            return True
        
        return False
    
    def get_model_info(self, model_name: str, version: str = None) -> Dict:
        """Get model information"""
        
        registry = self._load_registry()
        
        if model_name not in registry:
            return {}
        
        if version:
            return registry[model_name].get(version, {})
        else:
            return registry[model_name]
    
    def list_models(self, status: str = None) -> List[Dict]:
        """List all models, optionally filtered by status"""
        
        registry = self._load_registry()
        models = []
        
        for model_name, versions in registry.items():
            for version, model_info in versions.items():
                if status is None or model_info.get("status") == status:
                    models.append({
                        "model_name": model_name,
                        "version": version,
                        **model_info
                    })
        
        return sorted(models, key=lambda x: x["timestamp"], reverse=True)
    
    def _load_registry(self) -> Dict:
        """Load registry from S3"""
        try:
            response = self.s3_client.get_object(
                Bucket=self.registry_bucket,
                Key=self.registry_file
            )
            return json.loads(response['Body'].read().decode())
        except:
            return {}
    
    def _save_registry(self, registry: Dict):
        """Save registry to S3"""
        self.s3_client.put_object(
            Bucket=self.registry_bucket,
            Key=self.registry_file,
            Body=json.dumps(registry, indent=2)
        )

class ModelVersionManager:
    """Manage model versions and deployments"""
    
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self.sagemaker_client = boto3.client('sagemaker')
    
    def create_model_version(self, 
                           model_name: str,
                           model_artifacts_path: str,
                           training_job_name: str,
                           performance_metrics: Dict) -> str:
        """Create a new model version"""
        
        # Generate version number
        existing_models = self.registry.list_models()
        model_versions = [m["version"] for m in existing_models if m["model_name"] == model_name]
        
        if model_versions:
            latest_version = max([int(v.split('.')[0]) for v in model_versions])
            new_version = f"{latest_version + 1}.0"
        else:
            new_version = "1.0"
        
        # Get training job details
        training_job = self.sagemaker_client.describe_training_job(
            TrainingJobName=training_job_name
        )
        
        # Create metadata
        metadata = {
            "training_job_name": training_job_name,
            "training_job_arn": training_job["TrainingJobArn"],
            "algorithm_specification": training_job["AlgorithmSpecification"],
            "hyperparameters": training_job.get("HyperParameters", {}),
            "input_data_config": training_job["InputDataConfig"],
            "created_by": "automated_pipeline",
            "framework": "scikit-learn",
            "python_version": "3.9"
        }
        
        # Register model
        model_id = self.registry.register_model(
            model_name=model_name,
            model_version=new_version,
            model_path=model_artifacts_path,
            metadata=metadata,
            performance_metrics=performance_metrics
        )
        
        return model_id
    
    def deploy_model_version(self, 
                           model_name: str,
                           version: str,
                           endpoint_name: str,
                           instance_type: str = "ml.t2.medium") -> str:
        """Deploy a specific model version to an endpoint"""
        
        model_info = self.registry.get_model_info(model_name, version)
        if not model_info:
            raise ValueError(f"Model {model_name} v{version} not found")
        
        # Create SageMaker model
        model_name_sm = f"{model_name}-{version}-{int(datetime.now().timestamp())}"
        
        create_model_response = self.sagemaker_client.create_model(
            ModelName=model_name_sm,
            PrimaryContainer={
                'Image': '683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3',
                'ModelDataUrl': model_info["model_path"],
                'Environment': {
                    'SAGEMAKER_PROGRAM': 'inference.py',
                    'SAGEMAKER_SUBMIT_DIRECTORY': '/opt/ml/code'
                }
            },
            ExecutionRoleArn='arn:aws:iam::YOUR_ACCOUNT:role/SageMakerExecutionRole'
        )
        
        # Update deployment history
        registry = self.registry._load_registry()
        if "deployment_history" not in registry[model_name][version]:
            registry[model_name][version]["deployment_history"] = []
        
        registry[model_name][version]["deployment_history"].append({
            "endpoint_name": endpoint_name,
            "deployed_at": datetime.now().isoformat(),
            "instance_type": instance_type,
            "model_name": model_name_sm
        })
        
        self.registry._save_registry(registry)
        
        return create_model_response["ModelArn"]

# Example usage
if __name__ == "__main__":
    # Initialize registry
    registry = ModelRegistry("your-model-registry-bucket")
    version_manager = ModelVersionManager(registry)
    
    # Example: Register a new model
    performance_metrics = {
        "rmse": 0.2935,
        "mae": 0.1142,
        "r2_score": 0.9905
    }
    
    model_id = version_manager.create_model_version(
        model_name="solar-power-generation",
        model_artifacts_path="s3://your-bucket/models/model.tar.gz",
        training_job_name="solar-training-job-123",
        performance_metrics=performance_metrics
    )
    
    print(f"Created model version: {model_id}")
