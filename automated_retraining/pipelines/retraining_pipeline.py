#!/usr/bin/env python3
"""
SageMaker Pipeline for Solar Power Generation Model Retraining
"""

import boto3
import json
from sagemaker import get_execution_role
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.inputs import TrainingInput
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.parameters import ParameterString, ParameterFloat

def create_pipeline(
    region='us-east-1',
    role=None,
    bucket='your-bucket',
    pipeline_name='solar-power-retraining-pipeline'
):
    """Create the SageMaker pipeline"""
    
    # Parameters
    input_data = ParameterString(
        name="InputData",
        default_value=f"s3://{bucket}/solar-data/latest/"
    )
    
    accuracy_threshold = ParameterFloat(
        name="AccuracyThreshold",
        default_value=0.95
    )
    
    # Processing step for data validation and preprocessing
    sklearn_processor = SKLearnProcessor(
        framework_version="0.23-1",
        instance_type="ml.m5.large",
        instance_count=1,
        base_job_name="solar-data-processing",
        role=role,
    )
    
    processing_step = ProcessingStep(
        name="DataProcessing",
        processor=sklearn_processor,
        inputs=[
            ProcessingInput(
                source=input_data,
                destination="/opt/ml/processing/input"
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="train",
                source="/opt/ml/processing/train",
                destination=f"s3://{bucket}/processed-data/train"
            ),
            ProcessingOutput(
                output_name="validation",
                source="/opt/ml/processing/validation", 
                destination=f"s3://{bucket}/processed-data/validation"
            )
        ],
        code="preprocessing.py"
    )
    
    # Training step
    sklearn_estimator = SKLearn(
        entry_point="train.py",
        framework_version="0.23-1",
        instance_type="ml.m5.large",
        role=role,
        hyperparameters={
            "model-type": "neural_network",
            "hidden-layers": "100,50,25",
            "alpha": "0.001"
        }
    )
    
    training_step = TrainingStep(
        name="TrainModel",
        estimator=sklearn_estimator,
        inputs={
            "train": TrainingInput(
                s3_data=processing_step.properties.ProcessingOutputConfig.Outputs[
                    "train"
                ].S3Output.S3Uri,
                content_type="text/csv"
            ),
            "validation": TrainingInput(
                s3_data=processing_step.properties.ProcessingOutputConfig.Outputs[
                    "validation"
                ].S3Output.S3Uri,
                content_type="text/csv"
            )
        }
    )
    
    # Create pipeline
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[input_data, accuracy_threshold],
        steps=[processing_step, training_step]
    )
    
    return pipeline

if __name__ == '__main__':
    # Example usage
    pipeline = create_pipeline()
    print("SageMaker Pipeline created successfully!")
