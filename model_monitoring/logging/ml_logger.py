#!/usr/bin/env python3
"""
Comprehensive Logging System for Solar Power ML Pipeline
"""

import logging
import json
import boto3
from datetime import datetime
from typing import Dict, Any
import traceback

class MLPipelineLogger:
    """Centralized logging for ML pipeline"""
    
    def __init__(self, log_group_name: str = "solar-power-ml-pipeline"):
        self.log_group_name = log_group_name
        self.cloudwatch_logs = boto3.client('logs')
        self.s3_client = boto3.client('s3')
        
        # Create log group if it doesn't exist
        self._create_log_group()
        
        # Set up local logger
        self.logger = logging.getLogger('solar_ml_pipeline')
        self.logger.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
    
    def _create_log_group(self):
        """Create CloudWatch log group"""
        try:
            self.cloudwatch_logs.create_log_group(logGroupName=self.log_group_name)
        except self.cloudwatch_logs.exceptions.ResourceAlreadyExistsException:
            pass
        except Exception as e:
            print(f"Error creating log group: {e}")
    
    def log_training_start(self, 
                          training_job_name: str,
                          model_config: Dict,
                          data_config: Dict):
        """Log training job start"""
        
        log_entry = {
            "event_type": "training_start",
            "timestamp": datetime.now().isoformat(),
            "training_job_name": training_job_name,
            "model_config": model_config,
            "data_config": data_config
        }
        
        self._send_log("INFO", "Training job started", log_entry)
    
    def log_training_complete(self, 
                            training_job_name: str,
                            metrics: Dict,
                            model_artifacts_path: str):
        """Log training job completion"""
        
        log_entry = {
            "event_type": "training_complete",
            "timestamp": datetime.now().isoformat(),
            "training_job_name": training_job_name,
            "metrics": metrics,
            "model_artifacts_path": model_artifacts_path
        }
        
        self._send_log("INFO", "Training job completed", log_entry)
    
    def log_deployment(self, 
                      endpoint_name: str,
                      model_version: str,
                      instance_type: str):
        """Log model deployment"""
        
        log_entry = {
            "event_type": "model_deployment",
            "timestamp": datetime.now().isoformat(),
            "endpoint_name": endpoint_name,
            "model_version": model_version,
            "instance_type": instance_type
        }
        
        self._send_log("INFO", "Model deployed", log_entry)
    
    def log_prediction_batch(self, 
                           batch_id: str,
                           sample_count: int,
                           model_version: str,
                           performance_metrics: Dict = None):
        """Log prediction batch"""
        
        log_entry = {
            "event_type": "prediction_batch",
            "timestamp": datetime.now().isoformat(),
            "batch_id": batch_id,
            "sample_count": sample_count,
            "model_version": model_version,
            "performance_metrics": performance_metrics
        }
        
        self._send_log("INFO", "Prediction batch processed", log_entry)
    
    def log_drift_detection(self, 
                          drift_type: str,
                          drift_detected: bool,
                          drift_score: float,
                          details: Dict = None):
        """Log drift detection results"""
        
        log_entry = {
            "event_type": "drift_detection",
            "timestamp": datetime.now().isoformat(),
            "drift_type": drift_type,
            "drift_detected": drift_detected,
            "drift_score": drift_score,
            "details": details
        }
        
        level = "WARNING" if drift_detected else "INFO"
        message = f"{drift_type} drift {'detected' if drift_detected else 'not detected'}"
        
        self._send_log(level, message, log_entry)
    
    def log_error(self, 
                 error_type: str,
                 error_message: str,
                 context: Dict = None,
                 exception: Exception = None):
        """Log errors and exceptions"""
        
        log_entry = {
            "event_type": "error",
            "timestamp": datetime.now().isoformat(),
            "error_type": error_type,
            "error_message": error_message,
            "context": context
        }
        
        if exception:
            log_entry["exception_type"] = type(exception).__name__
            log_entry["traceback"] = traceback.format_exc()
        
        self._send_log("ERROR", f"{error_type}: {error_message}", log_entry)
    
    def log_pipeline_execution(self, 
                             pipeline_name: str,
                             execution_id: str,
                             status: str,
                             steps_completed: List[str] = None):
        """Log pipeline execution status"""
        
        log_entry = {
            "event_type": "pipeline_execution",
            "timestamp": datetime.now().isoformat(),
            "pipeline_name": pipeline_name,
            "execution_id": execution_id,
            "status": status,
            "steps_completed": steps_completed or []
        }
        
        level = "ERROR" if status == "failed" else "INFO"
        self._send_log(level, f"Pipeline {status}", log_entry)
    
    def _send_log(self, level: str, message: str, details: Dict = None):
        """Send log to CloudWatch and local logger"""
        
        # Log locally
        getattr(self.logger, level.lower())(f"{message}: {json.dumps(details) if details else ''}")
        
        # Send to CloudWatch
        try:
            log_stream_name = f"ml-pipeline-{datetime.now().strftime('%Y-%m-%d')}"
            
            # Create log stream if it doesn't exist
            try:
                self.cloudwatch_logs.create_log_stream(
                    logGroupName=self.log_group_name,
                    logStreamName=log_stream_name
                )
            except self.cloudwatch_logs.exceptions.ResourceAlreadyExistsException:
                pass
            
            # Prepare log event
            log_event = {
                'timestamp': int(datetime.now().timestamp() * 1000),
                'message': json.dumps({
                    'level': level,
                    'message': message,
                    'details': details
                })
            }
            
            # Send log event
            self.cloudwatch_logs.put_log_events(
                logGroupName=self.log_group_name,
                logStreamName=log_stream_name,
                logEvents=[log_event]
            )
            
        except Exception as e:
            print(f"Error sending log to CloudWatch: {e}")

class AuditLogger:
    """Audit logging for compliance and governance"""
    
    def __init__(self, s3_bucket: str, audit_prefix: str = "audit-logs"):
        self.s3_client = boto3.client('s3')
        self.s3_bucket = s3_bucket
        self.audit_prefix = audit_prefix
    
    def log_data_access(self, 
                       user_id: str,
                       data_source: str,
                       access_type: str,
                       data_classification: str = "internal"):
        """Log data access for audit purposes"""
        
        audit_entry = {
            "event_type": "data_access",
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "data_source": data_source,
            "access_type": access_type,
            "data_classification": data_classification,
            "ip_address": "system",  # Would be actual IP in real implementation
            "session_id": f"session_{int(datetime.now().timestamp())}"
        }
        
        self._save_audit_log(audit_entry)
    
    def log_model_change(self, 
                        user_id: str,
                        model_name: str,
                        change_type: str,
                        old_version: str = None,
                        new_version: str = None):
        """Log model changes for audit purposes"""
        
        audit_entry = {
            "event_type": "model_change",
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "model_name": model_name,
            "change_type": change_type,
            "old_version": old_version,
            "new_version": new_version
        }
        
        self._save_audit_log(audit_entry)
    
    def _save_audit_log(self, audit_entry: Dict):
        """Save audit log to S3"""
        
        # Create audit log key with date partitioning
        date_str = datetime.now().strftime('%Y/%m/%d')
        timestamp = int(datetime.now().timestamp())
        audit_key = f"{self.audit_prefix}/{date_str}/audit_{timestamp}.json"
        
        try:
            self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=audit_key,
                Body=json.dumps(audit_entry, indent=2)
            )
        except Exception as e:
            print(f"Error saving audit log: {e}")

# Example usage
if __name__ == "__main__":
    # Initialize loggers
    ml_logger = MLPipelineLogger()
    audit_logger = AuditLogger("your-audit-bucket")
    
    # Example logging
    ml_logger.log_training_start(
        training_job_name="solar-training-123",
        model_config={"model_type": "neural_network"},
        data_config={"data_source": "s3://bucket/data"}
    )
    
    audit_logger.log_data_access(
        user_id="system",
        data_source="solar_power_data",
        access_type="read"
    )
    
    print("Logging system initialized successfully!")
