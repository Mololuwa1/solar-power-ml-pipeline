#!/usr/bin/env python3
"""
Model Performance Tracking System
"""

import pandas as pd
import numpy as np
import boto3
import json
from datetime import datetime, timedelta
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns

class PerformanceTracker:
    """Track model performance over time"""
    
    def __init__(self, s3_bucket: str, tracking_prefix: str = "performance-tracking"):
        self.s3_client = boto3.client('s3')
        self.cloudwatch = boto3.client('cloudwatch')
        self.s3_bucket = s3_bucket
        self.tracking_prefix = tracking_prefix
        self.performance_history = []
    
    def log_prediction_batch(self, 
                           predictions: np.ndarray,
                           actuals: np.ndarray,
                           features: pd.DataFrame,
                           model_version: str,
                           batch_id: str = None) -> str:
        """Log a batch of predictions for performance tracking"""
        
        if batch_id is None:
            batch_id = f"batch_{int(datetime.now().timestamp())}"
        
        # Calculate metrics
        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
        mae = np.mean(np.abs(predictions - actuals))
        mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
        r2 = 1 - (np.sum((actuals - predictions) ** 2) / np.sum((actuals - np.mean(actuals)) ** 2))
        
        # Create performance record
        performance_record = {
            "batch_id": batch_id,
            "timestamp": datetime.now().isoformat(),
            "model_version": model_version,
            "sample_count": len(predictions),
            "metrics": {
                "rmse": float(rmse),
                "mae": float(mae),
                "mape": float(mape),
                "r2_score": float(r2)
            },
            "prediction_stats": {
                "mean": float(np.mean(predictions)),
                "std": float(np.std(predictions)),
                "min": float(np.min(predictions)),
                "max": float(np.max(predictions))
            },
            "actual_stats": {
                "mean": float(np.mean(actuals)),
                "std": float(np.std(actuals)),
                "min": float(np.min(actuals)),
                "max": float(np.max(actuals))
            }
        }
        
        # Save detailed data
        detailed_data = pd.DataFrame({
            'predictions': predictions,
            'actuals': actuals,
            'residuals': actuals - predictions,
            'absolute_errors': np.abs(actuals - predictions)
        })
        
        # Add feature data if provided
        if features is not None:
            detailed_data = pd.concat([detailed_data, features.reset_index(drop=True)], axis=1)
        
        # Save to S3
        performance_key = f"{self.tracking_prefix}/performance_records/{batch_id}.json"
        data_key = f"{self.tracking_prefix}/detailed_data/{batch_id}.csv"
        
        self.s3_client.put_object(
            Bucket=self.s3_bucket,
            Key=performance_key,
            Body=json.dumps(performance_record, indent=2)
        )
        
        self.s3_client.put_object(
            Bucket=self.s3_bucket,
            Key=data_key,
            Body=detailed_data.to_csv(index=False)
        )
        
        # Send metrics to CloudWatch
        self._send_performance_metrics(performance_record)
        
        print(f"Performance logged for batch {batch_id}: RMSE={rmse:.4f}, R²={r2:.4f}")
        return batch_id
    
    def get_performance_history(self, 
                              model_version: str = None,
                              days_back: int = 30) -> pd.DataFrame:
        """Get performance history for analysis"""
        
        # List performance records
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.s3_bucket,
                Prefix=f"{self.tracking_prefix}/performance_records/"
            )
            
            performance_data = []
            
            if 'Contents' in response:
                for obj in response['Contents']:
                    # Get object
                    record_response = self.s3_client.get_object(
                        Bucket=self.s3_bucket,
                        Key=obj['Key']
                    )
                    
                    record = json.loads(record_response['Body'].read().decode())
                    record_time = datetime.fromisoformat(record['timestamp'].replace('Z', '+00:00'))
                    
                    # Filter by date and model version
                    if record_time >= cutoff_date:
                        if model_version is None or record.get('model_version') == model_version:
                            # Flatten the record for DataFrame
                            flat_record = {
                                'batch_id': record['batch_id'],
                                'timestamp': record['timestamp'],
                                'model_version': record['model_version'],
                                'sample_count': record['sample_count'],
                                **record['metrics']
                            }
                            performance_data.append(flat_record)
            
            return pd.DataFrame(performance_data)
            
        except Exception as e:
            print(f"Error retrieving performance history: {e}")
            return pd.DataFrame()
    
    def generate_performance_report(self, 
                                  model_version: str = None,
                                  days_back: int = 30) -> Dict:
        """Generate comprehensive performance report"""
        
        history_df = self.get_performance_history(model_version, days_back)
        
        if history_df.empty:
            return {"error": "No performance data available"}
        
        # Calculate summary statistics
        report = {
            "report_generated": datetime.now().isoformat(),
            "model_version": model_version,
            "time_period_days": days_back,
            "total_batches": len(history_df),
            "total_samples": history_df['sample_count'].sum(),
            "performance_summary": {
                "rmse": {
                    "mean": float(history_df['rmse'].mean()),
                    "std": float(history_df['rmse'].std()),
                    "min": float(history_df['rmse'].min()),
                    "max": float(history_df['rmse'].max())
                },
                "mae": {
                    "mean": float(history_df['mae'].mean()),
                    "std": float(history_df['mae'].std()),
                    "min": float(history_df['mae'].min()),
                    "max": float(history_df['mae'].max())
                },
                "r2_score": {
                    "mean": float(history_df['r2_score'].mean()),
                    "std": float(history_df['r2_score'].std()),
                    "min": float(history_df['r2_score'].min()),
                    "max": float(history_df['r2_score'].max())
                }
            }
        }
        
        # Trend analysis
        if len(history_df) > 1:
            history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
            history_df = history_df.sort_values('timestamp')
            
            # Calculate trends
            rmse_trend = np.polyfit(range(len(history_df)), history_df['rmse'], 1)[0]
            r2_trend = np.polyfit(range(len(history_df)), history_df['r2_score'], 1)[0]
            
            report["trends"] = {
                "rmse_trend": float(rmse_trend),
                "r2_trend": float(r2_trend),
                "performance_improving": rmse_trend < 0 and r2_trend > 0
            }
        
        return report
    
    def _send_performance_metrics(self, performance_record: Dict):
        """Send performance metrics to CloudWatch"""
        try:
            metrics = performance_record["metrics"]
            
            metric_data = [
                {
                    'MetricName': 'BatchRMSE',
                    'Value': metrics["rmse"],
                    'Unit': 'None',
                    'Dimensions': [
                        {
                            'Name': 'ModelVersion',
                            'Value': performance_record["model_version"]
                        }
                    ]
                },
                {
                    'MetricName': 'BatchMAE',
                    'Value': metrics["mae"],
                    'Unit': 'None',
                    'Dimensions': [
                        {
                            'Name': 'ModelVersion',
                            'Value': performance_record["model_version"]
                        }
                    ]
                },
                {
                    'MetricName': 'BatchR2Score',
                    'Value': metrics["r2_score"],
                    'Unit': 'None',
                    'Dimensions': [
                        {
                            'Name': 'ModelVersion',
                            'Value': performance_record["model_version"]
                        }
                    ]
                }
            ]
            
            self.cloudwatch.put_metric_data(
                Namespace='SolarPower/ModelPerformance',
                MetricData=metric_data
            )
            
        except Exception as e:
            print(f"Error sending performance metrics: {e}")

class ModelComparator:
    """Compare performance between different model versions"""
    
    def __init__(self, performance_tracker: PerformanceTracker):
        self.tracker = performance_tracker
    
    def compare_models(self, 
                      model_versions: List[str],
                      days_back: int = 30) -> Dict:
        """Compare performance between model versions"""
        
        comparison_result = {
            "comparison_date": datetime.now().isoformat(),
            "time_period_days": days_back,
            "models_compared": model_versions,
            "comparison_metrics": {}
        }
        
        for version in model_versions:
            history = self.tracker.get_performance_history(version, days_back)
            
            if not history.empty:
                comparison_result["comparison_metrics"][version] = {
                    "batch_count": len(history),
                    "avg_rmse": float(history['rmse'].mean()),
                    "avg_mae": float(history['mae'].mean()),
                    "avg_r2": float(history['r2_score'].mean()),
                    "rmse_std": float(history['rmse'].std()),
                    "consistency_score": float(1 / (1 + history['rmse'].std()))
                }
        
        # Determine best model
        if comparison_result["comparison_metrics"]:
            best_model = min(
                comparison_result["comparison_metrics"].items(),
                key=lambda x: x[1]["avg_rmse"]
            )
            comparison_result["best_model"] = {
                "version": best_model[0],
                "reason": "Lowest average RMSE"
            }
        
        return comparison_result

# Example usage
if __name__ == "__main__":
    # Initialize performance tracker
    tracker = PerformanceTracker("your-performance-bucket")
    
    # Example: Log performance
    # predictions = np.random.normal(2.5, 0.5, 100)
    # actuals = np.random.normal(2.5, 0.5, 100)
    # features = pd.DataFrame({'feature1': np.random.randn(100)})
    
    # batch_id = tracker.log_prediction_batch(
    #     predictions=predictions,
    #     actuals=actuals,
    #     features=features,
    #     model_version="1.0"
    # )
    
    print("Performance tracking system initialized successfully!")
