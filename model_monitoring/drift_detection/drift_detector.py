#!/usr/bin/env python3
"""
Data and Model Drift Detection System
"""

import numpy as np
import pandas as pd
import boto3
import json
from datetime import datetime, timedelta
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, List, Tuple

class DriftDetector:
    """Detect data and model drift"""
    
    def __init__(self, s3_bucket: str, baseline_data_path: str):
        self.s3_client = boto3.client('s3')
        self.cloudwatch = boto3.client('cloudwatch')
        self.s3_bucket = s3_bucket
        self.baseline_data_path = baseline_data_path
        self.baseline_data = None
        self._load_baseline_data()
    
    def _load_baseline_data(self):
        """Load baseline data for comparison"""
        try:
            response = self.s3_client.get_object(
                Bucket=self.s3_bucket,
                Key=self.baseline_data_path
            )
            self.baseline_data = pd.read_csv(response['Body'])
            print(f"Loaded baseline data: {len(self.baseline_data)} samples")
        except Exception as e:
            print(f"Error loading baseline data: {e}")
    
    def detect_data_drift(self, new_data: pd.DataFrame, 
                         features: List[str],
                         threshold: float = 0.05) -> Dict:
        """Detect data drift using statistical tests"""
        
        if self.baseline_data is None:
            return {"error": "Baseline data not available"}
        
        drift_results = {
            "timestamp": datetime.now().isoformat(),
            "features_tested": len(features),
            "drift_detected": False,
            "feature_drift": {},
            "overall_drift_score": 0.0
        }
        
        drift_scores = []
        
        for feature in features:
            if feature in self.baseline_data.columns and feature in new_data.columns:
                # Kolmogorov-Smirnov test
                baseline_values = self.baseline_data[feature].dropna()
                new_values = new_data[feature].dropna()
                
                if len(baseline_values) > 0 and len(new_values) > 0:
                    ks_statistic, p_value = stats.ks_2samp(baseline_values, new_values)
                    
                    feature_drift = {
                        "ks_statistic": ks_statistic,
                        "p_value": p_value,
                        "drift_detected": p_value < threshold,
                        "baseline_mean": float(baseline_values.mean()),
                        "new_mean": float(new_values.mean()),
                        "baseline_std": float(baseline_values.std()),
                        "new_std": float(new_values.std())
                    }
                    
                    drift_results["feature_drift"][feature] = feature_drift
                    drift_scores.append(ks_statistic)
        
        # Calculate overall drift score
        if drift_scores:
            drift_results["overall_drift_score"] = np.mean(drift_scores)
            drift_results["drift_detected"] = any(
                f["drift_detected"] for f in drift_results["feature_drift"].values()
            )
        
        # Send metrics to CloudWatch
        self._send_drift_metrics(drift_results)
        
        return drift_results
    
    def detect_model_drift(self, 
                          predictions: np.ndarray,
                          actuals: np.ndarray,
                          baseline_rmse: float,
                          threshold_percentage: float = 10.0) -> Dict:
        """Detect model performance drift"""
        
        current_rmse = np.sqrt(mean_squared_error(actuals, predictions))
        current_mae = mean_absolute_error(actuals, predictions)
        
        # Calculate performance degradation
        rmse_change = ((current_rmse - baseline_rmse) / baseline_rmse) * 100
        
        drift_result = {
            "timestamp": datetime.now().isoformat(),
            "current_rmse": current_rmse,
            "baseline_rmse": baseline_rmse,
            "rmse_change_percentage": rmse_change,
            "current_mae": current_mae,
            "performance_drift_detected": rmse_change > threshold_percentage,
            "threshold_percentage": threshold_percentage
        }
        
        # Send metrics to CloudWatch
        self._send_performance_metrics(drift_result)
        
        return drift_result
    
    def _send_drift_metrics(self, drift_results: Dict):
        """Send drift metrics to CloudWatch"""
        try:
            self.cloudwatch.put_metric_data(
                Namespace='SolarPower/ModelMonitoring',
                MetricData=[
                    {
                        'MetricName': 'DataDriftScore',
                        'Value': drift_results["overall_drift_score"],
                        'Unit': 'None',
                        'Timestamp': datetime.now()
                    },
                    {
                        'MetricName': 'DriftDetected',
                        'Value': 1 if drift_results["drift_detected"] else 0,
                        'Unit': 'Count',
                        'Timestamp': datetime.now()
                    }
                ]
            )
        except Exception as e:
            print(f"Error sending drift metrics: {e}")
    
    def _send_performance_metrics(self, performance_results: Dict):
        """Send performance metrics to CloudWatch"""
        try:
            self.cloudwatch.put_metric_data(
                Namespace='SolarPower/ModelMonitoring',
                MetricData=[
                    {
                        'MetricName': 'ModelRMSE',
                        'Value': performance_results["current_rmse"],
                        'Unit': 'None',
                        'Timestamp': datetime.now()
                    },
                    {
                        'MetricName': 'PerformanceDrift',
                        'Value': performance_results["rmse_change_percentage"],
                        'Unit': 'Percent',
                        'Timestamp': datetime.now()
                    }
                ]
            )
        except Exception as e:
            print(f"Error sending performance metrics: {e}")

class ContinuousMonitor:
    """Continuous monitoring system"""
    
    def __init__(self, drift_detector: DriftDetector):
        self.drift_detector = drift_detector
        self.sns_client = boto3.client('sns')
        self.topic_arn = "arn:aws:sns:us-east-1:YOUR_ACCOUNT:solar-monitoring-alerts"
    
    def monitor_endpoint_data(self, endpoint_name: str, 
                            time_window_hours: int = 24) -> Dict:
        """Monitor endpoint data for drift"""
        
        # This would typically pull data from SageMaker Data Capture
        # For now, we'll simulate the monitoring process
        
        monitoring_result = {
            "endpoint_name": endpoint_name,
            "monitoring_window": f"{time_window_hours} hours",
            "timestamp": datetime.now().isoformat(),
            "status": "healthy"
        }
        
        # In a real implementation, you would:
        # 1. Pull captured data from S3
        # 2. Run drift detection
        # 3. Check performance metrics
        # 4. Send alerts if needed
        
        return monitoring_result
    
    def send_alert(self, alert_type: str, message: str, severity: str = "WARNING"):
        """Send monitoring alert"""
        
        alert_message = {
            "alert_type": alert_type,
            "severity": severity,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "service": "solar-power-ml-monitoring"
        }
        
        try:
            self.sns_client.publish(
                TopicArn=self.topic_arn,
                Message=json.dumps(alert_message, indent=2),
                Subject=f"Solar Power ML Alert: {alert_type}"
            )
            print(f"Alert sent: {alert_type}")
        except Exception as e:
            print(f"Error sending alert: {e}")

# Example usage
if __name__ == "__main__":
    # Initialize drift detector
    drift_detector = DriftDetector(
        s3_bucket="your-monitoring-bucket",
        baseline_data_path="baseline/solar_data_baseline.csv"
    )
    
    # Initialize continuous monitor
    monitor = ContinuousMonitor(drift_detector)
    
    # Example: Detect data drift
    # new_data = pd.read_csv("new_solar_data.csv")
    # features = ["Irradiance_mean", "Temperature_mean", "hour", "month"]
    # drift_result = drift_detector.detect_data_drift(new_data, features)
    # print(f"Drift detection result: {drift_result}")
    
    print("Drift detection system initialized successfully!")
