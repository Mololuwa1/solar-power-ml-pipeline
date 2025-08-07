#!/usr/bin/env python3
"""
Monitoring and validation script for Bracon Ash solar farm model
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from bracon_ash_predictor import BraconAshPredictor
import json
from datetime import datetime, timedelta

class ModelMonitor:
    """Monitor model performance and data quality"""
    
    def __init__(self, predictor):
        self.predictor = predictor
        self.performance_history = []
    
    def validate_performance(self, actual_data):
        """Validate model performance on new data"""
        print("🔍 VALIDATING MODEL PERFORMANCE")
        print("-" * 40)
        
        # Make predictions
        results = self.predictor.predict(actual_data)
        
        # Calculate metrics
        if 'actual_generation_kwh' in results.columns:
            y_true = results['actual_generation_kwh']
            y_pred = results['predicted_generation_kwh']
            
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
            
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'samples': len(y_true),
                'rmse': rmse,
                'r2_score': r2,
                'mae': mae,
                'mape': mape
            }
            
            self.performance_history.append(metrics)
            
            print(f"✅ Performance Metrics:")
            print(f"   R² Score: {r2:.6f} ({r2*100:.4f}%)")
            print(f"   RMSE: {rmse:.4f} kWh")
            print(f"   MAE: {mae:.4f} kWh")
            print(f"   MAPE: {mape:.2f}%")
            
            # Performance alerts
            if r2 < 0.95:
                print("⚠️  WARNING: R² below 95% threshold")
            if rmse > 50:
                print("⚠️  WARNING: RMSE above 50 kWh threshold")
            
            return metrics
        else:
            print("❌ No actual values available for validation")
            return None
    
    def check_data_quality(self, data):
        """Check data quality and completeness"""
        print("\n🔍 CHECKING DATA QUALITY")
        print("-" * 40)
        
        issues = []
        
        # Check for missing timestamps
        if 'timestamp' not in data.columns and not ('Date' in data.columns and 'Time' in data.columns):
            issues.append("Missing timestamp information")
        
        # Check weather data completeness
        weather_cols = [col for col in data.columns if 'Weather Station' in col]
        missing_weather = []
        
        for col in weather_cols:
            missing_pct = data[col].isnull().mean() * 100
            if missing_pct > 10:
                missing_weather.append(f"{col}: {missing_pct:.1f}% missing")
        
        if missing_weather:
            issues.append(f"High missing weather data: {missing_weather}")
        
        # Check for data gaps
        if 'timestamp' in data.columns:
            time_diff = data['timestamp'].diff().dt.total_seconds() / 60
            expected_interval = 15  # 15 minutes
            gaps = time_diff[time_diff > expected_interval * 1.5]
            
            if len(gaps) > 0:
                issues.append(f"Found {len(gaps)} time gaps larger than expected")
        
        # Report results
        if issues:
            print("⚠️  Data Quality Issues Found:")
            for issue in issues:
                print(f"   - {issue}")
        else:
            print("✅ Data quality checks passed")
        
        return issues
    
    def generate_performance_report(self):
        """Generate performance monitoring report"""
        if not self.performance_history:
            print("No performance history available")
            return
        
        print("\n📊 PERFORMANCE MONITORING REPORT")
        print("-" * 50)
        
        recent_metrics = self.performance_history[-1]
        
        print(f"Latest Performance ({recent_metrics['timestamp'][:10]}):")
        print(f"   R² Score: {recent_metrics['r2_score']:.6f}")
        print(f"   RMSE: {recent_metrics['rmse']:.4f} kWh")
        print(f"   Samples: {recent_metrics['samples']}")
        
        if len(self.performance_history) > 1:
            prev_metrics = self.performance_history[-2]
            r2_change = recent_metrics['r2_score'] - prev_metrics['r2_score']
            rmse_change = recent_metrics['rmse'] - prev_metrics['rmse']
            
            print(f"\nChange from Previous:")
            print(f"   R² Change: {r2_change:+.6f}")
            print(f"   RMSE Change: {rmse_change:+.4f} kWh")
        
        # Save performance history
        with open('performance_history.json', 'w') as f:
            json.dump(self.performance_history, f, indent=2)
        
        print("\n✅ Performance history saved to performance_history.json")

# Example usage
if __name__ == "__main__":
    # Initialize predictor and monitor
    predictor = BraconAshPredictor()
    monitor = ModelMonitor(predictor)
    
    # Load sample data for validation
    sample_data = pd.read_excel("../data/sample_data.xlsx")
    
    # Check data quality
    monitor.check_data_quality(sample_data)
    
    # Validate performance
    metrics = monitor.validate_performance(sample_data.head(500))
    
    # Generate report
    monitor.generate_performance_report()
