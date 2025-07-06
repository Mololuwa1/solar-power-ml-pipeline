# Model Monitoring and Versioning System

This system provides comprehensive monitoring, versioning, and governance capabilities for the solar power generation ML models.

## Components

### 1. Model Versioning (`versioning/model_registry.py`)
- Centralized model registry
- Version management and promotion
- Deployment tracking
- Model metadata storage

### 2. Drift Detection (`drift_detection/drift_detector.py`)
- Data drift detection using statistical tests
- Model performance drift monitoring
- Automated alerting for drift events
- Continuous monitoring capabilities

### 3. Performance Tracking (`performance_tracking/performance_tracker.py`)
- Real-time performance logging
- Historical performance analysis
- Model comparison capabilities
- Trend analysis and reporting

### 4. Logging System (`logging/ml_logger.py`)
- Centralized logging for all ML operations
- CloudWatch integration
- Audit logging for compliance
- Error tracking and debugging

### 5. Monitoring Dashboard (`dashboards/monitoring_dashboard.json`)
- Real-time performance visualization
- Drift detection metrics
- Endpoint health monitoring
- Custom alerts and notifications

## Key Features

### Model Registry
- **Version Control**: Track all model versions with metadata
- **Promotion Workflow**: Staging → Production promotion process
- **Deployment History**: Track where and when models are deployed
- **Performance Comparison**: Compare metrics across versions

### Drift Detection
- **Statistical Tests**: Kolmogorov-Smirnov test for data drift
- **Performance Monitoring**: Track RMSE degradation over time
- **Automated Alerts**: SNS notifications for drift events
- **Baseline Comparison**: Compare new data against training baseline

### Performance Tracking
- **Batch Logging**: Log predictions and actuals for analysis
- **Metric Calculation**: RMSE, MAE, MAPE, R² tracking
- **Trend Analysis**: Identify performance trends over time
- **Reporting**: Generate comprehensive performance reports

### Comprehensive Logging
- **Event Logging**: Training, deployment, prediction events
- **Error Tracking**: Detailed error logging with context
- **Audit Trail**: Compliance-ready audit logging
- **CloudWatch Integration**: Centralized log management

## Usage Examples

### Model Registry
```python
from versioning.model_registry import ModelRegistry, ModelVersionManager

# Initialize registry
registry = ModelRegistry("your-model-registry-bucket")
version_manager = ModelVersionManager(registry)

# Register new model
model_id = version_manager.create_model_version(
    model_name="solar-power-generation",
    model_artifacts_path="s3://bucket/model.tar.gz",
    training_job_name="training-job-123",
    performance_metrics={"rmse": 0.29, "r2": 0.99}
)

# Promote to production
registry.promote_model("solar-power-generation", "1.0", "production")
```

### Drift Detection
```python
from drift_detection.drift_detector import DriftDetector

# Initialize detector
detector = DriftDetector("monitoring-bucket", "baseline/data.csv")

# Detect data drift
drift_result = detector.detect_data_drift(
    new_data=new_df,
    features=["Irradiance_mean", "Temperature_mean"],
    threshold=0.05
)

# Detect model drift
model_drift = detector.detect_model_drift(
    predictions=predictions,
    actuals=actuals,
    baseline_rmse=0.29,
    threshold_percentage=10.0
)
```

### Performance Tracking
```python
from performance_tracking.performance_tracker import PerformanceTracker

# Initialize tracker
tracker = PerformanceTracker("performance-bucket")

# Log prediction batch
batch_id = tracker.log_prediction_batch(
    predictions=predictions,
    actuals=actuals,
    features=features_df,
    model_version="1.0"
)

# Generate performance report
report = tracker.generate_performance_report(
    model_version="1.0",
    days_back=30
)
```

### Logging
```python
from logging.ml_logger import MLPipelineLogger

# Initialize logger
logger = MLPipelineLogger()

# Log training start
logger.log_training_start(
    training_job_name="solar-training-123",
    model_config={"model_type": "neural_network"},
    data_config={"data_source": "s3://bucket/data"}
)

# Log error
logger.log_error(
    error_type="DataValidationError",
    error_message="Missing required features",
    context={"missing_features": ["temperature", "irradiance"]}
)
```

## Monitoring Metrics

### Performance Metrics
- **BatchRMSE**: Root Mean Square Error per prediction batch
- **BatchMAE**: Mean Absolute Error per prediction batch
- **BatchR2Score**: R² score per prediction batch
- **ModelLatency**: Endpoint response time
- **ModelInvocations**: Number of predictions made

### Drift Metrics
- **DataDriftScore**: Statistical measure of data drift
- **DriftDetected**: Binary indicator of drift detection
- **PerformanceDrift**: Percentage change in model performance

### Operational Metrics
- **EndpointHealth**: Endpoint availability and health
- **ErrorRates**: 4XX and 5XX error rates
- **ThroughputMetrics**: Requests per second

## Alerting

### Drift Alerts
- Data drift detected (p-value < threshold)
- Model performance degradation (>10% RMSE increase)
- Feature distribution changes

### Performance Alerts
- High endpoint latency (>5 seconds)
- High error rates (>5%)
- Low prediction accuracy (R² < 0.9)

### Operational Alerts
- Endpoint failures
- Training job failures
- Data pipeline issues

## Best Practices

### Model Versioning
1. **Semantic Versioning**: Use major.minor.patch format
2. **Metadata Tracking**: Store comprehensive model metadata
3. **Promotion Gates**: Require performance validation before promotion
4. **Rollback Capability**: Maintain ability to rollback to previous versions

### Monitoring
1. **Baseline Establishment**: Establish performance baselines
2. **Regular Review**: Review monitoring data regularly
3. **Threshold Tuning**: Adjust alert thresholds based on experience
4. **Documentation**: Document all monitoring procedures

### Logging
1. **Structured Logging**: Use consistent log formats
2. **Appropriate Levels**: Use correct log levels (INFO, WARNING, ERROR)
3. **Context Information**: Include relevant context in logs
4. **Retention Policies**: Set appropriate log retention periods

## Deployment

### Prerequisites
- AWS CLI configured
- Python 3.7+ with required packages
- S3 buckets for storage
- CloudWatch access
- SNS topic for alerts

### Setup Steps
1. **Configure AWS Resources**:
   ```bash
   # Create S3 buckets
   aws s3 mb s3://your-model-registry-bucket
   aws s3 mb s3://your-monitoring-bucket
   
   # Create SNS topic
   aws sns create-topic --name solar-monitoring-alerts
   ```

2. **Deploy Monitoring Components**:
   ```bash
   # Update configuration files
   # Deploy CloudWatch dashboard
   # Set up alerts and notifications
   ```

3. **Initialize Systems**:
   ```python
   # Initialize model registry
   # Set up baseline data
   # Configure monitoring thresholds
   ```

## Troubleshooting

### Common Issues
1. **Missing Baseline Data**: Ensure baseline data is uploaded to S3
2. **CloudWatch Permissions**: Verify IAM permissions for CloudWatch
3. **S3 Access**: Check S3 bucket permissions and policies
4. **SNS Configuration**: Verify SNS topic and subscription setup

### Debugging
- Check CloudWatch logs for detailed error messages
- Verify S3 bucket contents and permissions
- Test SNS notifications manually
- Review IAM roles and policies

## Security Considerations

### Data Protection
- Encrypt data at rest and in transit
- Use IAM roles with least privilege
- Implement data retention policies
- Audit data access regularly

### Access Control
- Restrict access to monitoring systems
- Use VPC endpoints for S3 access
- Enable CloudTrail for audit logging
- Implement multi-factor authentication

## Cost Optimization

### Storage Costs
- Use S3 lifecycle policies
- Compress log files
- Archive old monitoring data
- Clean up temporary files

### Compute Costs
- Right-size monitoring instances
- Use spot instances where appropriate
- Optimize CloudWatch metric frequency
- Implement cost alerts

## Support and Maintenance

### Regular Tasks
- Review monitoring thresholds monthly
- Update baseline data quarterly
- Archive old logs annually
- Test alert systems regularly

### Maintenance Schedule
- **Daily**: Check dashboard for anomalies
- **Weekly**: Review performance trends
- **Monthly**: Update monitoring thresholds
- **Quarterly**: Refresh baseline data
