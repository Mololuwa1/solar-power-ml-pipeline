# Solar Power Generation ML Pipeline

A complete end-to-end machine learning pipeline for predicting solar farm power generation, built with Python and compatible with Amazon SageMaker.

## 🌟 Overview

This repository contains a comprehensive machine learning solution for predicting solar power generation using weather data and solar panel characteristics. The pipeline includes data preprocessing, model training, evaluation, deployment, automated retraining, and comprehensive monitoring capabilities.

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │───▶│  Preprocessing  │───▶│ Model Training  │
│                 │    │   & Features    │    │   & Evaluation  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Monitoring    │◀───│   SageMaker     │◀───│   Deployment    │
│  & Versioning   │    │   Deployment    │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                       ┌─────────────────┐
                       │   Automated     │
                       │   Retraining    │
                       └─────────────────┘
```

## 📊 Key Features

### Data Processing
- **Comprehensive Data Integration**: Combines weather data (temperature, irradiance, humidity, wind) with solar panel generation data
- **Feature Engineering**: Time-based features, lag variables, rolling statistics, and cyclical encoding
- **Data Validation**: Automated data quality checks and missing value handling
- **Scalable Processing**: Efficient processing of large datasets with memory optimization

### Model Development
- **Multiple Algorithms**: XGBoost, Random Forest, and Neural Network implementations
- **Hyperparameter Optimization**: Automated hyperparameter tuning with cross-validation
- **Performance Evaluation**: Comprehensive metrics including RMSE, MAE, R², and residual analysis
- **Model Comparison**: Systematic comparison of different model architectures

### SageMaker Integration
- **Training Jobs**: Scalable training on AWS infrastructure
- **Model Registry**: Centralized model versioning and management
- **Endpoint Deployment**: Real-time inference endpoints with auto-scaling
- **Batch Transform**: Large-scale batch prediction capabilities

### Automated Retraining
- **Data Monitoring**: Automatic detection of new data in S3
- **Pipeline Orchestration**: Step Functions workflow for end-to-end automation
- **Performance Validation**: Automated model validation before deployment
- **Blue/Green Deployment**: Safe model updates with rollback capabilities

### Monitoring & Governance
- **Drift Detection**: Statistical tests for data and model drift
- **Performance Tracking**: Real-time monitoring of model performance
- **Audit Logging**: Comprehensive logging for compliance and debugging
- **Alert System**: Automated notifications for anomalies and failures

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- AWS CLI configured
- SageMaker execution role
- S3 bucket for data storage

### Installation
```bash
# Clone the repository
git clone https://github.com/your-username/solar-power-ml-pipeline.git
cd solar-power-ml-pipeline

# Install dependencies
pip install -r requirements.txt

# Set up AWS credentials
aws configure
```

### Basic Usage
```bash
# 1. Data preprocessing
python src/data/preprocessing_utils.py

# 2. Model training
python src/models/train_models.py

# 3. Model evaluation
jupyter notebook notebooks/04_model_evaluation.ipynb

# 4. Deploy to SageMaker
cd sagemaker_deployment
python scripts/deploy.py
```

## 📁 Repository Structure

```
solar-power-ml-pipeline/
├── notebooks/                     # Jupyter notebooks for exploration and analysis
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_model_development.ipynb
│   └── 04_model_evaluation.ipynb
├── src/                           # Source code
│   ├── data/                      # Data processing utilities
│   ├── models/                    # Model training and evaluation
│   ├── evaluation/                # Model evaluation utilities
│   └── utils/                     # General utilities
├── sagemaker_deployment/          # SageMaker deployment files
│   ├── code/                      # Training and inference scripts
│   ├── config/                    # Deployment configurations
│   └── scripts/                   # Deployment automation
├── automated_retraining/          # Automated retraining system
│   ├── pipelines/                 # SageMaker Pipelines
│   ├── step_functions/            # Step Functions workflows
│   ├── lambda_functions/          # Lambda functions
│   └── monitoring/                # CloudWatch monitoring
├── model_monitoring/              # Model monitoring and versioning
│   ├── versioning/                # Model registry
│   ├── drift_detection/           # Drift detection
│   ├── performance_tracking/      # Performance monitoring
│   └── logging/                   # Logging system
├── data/                          # Data storage
│   ├── raw/                       # Raw data files
│   ├── processed/                 # Processed datasets
│   └── baseline/                  # Baseline data for monitoring
├── models/                        # Trained models
│   ├── trained/                   # Model artifacts
│   └── artifacts/                 # Additional model files
├── tests/                         # Test suite
│   ├── unit/                      # Unit tests
│   ├── integration/               # Integration tests
│   └── data/                      # Data validation tests
├── docs/                          # Documentation
├── scripts/                       # Utility scripts
├── config/                        # Configuration files
└── .github/                       # GitHub workflows
    └── workflows/                 # CI/CD pipelines
```

## 🔧 Configuration

### Environment Variables
```bash
export AWS_DEFAULT_REGION=us-east-1
export SAGEMAKER_ROLE_ARN=arn:aws:iam::YOUR_ACCOUNT:role/SageMakerExecutionRole
export S3_BUCKET=your-solar-data-bucket
export MODEL_REGISTRY_BUCKET=your-model-registry-bucket
```

### Model Configuration
Key parameters can be configured in `config/model_config.json`:
```json
{
  "model_type": "neural_network",
  "hyperparameters": {
    "hidden_layers": [100, 50, 25],
    "alpha": 0.001,
    "max_iter": 300
  },
  "performance_thresholds": {
    "min_r2_score": 0.95,
    "max_rmse": 0.5
  }
}
```

## 📈 Performance

### Model Results
The best performing model (Neural Network) achieves:
- **RMSE**: 0.2935 kWh
- **MAE**: 0.1142 kWh  
- **R² Score**: 0.9905
- **MAPE**: 8.2%

### Benchmark Comparison
| Model | RMSE | MAE | R² | Training Time |
|-------|------|-----|----|--------------| 
| Neural Network | 0.2935 | 0.1142 | 0.9905 | 45s |
| XGBoost | 0.4123 | 0.1402 | 0.9742 | 12s |
| Random Forest | 0.4310 | 0.1345 | 0.9721 | 8s |

## 🔄 Automated Workflows

### Retraining Pipeline
The automated retraining system:
1. **Monitors** S3 for new data daily
2. **Validates** data quality and schema
3. **Trains** new model with updated dataset
4. **Evaluates** performance against thresholds
5. **Deploys** model if performance improves
6. **Notifies** stakeholders of results

### CI/CD Pipeline
GitHub Actions workflow for:
- Code quality checks (linting, formatting)
- Unit and integration testing
- Model validation tests
- Automated deployment to staging
- Performance regression testing

## 📊 Monitoring

### Key Metrics
- **Model Performance**: RMSE, MAE, R² tracking over time
- **Data Quality**: Missing values, outliers, distribution changes
- **Operational**: Endpoint latency, throughput, error rates
- **Business**: Prediction accuracy, cost savings, energy forecasting

### Alerting
Automated alerts for:
- Model performance degradation (>10% RMSE increase)
- Data drift detection (p-value < 0.05)
- Endpoint failures or high latency
- Training job failures

## 🧪 Testing

### Test Coverage
- **Unit Tests**: Individual function testing
- **Integration Tests**: End-to-end pipeline testing
- **Data Tests**: Data quality and schema validation
- **Model Tests**: Model performance and behavior testing

### Running Tests
```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/data/
```

## 📚 Documentation

### Available Documentation
- [Data Processing Guide](docs/data_processing.md)
- [Model Development Guide](docs/model_development.md)
- [Deployment Guide](docs/deployment.md)
- [Monitoring Guide](docs/monitoring.md)
- [API Reference](docs/api_reference.md)

### Jupyter Notebooks
Interactive notebooks for:
- Data exploration and visualization
- Feature engineering experiments
- Model development and tuning
- Performance analysis and interpretation

## 🤝 Contributing

### Development Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

### Contribution Guidelines
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Weather data provided by [Hong Kong Observatory](https://www.hko.gov.hk/)
- Solar panel data from university campus installations
- AWS SageMaker team for excellent documentation and examples
- Open source community for the amazing ML libraries

## 📞 Support

For questions, issues, or contributions:
- **Issues**: [GitHub Issues](https://github.com/your-username/solar-power-ml-pipeline/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/solar-power-ml-pipeline/discussions)
- **Email**: your-email@example.com

## 🔗 Related Projects

- [Solar Power Forecasting Dashboard](https://github.com/your-username/solar-dashboard)
- [Weather Data Pipeline](https://github.com/your-username/weather-pipeline)
- [Energy Management System](https://github.com/your-username/energy-management)

---

**Built with ❤️ by [Your Name] | Powered by AWS SageMaker**
