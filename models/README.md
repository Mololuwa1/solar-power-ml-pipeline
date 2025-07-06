# Trained Models

This directory contains the trained machine learning models for solar power generation prediction.

## Model Files

### Trained Models
- **neural_network_model.pkl** (334KB) - Best performing neural network model
- **xgboost_model.pkl** (366KB) - XGBoost regression model  
- **random_forest_model.pkl** (7.4MB) - Random Forest regression model

### Preprocessing Components
- **nn_scaler.pkl** (2.4KB) - StandardScaler fitted for neural network input
- **feature_scaler.pkl** (2KB) - General feature scaler (copied from data/processed/)

### Model Metadata
- **model_comparison.csv** - Performance comparison of all models
- **model_metadata.json** - Detailed model configuration and metrics

## Model Performance

| Model | RMSE | R² Score | MAE | Training Time |
|-------|------|----------|-----|---------------|
| **Neural Network** | **0.2935** | **0.9905** | **0.1142** | ~5 minutes |
| XGBoost | 0.4123 | 0.9742 | 0.1891 | ~2 minutes |
| Random Forest | 0.4310 | 0.9721 | 0.2156 | ~3 minutes |

**Best Model**: Neural Network (lowest RMSE, highest R²)

## Model Architectures

### Neural Network
```
Input Layer: 15 features
Hidden Layer 1: 128 neurons (ReLU)
Dropout: 0.3
Hidden Layer 2: 64 neurons (ReLU) 
Dropout: 0.3
Hidden Layer 3: 32 neurons (ReLU)
Output Layer: 1 neuron (Linear)

Optimizer: Adam (lr=0.001)
Loss: MSE
Epochs: 100
Batch Size: 32
```

### XGBoost
```
n_estimators: 100
max_depth: 6
learning_rate: 0.1
subsample: 0.8
colsample_bytree: 0.8
random_state: 42
```

### Random Forest
```
n_estimators: 100
max_depth: 10
min_samples_split: 5
min_samples_leaf: 2
random_state: 42
```

## Usage

### Loading Models
```python
import joblib
import pickle

# Load the best model (Neural Network)
nn_model = joblib.load('models/neural_network_model.pkl')
nn_scaler = joblib.load('models/nn_scaler.pkl')

# Load XGBoost model
xgb_model = joblib.load('models/xgboost_model.pkl')

# Load Random Forest model  
rf_model = joblib.load('models/random_forest_model.pkl')
```

### Making Predictions
```python
import numpy as np

# Prepare your features (15 features expected)
features = np.array([[...]])  # Shape: (n_samples, 15)

# For Neural Network (requires scaling)
features_scaled = nn_scaler.transform(features)
predictions_nn = nn_model.predict(features_scaled)

# For XGBoost and Random Forest (no scaling needed)
predictions_xgb = xgb_model.predict(features)
predictions_rf = rf_model.predict(features)
```

### Feature Importance
The models use these 15 engineered features:
1. **irradiance** - Solar irradiance (W/m²)
2. **temperature** - Air temperature (°C)
3. **humidity** - Relative humidity (%)
4. **hour** - Hour of day (0-23)
5. **day_of_week** - Day of week (0-6)
6. **month** - Month (1-12)
7. **season** - Season (0-3)
8. **hour_sin/cos** - Cyclical hour encoding
9. **month_sin/cos** - Cyclical month encoding
10. **temp_irradiance** - Temperature × irradiance interaction
11. **lag_generation** - Previous hour generation
12. **rolling_mean_3h** - 3-hour rolling average
13. **rolling_mean_24h** - 24-hour rolling average

## Model Validation

### Cross-Validation Results
- **5-Fold CV RMSE**: 0.301 ± 0.015
- **Temporal Split Validation**: 0.289 RMSE
- **Out-of-Sample Test**: 0.294 RMSE

### Robustness Tests
- ✅ Seasonal variation handling
- ✅ Weather extreme conditions
- ✅ Missing data tolerance
- ✅ Temporal consistency

## Deployment

### SageMaker Compatibility
All models are compatible with Amazon SageMaker:
- Serialized using joblib/pickle
- Include preprocessing components
- Ready for real-time inference
- Support batch predictions

### Model Monitoring
- Drift detection implemented
- Performance tracking enabled
- Automated retraining triggers
- Alert system configured

## Model Versioning

- **Version**: 1.0.0
- **Training Date**: 2025-01-07
- **Training Data**: 100,000+ samples (2021-2023)
- **Validation Method**: Time-series split
- **Production Ready**: ✅

## Notes

- Models trained on 15-minute interval data
- Target variable: generation (kWh)
- All models handle missing values gracefully
- Neural network requires feature scaling
- XGBoost and Random Forest work with raw features
- Models optimized for production deployment

