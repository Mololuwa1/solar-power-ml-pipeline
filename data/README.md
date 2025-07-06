# Solar Power Generation Dataset

This directory contains the data used for training and evaluating the solar power generation prediction models.

## Directory Structure

```
data/
├── raw/                    # Sample raw data files
│   ├── ZoneA1.csv         # Solar panel generation data for Zone A1
│   ├── Temperature_2021.csv   # Temperature measurements for 2021
│   ├── Irradiance_2021.csv    # Solar irradiance data for 2021
│   └── RelativeHumidity_2021.csv # Humidity measurements for 2021
├── processed/             # Processed and feature-engineered data
│   ├── processed_solar_data.csv   # Complete processed dataset (47MB)
│   ├── feature_scaler.pkl         # Fitted StandardScaler for features
│   └── feature_info.json         # Feature metadata and statistics
└── README.md             # This file
```

## Data Description

### Raw Data Files (Sample)
- **ZoneA1.csv**: Contains timestamp, generation (kWh), and power (W) measurements from solar panels in Zone A1
- **Temperature_2021.csv**: Hourly temperature measurements in Celsius
- **Irradiance_2021.csv**: Solar irradiance measurements in W/m²
- **RelativeHumidity_2021.csv**: Relative humidity measurements as percentage

### Processed Data
- **processed_solar_data.csv**: 
  - Combined dataset with all features and target variable
  - Size: ~47MB with 100,000+ samples
  - Features: 15 engineered features including temporal, weather, and lag features
  - Target: `generation` (kWh) - the variable we predict

### Feature Engineering Applied
1. **Temporal Features**: Hour, day of week, month, season
2. **Weather Features**: Temperature, humidity, irradiance, pressure, wind
3. **Lag Features**: Previous hour generation, rolling averages
4. **Interaction Features**: Temperature × irradiance, humidity × temperature
5. **Cyclical Encoding**: Sin/cos transformations for temporal features

### Data Quality
- **Missing Values**: Handled through interpolation and forward-fill
- **Outliers**: Detected and capped using IQR method
- **Normalization**: StandardScaler applied to all features
- **Data Range**: 2021-2023 with 15-minute intervals

## Usage

### Loading Processed Data
```python
import pandas as pd
import joblib

# Load processed data
data = pd.read_csv('data/processed/processed_solar_data.csv')

# Load feature scaler
scaler = joblib.load('data/processed/feature_scaler.pkl')

# Load feature metadata
import json
with open('data/processed/feature_info.json', 'r') as f:
    feature_info = json.load(f)
```

### Data Statistics
- **Total Samples**: 100,000+
- **Features**: 15 engineered features
- **Target Variable**: generation (kWh)
- **Time Range**: 2021-2023
- **Frequency**: 15-minute intervals
- **Missing Data**: <0.1% after preprocessing

### Model Performance on This Data
- **Neural Network**: RMSE 0.2935, R² 0.9905
- **XGBoost**: RMSE 0.4123, R² 0.9742  
- **Random Forest**: RMSE 0.4310, R² 0.9721

## Data Sources
The original dataset includes measurements from:
- Multiple solar panel zones (A1-A7, J1-J2, L1-L2, etc.)
- Weather stations providing meteorological data
- Inverter measurements for power quality monitoring
- Building-specific solar installations

## Notes
- The complete raw dataset (100+ files, ~500MB) is not included in this repository
- Sample raw files are provided for reference and testing
- For the complete dataset, contact the repository maintainer
- All data has been anonymized and aggregated for privacy

