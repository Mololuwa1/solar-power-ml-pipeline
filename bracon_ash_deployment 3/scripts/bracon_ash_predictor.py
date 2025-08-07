#!/usr/bin/env python3
"""
Production prediction script for Bracon Ash solar farm
"""

import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class BraconAshPredictor:
    """Production predictor for Bracon Ash solar farm"""
    
    def __init__(self, model_dir="models"):
        """Initialize predictor with trained models"""
        self.model_dir = model_dir
        self.rf_model = None
        self.xgb_model = None
        self.scaler = None
        self.feature_columns = None
        self.load_models()
    
    def load_models(self):
        """Load trained models and supporting data"""
        try:
            self.rf_model = joblib.load(f"{self.model_dir}/bracon_ash_rf_model.pkl")
            self.xgb_model = joblib.load(f"{self.model_dir}/bracon_ash_xgb_model.pkl")
            self.scaler = joblib.load(f"{self.model_dir}/bracon_ash_scaler.pkl")
            
            with open(f"{self.model_dir}/bracon_ash_features.json", 'r') as f:
                self.feature_columns = json.load(f)
            
            print("✅ Models loaded successfully")
            print(f"   Features: {len(self.feature_columns)}")
            
        except Exception as e:
            raise Exception(f"Failed to load models: {str(e)}")
    
    def preprocess_data(self, data):
        """Preprocess input data to match training format"""
        # Create working copy
        processed = data.copy()
        
        # Ensure timestamp column
        if 'timestamp' not in processed.columns:
            if 'Date' in processed.columns and 'Time' in processed.columns:
                processed['timestamp'] = pd.to_datetime(
                    processed['Date'].astype(str) + ' ' + processed['Time'].astype(str),
                    dayfirst=True
                )
            else:
                raise ValueError("No timestamp information found")
        
        # Target variable (if present)
        target_col = 'Bracon Ash - Total - Total Meters Energy (kWh)'
        if target_col in processed.columns:
            processed['generation'] = processed[target_col]
        else:
            processed['generation'] = 0  # Placeholder for prediction
        
        # Temporal features
        processed['year'] = processed['timestamp'].dt.year
        processed['month'] = processed['timestamp'].dt.month
        processed['day'] = processed['timestamp'].dt.day
        processed['hour'] = processed['timestamp'].dt.hour
        processed['minute'] = processed['timestamp'].dt.minute
        processed['day_of_week'] = processed['timestamp'].dt.dayofweek
        processed['day_of_year'] = processed['timestamp'].dt.dayofyear
        processed['week_of_year'] = processed['timestamp'].dt.isocalendar().week
        
        # Cyclical encodings
        processed['hour_sin'] = np.sin(2 * np.pi * processed['hour'] / 24)
        processed['hour_cos'] = np.cos(2 * np.pi * processed['hour'] / 24)
        processed['month_sin'] = np.sin(2 * np.pi * processed['month'] / 12)
        processed['month_cos'] = np.cos(2 * np.pi * processed['month'] / 12)
        processed['day_of_week_sin'] = np.sin(2 * np.pi * processed['day_of_week'] / 7)
        processed['day_of_week_cos'] = np.cos(2 * np.pi * processed['day_of_week'] / 7)
        processed['day_of_year_sin'] = np.sin(2 * np.pi * processed['day_of_year'] / 365)
        processed['day_of_year_cos'] = np.cos(2 * np.pi * processed['day_of_year'] / 365)
        
        # Solar features
        def calculate_solar_elevation(timestamp, latitude=52.5):
            day_of_year = timestamp.dt.dayofyear
            hour = timestamp.dt.hour + timestamp.dt.minute / 60.0
            
            declination = 23.45 * np.sin(np.radians(360 * (284 + day_of_year) / 365))
            hour_angle = 15 * (hour - 12)
            
            elevation = np.arcsin(
                np.sin(np.radians(declination)) * np.sin(np.radians(latitude)) +
                np.cos(np.radians(declination)) * np.cos(np.radians(latitude)) * 
                np.cos(np.radians(hour_angle))
            )
            
            return np.degrees(elevation)
        
        processed['solar_elevation'] = calculate_solar_elevation(processed['timestamp'])
        processed['is_daytime'] = (processed['solar_elevation'] > 0).astype(int)
        processed['is_peak_sun'] = ((processed['hour'] >= 10) & (processed['hour'] <= 14)).astype(int)
        processed['is_weekend'] = (processed['day_of_week'] >= 5).astype(int)
        
        # Weather features mapping
        weather_mapping = {
            'irradiance': 'Bracon Ash - Weather Station CT03 - REF.CELL-1_IRRADIANCE (W/m2)',
            'temperature': 'Bracon Ash - Weather Station CT03 - AMBIENT TEMPERATURE-1 (ºC)',
            'module_temp': 'Bracon Ash - Weather Station CT03 - PT1000-MODULE-1 (ºC)',
            'ref_cell_temp': 'Bracon Ash - Weather Station CT03 - REF.CELL-1_TEMPERATURE (ºC)',
            'ghi_b': 'Bracon Ash - Weather Station CT03 - SMP10_IRRADIANCE-GHI-B (W/m2)',
            'ghi_f': 'Bracon Ash - Weather Station CT03 - SMP10_IRRADIANCE-GHI-F (W/m2)',
            'poa': 'Bracon Ash - Weather Station CT03 - SMP10_IRRADIANCE-POA-1 (W/m2)',
            'wind_speed': 'Bracon Ash - Weather Station CT03 - WS (m/s)',
            'wind_direction': 'Bracon Ash - Weather Station CT03 - WD (º)',
            'precipitation': 'Bracon Ash - Weather Station CT03 - ABS_PRECIPITATION (l/m2)'
        }
        
        for feature, col in weather_mapping.items():
            if col in processed.columns:
                processed[feature] = processed[col].fillna(processed[col].median())
            else:
                # Use default values if weather data is missing
                defaults = {
                    'irradiance': 0,
                    'temperature': 15,
                    'module_temp': 15,
                    'ref_cell_temp': 15,
                    'ghi_b': 0,
                    'ghi_f': 0,
                    'poa': 0,
                    'wind_speed': 2,
                    'wind_direction': 180,
                    'precipitation': 0
                }
                processed[feature] = defaults.get(feature, 0)
        
        # Interaction features
        processed['temp_irradiance'] = processed['temperature'] * processed['irradiance']
        processed['temp_squared'] = processed['temperature'] ** 2
        processed['irradiance_squared'] = processed['irradiance'] ** 2
        processed['wind_cooling'] = processed['wind_speed'] * processed['temperature']
        
        # Sort by timestamp for lag features
        processed = processed.sort_values('timestamp').reset_index(drop=True)
        
        # Lag features
        lag_periods = [1, 2, 4, 12, 24, 48]
        for lag in lag_periods:
            processed[f'generation_lag_{lag}'] = processed['generation'].shift(lag)
        
        # Rolling statistics
        windows = [4, 12, 24, 48, 96]
        for window in windows:
            processed[f'generation_rolling_mean_{window}'] = processed['generation'].rolling(
                window=window, min_periods=1).mean()
            processed[f'generation_rolling_std_{window}'] = processed['generation'].rolling(
                window=window, min_periods=1).std()
            processed[f'generation_rolling_max_{window}'] = processed['generation'].rolling(
                window=window, min_periods=1).max()
            processed[f'generation_rolling_min_{window}'] = processed['generation'].rolling(
                window=window, min_periods=1).min()
        
        # Weather lag features
        key_weather = ['irradiance', 'temperature']
        for weather in key_weather:
            for lag in [1, 2, 4]:
                processed[f'{weather}_lag_{lag}'] = processed[weather].shift(lag)
            
            for window in [4, 12, 24]:
                processed[f'{weather}_rolling_mean_{window}'] = processed[weather].rolling(
                    window=window, min_periods=1).mean()
        
        # Fill NaN values
        processed = processed.fillna(method='bfill').fillna(method='ffill').fillna(0)
        
        return processed
    
    def predict(self, data, model='random_forest'):
        """Make predictions on input data"""
        # Preprocess data
        processed_data = self.preprocess_data(data)
        
        # Extract features
        X = processed_data[self.feature_columns]
        
        # Make predictions
        if model == 'random_forest':
            predictions = self.rf_model.predict(X)
        elif model == 'xgboost':
            predictions = self.xgb_model.predict(X)
        else:
            raise ValueError("Model must be 'random_forest' or 'xgboost'")
        
        # Ensure non-negative predictions
        predictions = np.maximum(predictions, 0)
        
        # Create results dataframe
        results = pd.DataFrame({
            'timestamp': processed_data['timestamp'],
            'predicted_generation_kwh': predictions
        })
        
        # Add actual values if available
        if 'generation' in processed_data.columns and processed_data['generation'].sum() > 0:
            results['actual_generation_kwh'] = processed_data['generation']
            results['prediction_error_kwh'] = results['actual_generation_kwh'] - results['predicted_generation_kwh']
        
        return results
    
    def predict_future(self, start_time, hours=24, weather_data=None):
        """Predict future generation with weather forecast"""
        # Create future timestamps
        timestamps = pd.date_range(start=start_time, periods=hours*4, freq='15min')
        
        # Create base dataframe
        future_data = pd.DataFrame({
            'timestamp': timestamps
        })
        
        # Add weather data if provided, otherwise use seasonal defaults
        if weather_data is not None:
            # Merge with provided weather data
            future_data = future_data.merge(weather_data, on='timestamp', how='left')
        else:
            # Use seasonal defaults based on time of year
            month = start_time.month
            hour = timestamps.hour
            
            # Seasonal irradiance patterns
            if month in [6, 7, 8]:  # Summer
                max_irradiance = 1000
            elif month in [12, 1, 2]:  # Winter
                max_irradiance = 400
            else:  # Spring/Fall
                max_irradiance = 700
            
            # Daily irradiance pattern
            irradiance = np.maximum(0, max_irradiance * np.sin(np.pi * (hour - 6) / 12) * 
                                   (hour >= 6) * (hour <= 18))
            
            # Temperature patterns
            if month in [6, 7, 8]:  # Summer
                base_temp = 20
            elif month in [12, 1, 2]:  # Winter
                base_temp = 5
            else:  # Spring/Fall
                base_temp = 12
            
            temperature = base_temp + 8 * np.sin(2 * np.pi * hour / 24 - np.pi/2)
            
            # Add weather columns
            future_data['Bracon Ash - Weather Station CT03 - REF.CELL-1_IRRADIANCE (W/m2)'] = irradiance
            future_data['Bracon Ash - Weather Station CT03 - AMBIENT TEMPERATURE-1 (ºC)'] = temperature
        
        # Make predictions
        predictions = self.predict(future_data)
        
        return predictions

# Example usage
if __name__ == "__main__":
    # Initialize predictor
    predictor = BraconAshPredictor()
    
    # Example 1: Predict on historical data
    print("Example 1: Historical data prediction")
    sample_data = pd.read_excel("data/sample_data.xlsx")
    results = predictor.predict(sample_data.head(100))
    print(f"Predicted {len(results)} time points")
    print(f"Average prediction: {results['predicted_generation_kwh'].mean():.2f} kWh")
    
    # Example 2: Future prediction
    print("\nExample 2: Future prediction")
    future_results = predictor.predict_future(
        start_time=datetime.now(),
        hours=24
    )
    print(f"24-hour forecast generated: {len(future_results)} time points")
    print(f"Total predicted generation: {future_results['predicted_generation_kwh'].sum():.2f} kWh")
