"""
Efficient preprocessing utilities for solar power generation data
"""

import pandas as pd
import numpy as np
import os
import glob
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
import joblib
import json

def load_sample_data(data_dir='/home/ubuntu/upload/', n_stations=5):
    """Load a sample of the data for efficient processing"""
    
    # Get all files
    all_files = glob.glob(os.path.join(data_dir, '*.csv'))
    
    # Filter power generation files
    power_files = [f for f in all_files if not any(weather in f for weather in 
                   ['Temperature', 'Humidity', 'Irradiance', 'Wind', 'Visibility', 
                    'SeaLevelPressure', 'RelativeHumidity', 'Rainfall']) 
                   and not 'Inverter' in f]
    
    # Take first n_stations for sample
    sample_power_files = power_files[:n_stations]
    
    # Load power data
    power_data_list = []
    for file in sample_power_files:
        try:
            df = pd.read_csv(file)
            df['Time'] = pd.to_datetime(df['Time'])
            df['station'] = os.path.basename(file).replace('.csv', '')
            power_data_list.append(df)
            print(f"Loaded {os.path.basename(file)}: {df.shape[0]} records")
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    combined_power = pd.concat(power_data_list, ignore_index=True)
    combined_power = combined_power.sort_values(['station', 'Time']).reset_index(drop=True)
    
    # Load key weather data (Irradiance and Temperature)
    weather_files = [f for f in all_files if any(weather in f for weather in 
                    ['Irradiance', 'Temperature'])]
    
    weather_data = {}
    for file in weather_files:
        weather_type = os.path.basename(file).split('_')[0]
        try:
            df = pd.read_csv(file)
            df['Time'] = pd.to_datetime(df['Time'])
            
            if weather_type not in weather_data:
                weather_data[weather_type] = []
            weather_data[weather_type].append(df)
            print(f"Loaded {os.path.basename(file)}: {df.shape[0]} records")
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    # Combine weather data
    combined_weather = {}
    for weather_type, dfs in weather_data.items():
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df = combined_df.sort_values('Time').reset_index(drop=True)
        combined_df = combined_df.drop_duplicates(subset=['Time']).reset_index(drop=True)
        combined_weather[weather_type] = combined_df
        print(f"Combined {weather_type}: {combined_df.shape[0]} records")
    
    return combined_power, combined_weather

def create_time_features(df, time_col='Time'):
    """Create time-based features"""
    df = df.copy()
    
    # Basic time features
    df['year'] = df[time_col].dt.year
    df['month'] = df[time_col].dt.month
    df['day'] = df[time_col].dt.day
    df['hour'] = df[time_col].dt.hour
    df['minute'] = df[time_col].dt.minute
    df['day_of_week'] = df[time_col].dt.dayofweek
    df['day_of_year'] = df[time_col].dt.dayofyear
    
    # Cyclical features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Solar position approximation
    df['solar_elevation'] = np.sin(2 * np.pi * (df['day_of_year'] - 81) / 365) * 23.45
    
    # Time categories
    df['is_daytime'] = ((df['hour'] >= 6) & (df['hour'] <= 18)).astype(int)
    df['is_peak_sun'] = ((df['hour'] >= 10) & (df['hour'] <= 14)).astype(int)
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    return df

def resample_weather_data(weather_data, target_freq='15T'):
    """Resample weather data to target frequency"""
    resampled_weather = {}
    
    for weather_type, df in weather_data.items():
        numeric_col = [col for col in df.columns if col != 'Time'][0]
        df_indexed = df.set_index('Time')
        
        # Resample with aggregations
        resampled = df_indexed.resample(target_freq).agg({
            numeric_col: ['mean', 'max', 'min', 'std']
        })
        
        # Flatten column names
        resampled.columns = [f"{weather_type}_{stat}" for stat in ['mean', 'max', 'min', 'std']]
        resampled = resampled.reset_index()
        
        # Fill NaN values
        numeric_cols = [col for col in resampled.columns if col != 'Time']
        resampled[numeric_cols] = resampled[numeric_cols].fillna(method='ffill').fillna(0)
        
        resampled_weather[weather_type] = resampled
        print(f"Resampled {weather_type}: {resampled.shape[0]} records")
    
    return resampled_weather

def create_lag_features(df, target_col, lags=[1, 2, 4, 24]):
    """Create lag features"""
    df = df.copy()
    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    return df

def create_rolling_features(df, target_col, windows=[4, 12, 24]):
    """Create rolling window features"""
    df = df.copy()
    for window in windows:
        df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window=window, min_periods=1).mean()
        df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window=window, min_periods=1).std()
    return df

def preprocess_data(output_dir='/home/ubuntu/processed_data/'):
    """Main preprocessing function"""
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading sample data...")
    power_data, weather_data = load_sample_data(n_stations=5)
    
    print("\nCreating time features...")
    power_data = create_time_features(power_data)
    
    print("\nResampling weather data...")
    resampled_weather = resample_weather_data(weather_data)
    
    print("\nMerging data...")
    merged_data = power_data.copy()
    for weather_type, weather_df in resampled_weather.items():
        merged_data = pd.merge(merged_data, weather_df, on='Time', how='left')
        print(f"Merged {weather_type}: {merged_data.shape}")
    
    print("\nHandling missing values...")
    # Remove rows with missing target
    merged_data = merged_data.dropna(subset=['generation(kWh)']).reset_index(drop=True)
    
    # Fill missing weather data
    weather_cols = [col for col in merged_data.columns if any(w in col for w in ['Temperature', 'Irradiance'])]
    merged_data[weather_cols] = merged_data[weather_cols].fillna(method='ffill').fillna(method='bfill')
    
    print("\nCreating lag and rolling features...")
    featured_data_list = []
    for station in merged_data['station'].unique():
        station_data = merged_data[merged_data['station'] == station].copy()
        station_data = create_lag_features(station_data, 'generation(kWh)')
        station_data = create_rolling_features(station_data, 'generation(kWh)')
        
        # Add weather lag features
        if 'Irradiance_mean' in station_data.columns:
            station_data = create_lag_features(station_data, 'Irradiance_mean', lags=[1, 2])
        
        featured_data_list.append(station_data)
        print(f"Processed {station}: {station_data.shape}")
    
    final_data = pd.concat(featured_data_list, ignore_index=True)
    
    # Remove rows with NaN in lag features (first few rows)
    final_data = final_data.dropna().reset_index(drop=True)
    
    print(f"\nFinal data shape: {final_data.shape}")
    
    # Prepare features and target
    exclude_cols = ['Time', 'generation(kWh)', 'power(W)', 'station']
    feature_cols = [col for col in final_data.columns if col not in exclude_cols]
    
    X = final_data[feature_cols].copy()
    y = final_data['generation(kWh)'].copy()
    
    # Scale features
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)
    
    # Create final dataset
    processed_data = final_data.copy()
    processed_data[feature_cols] = X_scaled
    
    # Save data
    processed_data.to_csv(os.path.join(output_dir, 'processed_solar_data.csv'), index=False)
    
    # Save metadata
    feature_info = {
        'feature_columns': feature_cols,
        'target_column': 'generation(kWh)',
        'metadata_columns': ['Time', 'station'],
        'total_features': len(feature_cols),
        'total_samples': len(processed_data),
        'stations': list(processed_data['station'].unique()),
        'date_range': [str(processed_data['Time'].min()), str(processed_data['Time'].max())]
    }
    
    with open(os.path.join(output_dir, 'feature_info.json'), 'w') as f:
        json.dump(feature_info, f, indent=2)
    
    # Save scaler
    joblib.dump(scaler, os.path.join(output_dir, 'feature_scaler.pkl'))
    
    print(f"\nPreprocessing complete!")
    print(f"Processed data saved to: {output_dir}")
    print(f"Features: {len(feature_cols)}")
    print(f"Samples: {len(processed_data)}")
    print(f"Target range: {y.min():.4f} to {y.max():.4f} kWh")
    
    return processed_data, feature_cols, scaler

if __name__ == "__main__":
    preprocess_data()

