"""
Solar Power Generation Model Training Script
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Set random seed
np.random.seed(42)

# Directories
DATA_DIR = '/home/ubuntu/processed_data/'
MODEL_DIR = '/home/ubuntu/models/'
os.makedirs(MODEL_DIR, exist_ok=True)

def load_data():
    """Load processed data"""
    print("Loading processed data...")
    data = pd.read_csv(os.path.join(DATA_DIR, 'processed_solar_data.csv'))
    data['Time'] = pd.to_datetime(data['Time'])
    
    # Load feature info
    with open(os.path.join(DATA_DIR, 'feature_info.json'), 'r') as f:
        feature_info = json.load(f)
    
    feature_cols = feature_info['feature_columns']
    target_col = feature_info['target_column']
    
    print(f"Data shape: {data.shape}")
    print(f"Features: {len(feature_cols)}")
    print(f"Target: {target_col}")
    
    return data, feature_cols, target_col

def prepare_data(data, feature_cols, target_col):
    """Prepare data for training"""
    # Prepare features and target
    X = data[feature_cols].copy()
    y = data[target_col].copy()
    
    # Remove any rows with missing values
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)
    data_clean = data[mask].reset_index(drop=True)
    
    # Time-based split
    split_date = data_clean['Time'].quantile(0.8)
    train_mask = data_clean['Time'] <= split_date
    test_mask = data_clean['Time'] > split_date
    
    X_train = X[train_mask].reset_index(drop=True)
    X_test = X[test_mask].reset_index(drop=True)
    y_train = y[train_mask].reset_index(drop=True)
    y_test = y[test_mask].reset_index(drop=True)
    
    # Further split training data for validation
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {X_train_split.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    return X_train_split, X_val, X_test, y_train_split, y_val, y_test, X_train, y_train

def train_xgboost(X_train, y_train, X_val, y_val, X_test, y_test):
    """Train XGBoost model"""
    print("Training XGBoost Regressor...")
    
    xgb_params = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1
    }
    
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    # Metrics
    metrics = {
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'train_r2': r2_score(y_train, y_train_pred),
        'val_rmse': np.sqrt(mean_squared_error(y_val, y_val_pred)),
        'val_mae': mean_absolute_error(y_val, y_val_pred),
        'val_r2': r2_score(y_val, y_val_pred),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'test_mae': mean_absolute_error(y_test, y_test_pred),
        'test_r2': r2_score(y_test, y_test_pred)
    }
    
    print("XGBoost Results:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Save model
    joblib.dump(model, os.path.join(MODEL_DIR, 'xgboost_model.pkl'))
    
    return model, metrics, y_test_pred

def train_random_forest(X_train, y_train, X_val, y_val, X_test, y_test):
    """Train Random Forest model"""
    print("Training Random Forest Regressor...")
    
    rf_params = {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42,
        'n_jobs': -1
    }
    
    model = RandomForestRegressor(**rf_params)
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    # Metrics
    metrics = {
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'train_r2': r2_score(y_train, y_train_pred),
        'val_rmse': np.sqrt(mean_squared_error(y_val, y_val_pred)),
        'val_mae': mean_absolute_error(y_val, y_val_pred),
        'val_r2': r2_score(y_val, y_val_pred),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'test_mae': mean_absolute_error(y_test, y_test_pred),
        'test_r2': r2_score(y_test, y_test_pred)
    }
    
    print("Random Forest Results:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Save model
    joblib.dump(model, os.path.join(MODEL_DIR, 'random_forest_model.pkl'))
    
    return model, metrics, y_test_pred

def train_neural_network(X_train, y_train, X_val, y_val, X_test, y_test):
    """Train Neural Network model"""
    print("Training Neural Network (MLP Regressor)...")
    
    # Scale features for neural network
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    nn_params = {
        'hidden_layer_sizes': (100, 50, 25),
        'activation': 'relu',
        'solver': 'adam',
        'alpha': 0.001,
        'learning_rate': 'adaptive',
        'max_iter': 300,
        'random_state': 42
    }
    
    model = MLPRegressor(**nn_params)
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_val_pred = model.predict(X_val_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Metrics
    metrics = {
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'train_r2': r2_score(y_train, y_train_pred),
        'val_rmse': np.sqrt(mean_squared_error(y_val, y_val_pred)),
        'val_mae': mean_absolute_error(y_val, y_val_pred),
        'val_r2': r2_score(y_val, y_val_pred),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'test_mae': mean_absolute_error(y_test, y_test_pred),
        'test_r2': r2_score(y_test, y_test_pred)
    }
    
    print("Neural Network Results:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Save model and scaler
    joblib.dump(model, os.path.join(MODEL_DIR, 'neural_network_model.pkl'))
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'nn_scaler.pkl'))
    
    return model, metrics, y_test_pred, scaler

def create_comparison(xgb_metrics, rf_metrics, nn_metrics):
    """Create model comparison"""
    comparison_data = {
        'Model': ['XGBoost', 'Random Forest', 'Neural Network'],
        'Train_RMSE': [xgb_metrics['train_rmse'], rf_metrics['train_rmse'], nn_metrics['train_rmse']],
        'Val_RMSE': [xgb_metrics['val_rmse'], rf_metrics['val_rmse'], nn_metrics['val_rmse']],
        'Test_RMSE': [xgb_metrics['test_rmse'], rf_metrics['test_rmse'], nn_metrics['test_rmse']],
        'Train_MAE': [xgb_metrics['train_mae'], rf_metrics['train_mae'], nn_metrics['train_mae']],
        'Val_MAE': [xgb_metrics['val_mae'], rf_metrics['val_mae'], nn_metrics['val_mae']],
        'Test_MAE': [xgb_metrics['test_mae'], rf_metrics['test_mae'], nn_metrics['test_mae']],
        'Train_R2': [xgb_metrics['train_r2'], rf_metrics['train_r2'], nn_metrics['train_r2']],
        'Val_R2': [xgb_metrics['val_r2'], rf_metrics['val_r2'], nn_metrics['val_r2']],
        'Test_R2': [xgb_metrics['test_r2'], rf_metrics['test_r2'], nn_metrics['test_r2']]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    print("\\n=== MODEL COMPARISON ===")
    print(comparison_df.round(4))
    
    # Save comparison
    comparison_df.to_csv(os.path.join(MODEL_DIR, 'model_comparison.csv'), index=False)
    
    # Find best model
    best_model_idx = comparison_df['Val_RMSE'].idxmin()
    best_model_name = comparison_df.loc[best_model_idx, 'Model']
    
    print(f"\\nBest model based on validation RMSE: {best_model_name}")
    print(f"Validation RMSE: {comparison_df.loc[best_model_idx, 'Val_RMSE']:.4f}")
    print(f"Test RMSE: {comparison_df.loc[best_model_idx, 'Test_RMSE']:.4f}")
    
    return comparison_df, best_model_name, best_model_idx

def save_metadata(xgb_metrics, rf_metrics, nn_metrics, best_model_name, best_model_idx, comparison_df, feature_cols, target_col):
    """Save model metadata"""
    model_metadata = {
        'timestamp': datetime.now().isoformat(),
        'models': {
            'xgboost': {
                'metrics': xgb_metrics,
                'model_file': 'xgboost_model.pkl'
            },
            'random_forest': {
                'metrics': rf_metrics,
                'model_file': 'random_forest_model.pkl'
            },
            'neural_network': {
                'metrics': nn_metrics,
                'model_file': 'neural_network_model.pkl',
                'scaler_file': 'nn_scaler.pkl'
            }
        },
        'best_model': {
            'name': best_model_name,
            'val_rmse': float(comparison_df.loc[best_model_idx, 'Val_RMSE']),
            'test_rmse': float(comparison_df.loc[best_model_idx, 'Test_RMSE'])
        },
        'feature_columns': feature_cols,
        'target_column': target_col
    }
    
    with open(os.path.join(MODEL_DIR, 'model_metadata.json'), 'w') as f:
        json.dump(model_metadata, f, indent=2)

def main():
    """Main training function"""
    # Load data
    data, feature_cols, target_col = load_data()
    
    # Prepare data
    X_train_split, X_val, X_test, y_train_split, y_val, y_test, X_train, y_train = prepare_data(data, feature_cols, target_col)
    
    # Train models
    xgb_model, xgb_metrics, xgb_pred = train_xgboost(X_train_split, y_train_split, X_val, y_val, X_test, y_test)
    rf_model, rf_metrics, rf_pred = train_random_forest(X_train_split, y_train_split, X_val, y_val, X_test, y_test)
    nn_model, nn_metrics, nn_pred, nn_scaler = train_neural_network(X_train_split, y_train_split, X_val, y_val, X_test, y_test)
    
    # Create comparison
    comparison_df, best_model_name, best_model_idx = create_comparison(xgb_metrics, rf_metrics, nn_metrics)
    
    # Save metadata
    save_metadata(xgb_metrics, rf_metrics, nn_metrics, best_model_name, best_model_idx, comparison_df, feature_cols, target_col)
    
    print("\\n=== MODEL TRAINING COMPLETE ===")
    print(f"Models saved to: {MODEL_DIR}")
    print(f"Best model: {best_model_name}")
    
    return xgb_model, rf_model, nn_model, comparison_df

if __name__ == "__main__":
    main()

