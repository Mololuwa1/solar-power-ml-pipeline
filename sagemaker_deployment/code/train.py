#!/usr/bin/env python3
"""
SageMaker Training Script for Solar Power Generation Models
"""

import argparse
import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import xgboost as xgb

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    
    # SageMaker specific arguments
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
    
    # Model hyperparameters
    parser.add_argument('--model-type', type=str, default='neural_network', 
                       choices=['xgboost', 'random_forest', 'neural_network'])
    parser.add_argument('--n-estimators', type=int, default=100)
    parser.add_argument('--max-depth', type=int, default=6)
    parser.add_argument('--learning-rate', type=float, default=0.1)
    parser.add_argument('--hidden-layers', type=str, default='100,50,25')
    parser.add_argument('--alpha', type=float, default=0.001)
    
    return parser.parse_args()

def load_data(train_path, validation_path=None):
    """Load training and validation data"""
    print(f"Loading training data from: {train_path}")
    
    # Load training data
    train_files = [f for f in os.listdir(train_path) if f.endswith('.csv')]
    if not train_files:
        raise ValueError(f"No CSV files found in {train_path}")
    
    train_data = pd.read_csv(os.path.join(train_path, train_files[0]))
    print(f"Training data shape: {train_data.shape}")
    
    # Load validation data if provided
    val_data = None
    if validation_path and os.path.exists(validation_path):
        val_files = [f for f in os.listdir(validation_path) if f.endswith('.csv')]
        if val_files:
            val_data = pd.read_csv(os.path.join(validation_path, val_files[0]))
            print(f"Validation data shape: {val_data.shape}")
    
    return train_data, val_data

def prepare_features(data):
    """Prepare features and target from data"""
    # Identify feature columns (exclude metadata and target)
    exclude_cols = ['Time', 'generation(kWh)', 'power(W)', 'station']
    feature_cols = [col for col in data.columns if col not in exclude_cols]
    
    X = data[feature_cols].copy()
    y = data['generation(kWh)'].copy()
    
    # Remove rows with missing values
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)
    
    return X, y, feature_cols

def train_model(model_type, X_train, y_train, X_val, y_val, args):
    """Train the specified model"""
    print(f"Training {model_type} model...")
    
    if model_type == 'xgboost':
        model = xgb.XGBRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            random_state=42,
            n_jobs=-1
        )
        scaler = None
        
    elif model_type == 'random_forest':
        model = RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=42,
            n_jobs=-1
        )
        scaler = None
        
    elif model_type == 'neural_network':
        # Parse hidden layers
        hidden_layers = tuple(map(int, args.hidden_layers.split(',')))
        
        model = MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            alpha=args.alpha,
            random_state=42,
            max_iter=300
        )
        
        # Scale features for neural network
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val) if X_val is not None else None
        
        model.fit(X_train_scaled, y_train)
        
        # Calculate metrics
        train_pred = model.predict(X_train_scaled)
        val_pred = model.predict(X_val_scaled) if X_val is not None else None
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Train model (for non-neural network models)
    if model_type != 'neural_network':
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val) if X_val is not None else None
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    train_r2 = r2_score(y_train, train_pred)
    
    val_rmse = None
    val_r2 = None
    if X_val is not None and val_pred is not None:
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        val_r2 = r2_score(y_val, val_pred)
    
    print(f"Training RMSE: {train_rmse:.4f}, R²: {train_r2:.4f}")
    if val_rmse is not None:
        print(f"Validation RMSE: {val_rmse:.4f}, R²: {val_r2:.4f}")
    
    return model, scaler, {
        'train_rmse': train_rmse,
        'train_r2': train_r2,
        'val_rmse': val_rmse,
        'val_r2': val_r2
    }

def save_model(model, scaler, feature_cols, metrics, model_dir, model_type):
    """Save model and metadata"""
    print(f"Saving model to: {model_dir}")
    
    # Save model
    model_path = os.path.join(model_dir, 'model.pkl')
    joblib.dump(model, model_path)
    
    # Save scaler if exists
    if scaler is not None:
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        joblib.dump(scaler, scaler_path)
    
    # Save metadata
    metadata = {
        'model_type': model_type,
        'feature_columns': feature_cols,
        'target_column': 'generation(kWh)',
        'has_scaler': scaler is not None,
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }
    
    metadata_path = os.path.join(model_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("Model saved successfully!")

def main():
    """Main training function"""
    args = parse_args()
    
    print("Starting SageMaker training...")
    print(f"Model type: {args.model_type}")
    print(f"Model directory: {args.model_dir}")
    
    # Load data
    train_data, val_data = load_data(args.train, args.validation)
    
    # Prepare features
    X_train, y_train, feature_cols = prepare_features(train_data)
    
    X_val, y_val = None, None
    if val_data is not None:
        X_val, y_val, _ = prepare_features(val_data)
    else:
        # Split training data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
    
    # Train model
    model, scaler, metrics = train_model(
        args.model_type, X_train, y_train, X_val, y_val, args
    )
    
    # Save model
    save_model(model, scaler, feature_cols, metrics, args.model_dir, args.model_type)
    
    print("Training completed successfully!")

if __name__ == '__main__':
    main()
