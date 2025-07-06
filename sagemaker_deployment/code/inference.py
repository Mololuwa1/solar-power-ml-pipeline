#!/usr/bin/env python3
"""
SageMaker Inference Script for Solar Power Generation Models
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from io import StringIO

def model_fn(model_dir):
    """Load model from the model directory"""
    print(f"Loading model from: {model_dir}")
    
    # Load metadata
    metadata_path = os.path.join(model_dir, 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Load model
    model_path = os.path.join(model_dir, 'model.pkl')
    model = joblib.load(model_path)
    
    # Load scaler if exists
    scaler = None
    if metadata.get('has_scaler', False):
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        scaler = joblib.load(scaler_path)
    
    return {
        'model': model,
        'scaler': scaler,
        'metadata': metadata
    }

def input_fn(request_body, request_content_type):
    """Parse input data for inference"""
    print(f"Content type: {request_content_type}")
    
    if request_content_type == 'text/csv':
        # Parse CSV data
        data = pd.read_csv(StringIO(request_body))
        return data
    
    elif request_content_type == 'application/json':
        # Parse JSON data
        data = json.loads(request_body)
        if isinstance(data, dict):
            # Single prediction
            data = pd.DataFrame([data])
        else:
            # Multiple predictions
            data = pd.DataFrame(data)
        return data
    
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model_dict):
    """Make predictions using the loaded model"""
    model = model_dict['model']
    scaler = model_dict['scaler']
    metadata = model_dict['metadata']
    
    print(f"Input data shape: {input_data.shape}")
    
    # Ensure we have the right features
    feature_cols = metadata['feature_columns']
    
    # Check if all required features are present
    missing_features = set(feature_cols) - set(input_data.columns)
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    # Select and order features
    X = input_data[feature_cols].copy()
    
    # Handle missing values (simple forward fill)
    X = X.fillna(method='ffill').fillna(0)
    
    # Scale features if scaler exists
    if scaler is not None:
        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)
    else:
        predictions = model.predict(X)
    
    print(f"Generated {len(predictions)} predictions")
    return predictions

def output_fn(predictions, accept):
    """Format the predictions for output"""
    print(f"Accept type: {accept}")
    
    if accept == 'application/json':
        # Return JSON format
        output = {
            'predictions': predictions.tolist(),
            'model_type': 'solar_power_generation',
            'target': 'generation(kWh)'
        }
        return json.dumps(output)
    
    elif accept == 'text/csv':
        # Return CSV format
        df = pd.DataFrame({'predictions': predictions})
        return df.to_csv(index=False)
    
    else:
        # Default to JSON
        output = {
            'predictions': predictions.tolist(),
            'model_type': 'solar_power_generation',
            'target': 'generation(kWh)'
        }
        return json.dumps(output)
