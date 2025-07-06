#!/usr/bin/env python3
"""
Test SageMaker Endpoint
"""

import boto3
import json
import pandas as pd
import numpy as np

def test_endpoint(endpoint_name, test_data):
    """Test the deployed endpoint"""
    runtime_client = boto3.client('sagemaker-runtime')
    
    # Prepare test data
    if isinstance(test_data, pd.DataFrame):
        payload = test_data.to_csv(index=False)
        content_type = 'text/csv'
    else:
        payload = json.dumps(test_data)
        content_type = 'application/json'
    
    print(f"Testing endpoint: {endpoint_name}")
    print(f"Payload size: {len(payload)} bytes")
    
    # Invoke endpoint
    response = runtime_client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType=content_type,
        Body=payload
    )
    
    # Parse response
    result = json.loads(response['Body'].read().decode())
    
    print("Prediction successful!")
    print(f"Predictions: {result['predictions'][:5]}...")  # Show first 5
    
    return result

def create_sample_data():
    """Create sample data for testing"""
    # This should match your feature structure
    sample_data = {
        'hour': [12, 13, 14],
        'month': [6, 6, 6],
        'hour_sin': [0.0, 0.259, 0.5],
        'hour_cos': [1.0, 0.966, 0.866],
        'is_daytime': [1, 1, 1],
        'is_peak_sun': [1, 1, 1],
        'Irradiance_mean': [800.0, 850.0, 900.0],
        'Temperature_mean': [25.0, 26.0, 27.0],
        'generation(kWh)_lag_1': [2.5, 2.8, 3.0]
        # Add more features as needed
    }
    
    return pd.DataFrame(sample_data)

def main():
    """Main test function"""
    endpoint_name = 'solar-power-generation-endpoint'
    
    # Create sample data
    test_data = create_sample_data()
    
    try:
        # Test endpoint
        result = test_endpoint(endpoint_name, test_data)
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"Test failed: {str(e)}")

if __name__ == '__main__':
    main()
