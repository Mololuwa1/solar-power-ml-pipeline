#!/usr/bin/env python3
"""
Basic prediction example for Bracon Ash solar farm
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))

import pandas as pd
from bracon_ash_predictor import BraconAshPredictor

def main():
    print("🔮 BRACON ASH SOLAR PREDICTION EXAMPLE")
    print("=" * 50)
    
    # Initialize predictor
    print("Loading model...")
    predictor = BraconAshPredictor(model_dir="../models")
    
    # Load sample data
    print("Loading sample data...")
    data = pd.read_excel("../data/sample_data.xlsx")
    print(f"Loaded {len(data)} data points")
    
    # Make predictions on first 100 points
    print("Making predictions...")
    sample_data = data.head(100)
    results = predictor.predict(sample_data)
    
    # Display results
    print(f"\n📊 PREDICTION RESULTS:")
    print(f"   Samples processed: {len(results)}")
    print(f"   Average prediction: {results['predicted_generation_kwh'].mean():.2f} kWh")
    print(f"   Max prediction: {results['predicted_generation_kwh'].max():.2f} kWh")
    print(f"   Min prediction: {results['predicted_generation_kwh'].min():.2f} kWh")
    
    # Show sample predictions
    print(f"\n📋 SAMPLE PREDICTIONS:")
    print(results[['timestamp', 'predicted_generation_kwh']].head(10).to_string(index=False))
    
    # Save results
    results.to_csv("prediction_results.csv", index=False)
    print(f"\n✅ Results saved to prediction_results.csv")

if __name__ == "__main__":
    main()
