#!/usr/bin/env python3
"""
Future forecasting example for Bracon Ash solar farm
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))

import pandas as pd
from datetime import datetime, timedelta
from bracon_ash_predictor import BraconAshPredictor

def main():
    print("🔮 BRACON ASH FUTURE FORECASTING EXAMPLE")
    print("=" * 50)
    
    # Initialize predictor
    predictor = BraconAshPredictor(model_dir="../models")
    
    # Generate 24-hour forecast
    start_time = datetime.now().replace(minute=0, second=0, microsecond=0)
    print(f"Generating 24-hour forecast starting from: {start_time}")
    
    forecast = predictor.predict_future(
        start_time=start_time,
        hours=24
    )
    
    # Analyze forecast
    total_generation = forecast['predicted_generation_kwh'].sum()
    peak_generation = forecast['predicted_generation_kwh'].max()
    peak_time = forecast.loc[forecast['predicted_generation_kwh'].idxmax(), 'timestamp']
    
    print(f"\n📊 24-HOUR FORECAST SUMMARY:")
    print(f"   Total predicted generation: {total_generation:.2f} kWh")
    print(f"   Peak generation: {peak_generation:.2f} kWh")
    print(f"   Peak time: {peak_time}")
    print(f"   Average hourly generation: {total_generation/24:.2f} kWh")
    
    # Show hourly summary
    forecast['hour'] = forecast['timestamp'].dt.hour
    hourly_summary = forecast.groupby('hour')['predicted_generation_kwh'].sum()
    
    print(f"\n📋 HOURLY GENERATION FORECAST:")
    for hour, generation in hourly_summary.items():
        print(f"   {hour:02d}:00 - {generation:.2f} kWh")
    
    # Save forecast
    forecast.to_csv("24hour_forecast.csv", index=False)
    print(f"\n✅ Forecast saved to 24hour_forecast.csv")

if __name__ == "__main__":
    main()
