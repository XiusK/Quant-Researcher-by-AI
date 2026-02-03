"""
Quick Test - Load Kaggle XAUUSD Data

Test loading data from local Kaggle files.
"""

import sys
sys.path.insert(0, 'e:/Python Project/Quant Researcher By AI')

from src.data import load_xauusd_from_kaggle, list_available_timeframes, calculate_features

print("="*60)
print("Testing Kaggle Data Loader")
print("="*60)

# List available timeframes
print("\n[1] Available Timeframes:")
available = list_available_timeframes()
for tf in available:
    print(f"  - {tf}")

# Load daily data
print("\n[2] Loading 1d (Daily) data:")
data = load_xauusd_from_kaggle(timeframe="1d")

print(f"\nData shape: {data.shape}")
print(f"\nFirst 5 rows:")
print(data.head())

print(f"\nLast 5 rows:")
print(data.tail())

print(f"\nData info:")
print(data.info())

# Calculate features
print("\n[3] Calculating technical features...")
data_with_features = calculate_features(data)

print(f"Features added: {len(data_with_features.columns) - len(data.columns)}")
print(f"\nAvailable features:")
for col in sorted(data_with_features.columns):
    print(f"  - {col}")

print("\n" + "="*60)
print("Test completed successfully!")
print("="*60)
