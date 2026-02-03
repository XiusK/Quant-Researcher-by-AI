"""
Download XAGUSD (Silver) Historical Data

Data Sources:
1. Yahoo Finance: SI=F (Silver Futures)
2. Alternative: Download from investing.com or MetaTrader 5

Output: data/raw/silver_xagusd/XAGUSD_Daily.csv
"""

import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime

def download_silver_data(start_date="2004-01-01", end_date=None):
    """
    Download Silver futures data from Yahoo Finance
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD), defaults to today
    
    Returns:
        DataFrame with OHLCV data
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    print(f"Downloading Silver data from {start_date} to {end_date}...")
    
    # Download from Yahoo Finance
    # SI=F is Silver Futures (most liquid contract)
    ticker = "SI=F"
    
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=True)
        
        if data.empty:
            raise ValueError("No data downloaded")
        
        # Handle MultiIndex columns (yfinance format)
        if isinstance(data.columns, pd.MultiIndex):
            # Extract first level (Price type: Open, High, Low, Close, Volume)
            data.columns = [col[0] for col in data.columns]
        
        # Reset index to get date as column
        data = data.reset_index()
        
        # Standardize column names to lowercase
        column_mapping = {
            'Date': 'time',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }
        data = data.rename(columns=column_mapping)
        
        # Convert time to proper format
        data['time'] = pd.to_datetime(data['time'])
        
        # Keep only needed columns
        data = data[['time', 'open', 'high', 'low', 'close', 'volume']]
        
        # Sort by time
        data = data.sort_values('time').reset_index(drop=True)
        
        print(f"\n✅ Downloaded {len(data)} bars")
        print(f"Period: {data['time'].iloc[0]} to {data['time'].iloc[-1]}")
        print(f"\nFirst 5 rows:")
        print(data.head())
        print(f"\nLast 5 rows:")
        print(data.tail())
        
        # Check for missing data
        missing = data.isnull().sum()
        if missing.any():
            print(f"\n⚠️  Missing data detected:")
            print(missing[missing > 0])
        
        return data
        
    except Exception as e:
        print(f"❌ Error downloading from Yahoo Finance: {e}")
        print("\nAlternative options:")
        print("1. Download from MetaTrader 5 (Tools > History Center)")
        print("2. Download from Investing.com: https://www.investing.com/commodities/silver-historical-data")
        print("3. Use Kaggle datasets")
        return None


def save_data(data, output_path):
    """Save data to CSV file"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data.to_csv(output_path, index=False)
    print(f"\n✅ Data saved to: {output_path}")


if __name__ == "__main__":
    # Download Silver data
    silver_data = download_silver_data(start_date="2004-01-01")
    
    if silver_data is not None:
        # Save to CSV
        output_path = "data/raw/silver_xagusd/XAGUSD_Daily.csv"
        save_data(silver_data, output_path)
        
        print("\n" + "="*60)
        print("✅ Silver data download complete!")
        print("="*60)
        print(f"\nYou can now run the Gold-Silver Ratio notebook:")
        print("jupyter notebook notebooks/03_gold_silver_ratio_research.ipynb")
    else:
        print("\n" + "="*60)
        print("❌ Download failed. Please try manual download.")
        print("="*60)
