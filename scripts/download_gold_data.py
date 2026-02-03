"""
Download XAUUSD (Gold) Historical Data

Uses yfinance to download Gold Futures (GC=F) data
"""

import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime


def download_gold_data(start_date="2004-01-01", end_date=None):
    """Download Gold futures data from Yahoo Finance"""
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    print(f"Downloading Gold data from {start_date} to {end_date}...")
    
    # GC=F is Gold Futures (most liquid)
    ticker = "GC=F"
    
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=True)
        
        if data.empty:
            raise ValueError("No data downloaded")
        
        # Handle MultiIndex columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]
        
        # Reset index
        data = data.reset_index()
        
        # Standardize column names
        column_mapping = {
            'Date': 'time',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }
        data = data.rename(columns=column_mapping)
        
        # Convert time
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
        
        return data
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


if __name__ == "__main__":
    # Download Gold data
    gold_data = download_gold_data(start_date="2004-01-01")
    
    if gold_data is not None:
        # Save to CSV
        output_path = Path("data/raw/gold_xauusd/XAUUSD_Daily.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        gold_data.to_csv(output_path, index=False)
        
        print(f"\n✅ Data saved to: {output_path}")
        print("\n" + "="*60)
        print("✅ Gold data download complete!")
        print("="*60)
