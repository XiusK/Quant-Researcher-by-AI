"""
Download Forex (EUR/USD) Historical Data from Yahoo Finance

Downloads daily EUR/USD exchange rate data for position sizing analysis.
"""

import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime

def download_forex_data(
    ticker: str = 'EURUSD=X',
    start_date: str = '2004-01-01',
    end_date: str = None,
    output_dir: str = None
) -> pd.DataFrame:
    """
    Download Forex data from Yahoo Finance.
    
    Parameters:
    -----------
    ticker : str
        Yahoo Finance ticker (EURUSD=X, GBPUSD=X, USDJPY=X)
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str
        End date in YYYY-MM-DD format (None = today)
    output_dir : str
        Output directory path (None = use project default)
        
    Returns:
    --------
    pd.DataFrame : Downloaded Forex data
    """
    
    # Set default output directory if not specified
    if output_dir is None:
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        output_dir = project_root / 'data' / 'raw' / 'forex_eurusd'
        output_dir = str(output_dir)
    
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    print(f"Downloading {ticker} data from {start_date} to {end_date}...")
    
    # Download data
    data = yf.download(ticker, start=start_date, end=end_date, progress=True)
    
    if data.empty:
        raise ValueError(f"No data downloaded for {ticker}")
    
    # Handle MultiIndex columns from yfinance
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]
    
    # Reset index to make Date a column
    data = data.reset_index()
    
    # Rename columns to match our format
    column_mapping = {
        'Date': 'time',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume',
        'Adj Close': 'adj_close'
    }
    
    data = data.rename(columns=column_mapping)
    
    # Select only needed columns
    columns_to_keep = ['time', 'open', 'high', 'low', 'close', 'volume']
    data = data[columns_to_keep]
    
    print(f"\n✅ Downloaded {len(data)} bars")
    print(f"Period: {data['time'].iloc[0]} to {data['time'].iloc[-1]}")
    
    # Display first few rows
    print(f"\nFirst 5 rows:")
    print(data.head())
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    output_file = output_path / 'EURUSD_Daily.csv'
    data.to_csv(output_file, index=False)
    
    print(f"\n✅ Data saved to: {output_file}")
    
    return data


if __name__ == '__main__':
    # Download EUR/USD
    print("="*60)
    print("FOREX DATA DOWNLOAD: EUR/USD")
    print("="*60)
    
    # Get absolute path relative to script location
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    output_dir = project_root / 'data' / 'raw' / 'forex_eurusd'
    
    eurusd_data = download_forex_data(
        ticker='EURUSD=X',
        start_date='2004-01-01',
        output_dir=str(output_dir)
    )
    
    print("\n" + "="*60)
    print("Download Summary:")
    print("="*60)
    print(f"Total bars: {len(eurusd_data)}")
    print(f"Date range: {eurusd_data['time'].iloc[0]} to {eurusd_data['time'].iloc[-1]}")
    print(f"EUR/USD range: {eurusd_data['close'].min():.4f} to {eurusd_data['close'].max():.4f}")
    
    # Optional: Download other major pairs
    print("\n" + "="*60)
    print("Would you like to download other Forex pairs?")
    print("Available: GBP/USD, USD/JPY, AUD/USD, USD/CHF")
    print("Run manually with:")
    print("  download_forex_data('GBPUSD=X', output_dir='../data/raw/forex_gbpusd')")
    print("="*60)
