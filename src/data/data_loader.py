"""
Data Loader Module

Functions for downloading and preprocessing market data with validation.
"""

from typing import Optional, Tuple, List
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os
from pathlib import Path


def load_xauusd_from_kaggle(
    timeframe: str = "1d",
    data_folder: str = "Kaggles_XAUUSD_Data"
) -> pd.DataFrame:
    """
    Load XAUUSD (Gold/USD) data from local Kaggle CSV files.
    
    Args:
        timeframe: Data frequency ('1d', '1h', '15m', '5m', etc.)
        data_folder: Folder name containing CSV files
        
    Returns:
        DataFrame with columns: date, open, high, low, close, volume
        
    Raises:
        ValueError: If file not found or data format invalid
    """
    # Map timeframe to filename
    timeframe_map = {
        '1d': 'XAU_1d_data (1).csv',
        '1h': 'XAU_1h_data (1).csv',
        '4h': 'XAU_4h_data (1).csv',
        '30m': 'XAU_30m_data (1).csv',
        '15m': 'XAU_15m_data (2).csv',
        '5m': 'XAU_5m_data.csv',
        '1m': 'XAU_1m_data.csv',
        '1w': 'XAU_1w_data (1).csv',
        '1M': 'XAU_1Month_data (1).csv'
    }
    
    if timeframe not in timeframe_map:
        raise ValueError(
            f"Unsupported timeframe: {timeframe}. "
            f"Available: {list(timeframe_map.keys())}"
        )
    
    # Construct file path
    filename = timeframe_map[timeframe]
    filepath = Path(data_folder) / filename
    
    if not filepath.exists():
        raise ValueError(f"Data file not found: {filepath}")
    
    print(f"Loading XAUUSD {timeframe} data from {filename}...")
    
    try:
        # Read CSV (semicolon delimiter, date format)
        df = pd.read_csv(
            filepath,
            sep=';',
            parse_dates=['Date'],
            dayfirst=False
        )
        
        # Standardize column names
        df.columns = [col.lower() for col in df.columns]
        
        # Set date as index
        df.set_index('date', inplace=True)
        df.index = pd.to_datetime(df.index, format='%Y.%m.%d %H:%M')
        
        # Sort by date
        df = df.sort_index()
        
        # Remove rows with missing data
        df = df.dropna()
        
        if len(df) < 100:
            raise ValueError(f"Insufficient data: only {len(df)} rows")
        
        print(f"Successfully loaded {len(df)} rows")
        print(f"Date range: {df.index[0]} to {df.index[-1]}")
        print(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        
        return df
        
    except Exception as e:
        raise ValueError(f"Failed to load data: {str(e)}")


def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical features from OHLC data.
    
    Features include:
    - Returns (log and simple)
    - Volatility (realized, Parkinson)
    - Range metrics (high-low, true range)
    - Moving averages
    - RSI
    - Bollinger Bands
    
    Args:
        df: DataFrame with OHLC data
        
    Returns:
        DataFrame with original data plus features
    """
    df = df.copy()
    
    # Returns
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # Volatility
    df['realized_vol'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
    
    # Parkinson volatility (high-low range estimator)
    df['parkinson_vol'] = np.sqrt(
        (1 / (4 * np.log(2))) * 
        np.log(df['high'] / df['low']) ** 2
    ).rolling(window=20).mean() * np.sqrt(252)
    
    # Range metrics
    df['hl_range'] = (df['high'] - df['low']) / df['close']
    df['oc_range'] = abs(df['open'] - df['close']) / df['close']
    
    # True Range (for ATR)
    df['prev_close'] = df['close'].shift(1)
    df['tr'] = df[['high', 'prev_close']].max(axis=1) - df[['low', 'prev_close']].min(axis=1)
    df['atr_14'] = df['tr'].rolling(window=14).mean()
    
    # Moving Averages
    for period in [20, 50, 200]:
        df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    bb_period = 20
    bb_std = 2
    df['bb_middle'] = df['close'].rolling(window=bb_period).mean()
    bb_std_dev = df['close'].rolling(window=bb_period).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * bb_std_dev)
    df['bb_lower'] = df['bb_middle'] - (bb_std * bb_std_dev)
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # Price momentum
    for period in [5, 10, 20]:
        df[f'momentum_{period}'] = df['close'].pct_change(periods=period)
    
    # Volume metrics (if available)
    if 'volume' in df.columns and df['volume'].sum() > 0:
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
    
    # Clean up temporary columns
    df = df.drop(columns=['prev_close', 'tr'], errors='ignore')
    
    return df


def validate_data_quality(df: pd.DataFrame) -> dict:
    """
    Check data quality and return diagnostic report.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Dict with validation results
    """
    report = {
        'total_rows': len(df),
        'date_range': (df.index[0], df.index[-1]),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_dates': df.index.duplicated().sum(),
        'price_anomalies': {}
    }
    
    # Check for price anomalies
    if 'close' in df.columns:
        returns = df['close'].pct_change()
        
        report['price_anomalies'] = {
            'extreme_returns': (abs(returns) > 0.10).sum(),  # >10% daily move
            'zero_prices': (df['close'] <= 0).sum(),
            'negative_prices': (df['close'] < 0).sum()
        }
    
    # Check OHLC consistency
    if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
        report['ohlc_violations'] = {
            'high_lt_low': (df['high'] < df['low']).sum(),
            'close_gt_high': (df['close'] > df['high']).sum(),
            'close_lt_low': (df['close'] < df['low']).sum(),
            'open_gt_high': (df['open'] > df['high']).sum(),
            'open_lt_low': (df['open'] < df['low']).sum()
        }
    
    return report


def split_train_test(
    df: pd.DataFrame,
    train_ratio: float = 0.7
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train and test sets (temporal split).
    
    Args:
        df: Full dataset
        train_ratio: Fraction of data for training
        
    Returns:
        (train_df, test_df)
    """
    split_idx = int(len(df) * train_ratio)
    
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    
    print(f"Train: {train.index[0]} to {train.index[-1]} ({len(train)} rows)")
    print(f"Test:  {test.index[0]} to {test.index[-1]} ({len(test)} rows)")
    
    return train, test


def list_available_timeframes(data_folder: str = "Kaggles_XAUUSD_Data") -> List[str]:
    """List all available timeframes in the data folder."""
    timeframe_map = {
        '1d': 'XAU_1d_data (1).csv',
        '1h': 'XAU_1h_data (1).csv',
        '4h': 'XAU_4h_data (1).csv',
        '30m': 'XAU_30m_data (1).csv',
        '15m': 'XAU_15m_data (2).csv',
        '5m': 'XAU_5m_data.csv',
        '1m': 'XAU_1m_data.csv',
        '1w': 'XAU_1w_data (1).csv',
        '1M': 'XAU_1Month_data (1).csv'
    }
    
    available = []
    for tf, filename in timeframe_map.items():
        filepath = Path(data_folder) / filename
        if filepath.exists():
            available.append(tf)
    
    return available


def download_xauusd(
    start_date: str = "2018-01-01",
    end_date: Optional[str] = None,
    interval: str = "1d"
) -> pd.DataFrame:
    """Deprecated: Use load_xauusd_from_kaggle() for local data."""
    print("WARNING: download_xauusd() is deprecated. Using load_xauusd_from_kaggle() instead...")
    return load_xauusd_from_kaggle(timeframe=interval)
