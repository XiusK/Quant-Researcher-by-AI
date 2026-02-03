"""
Data Module

Functions for downloading, preprocessing, and validating market data.
"""

from src.data.data_loader import (
    load_xauusd_from_kaggle,
    download_xauusd,
    calculate_features,
    validate_data_quality,
    split_train_test,
    list_available_timeframes
)

__all__ = [
    'load_xauusd_from_kaggle',
    'download_xauusd',
    'calculate_features',
    'validate_data_quality',
    'split_train_test',
    'list_available_timeframes'
]
