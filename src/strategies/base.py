"""
Base Strategy Class

Abstract base class for all trading strategies.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies
    
    All strategies must implement:
    - generate_signals(): Create buy/sell signals
    - calculate_metrics(): Compute performance metrics
    """
    
    def __init__(self):
        self.name = self.__class__.__name__
    
    @abstractmethod
    def generate_signals(self, *args, **kwargs) -> pd.DataFrame:
        """
        Generate trading signals
        
        Returns:
            DataFrame with signals and related data
        """
        pass
    
    @abstractmethod
    def calculate_metrics(self, results: pd.DataFrame) -> dict:
        """
        Calculate strategy performance metrics
        
        Args:
            results: DataFrame with signals and PnL
            
        Returns:
            Dictionary with performance metrics
        """
        pass
