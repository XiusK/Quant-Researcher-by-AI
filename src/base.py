"""
Base classes for stochastic models and trading strategies.

This module provides abstract base classes that enforce mathematical rigor
and standardized interfaces across all quantitative models and strategies.

Author: Quant Researcher AI
Date: 2026-02-03
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum


class AssetClass(Enum):
    """Asset classification for model selection."""
    FX = "fx"
    RATES = "rates"
    EQUITIES = "equities"
    CRYPTO = "crypto"
    COMMODITIES = "commodities"


@dataclass
class ModelParameters:
    """Container for stochastic model parameters with validation."""
    params: Dict[str, float]
    confidence_intervals: Optional[Dict[str, Tuple[float, float]]] = None
    calibration_window: Optional[int] = None
    aic: Optional[float] = None
    bic: Optional[float] = None
    
    def validate(self) -> bool:
        """
        Validate parameter ranges and statistical properties.
        
        Returns:
            bool: True if parameters pass validation checks
        """
        for key, value in self.params.items():
            if not np.isfinite(value):
                raise ValueError(f"Parameter {key} contains non-finite value: {value}")
        return True


class BaseStochasticModel(ABC):
    """
    Abstract base class for all stochastic processes.
    
    All derived models must implement the core mathematical operations:
    - simulate: Generate price paths
    - calibrate: Fit parameters to historical data
    - density: Calculate probability density function
    
    Attributes:
        asset_class: Type of financial instrument
        params: Calibrated model parameters
        name: Human-readable model identifier
    """
    
    def __init__(self, asset_class: AssetClass, name: str):
        """
        Initialize base stochastic model.
        
        Args:
            asset_class: Classification of the underlying asset
            name: Model identifier for logging and tracking
        """
        self.asset_class = asset_class
        self.name = name
        self.params: Optional[ModelParameters] = None
        self.is_calibrated: bool = False
        
    @abstractmethod
    def simulate(
        self,
        S0: float,
        T: float,
        n_steps: int,
        n_paths: int = 1,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate sample paths using the calibrated stochastic process.
        
        Args:
            S0: Initial value (spot price)
            T: Time horizon in years
            n_steps: Number of time steps
            n_paths: Number of simulation paths
            seed: Random seed for reproducibility
            
        Returns:
            np.ndarray: Shape (n_paths, n_steps+1) with simulated paths
            
        Mathematical Foundation:
            Implementation must follow Ito calculus and ensure
            no-arbitrage conditions are preserved.
        """
        pass
    
    @abstractmethod
    def calibrate(
        self,
        data: pd.Series,
        method: str = "mle"
    ) -> ModelParameters:
        """
        Estimate model parameters from historical data.
        
        Args:
            data: Time series of observed prices/returns
            method: Calibration method ('mle', 'mom', 'bayesian')
            
        Returns:
            ModelParameters: Calibrated parameters with diagnostics
            
        Raises:
            ValueError: If data fails preliminary tests (stationarity, etc.)
            
        Side Effects:
            Sets self.params and self.is_calibrated = True
        """
        pass
    
    @abstractmethod
    def log_likelihood(self, data: pd.Series) -> float:
        """
        Compute log-likelihood of observed data under current parameters.
        
        Args:
            data: Historical observations
            
        Returns:
            float: Log-likelihood value (higher is better fit)
        """
        pass
    
    def aic(self, data: pd.Series) -> float:
        """
        Calculate Akaike Information Criterion.
        
        AIC = 2k - 2ln(L)
        where k is number of parameters, L is likelihood
        
        Args:
            data: Historical data
            
        Returns:
            float: AIC value (lower is better)
        """
        if not self.is_calibrated:
            raise RuntimeError("Model must be calibrated before computing AIC")
        
        k = len(self.params.params)
        ll = self.log_likelihood(data)
        return 2 * k - 2 * ll
    
    def bic(self, data: pd.Series) -> float:
        """
        Calculate Bayesian Information Criterion.
        
        BIC = k*ln(n) - 2ln(L)
        where n is sample size
        
        Args:
            data: Historical data
            
        Returns:
            float: BIC value (lower is better)
        """
        if not self.is_calibrated:
            raise RuntimeError("Model must be calibrated before computing BIC")
        
        k = len(self.params.params)
        n = len(data)
        ll = self.log_likelihood(data)
        return k * np.log(n) - 2 * ll
    
    def validate_no_lookahead(self, timestamp: pd.Timestamp, data_available: pd.Timestamp) -> None:
        """
        Guard against look-ahead bias in backtesting.
        
        Args:
            timestamp: Current simulation time
            data_available: Latest available data timestamp
            
        Raises:
            ValueError: If future data is being used
        """
        if timestamp > data_available:
            raise ValueError(
                f"Look-ahead bias detected: Using data from {data_available} "
                f"at timestamp {timestamp}"
            )


@dataclass
class Signal:
    """Trading signal with confidence and metadata."""
    timestamp: pd.Timestamp
    direction: int  # +1 (long), -1 (short), 0 (neutral)
    confidence: float  # [0, 1]
    target_position: float  # Target notional or percentage
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class RiskMetrics:
    """Container for risk-adjusted performance metrics."""
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    var_95: float  # 95% Value at Risk
    cvar_95: float  # 95% Conditional VaR
    calmar_ratio: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary for logging."""
        return {
            'sharpe': self.sharpe_ratio,
            'sortino': self.sortino_ratio,
            'max_dd': self.max_drawdown,
            'var_95': self.var_95,
            'cvar_95': self.cvar_95,
            'calmar': self.calmar_ratio
        }


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.
    
    All strategies must implement signal generation and risk management.
    Enforces separation between alpha generation and position sizing.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize strategy with configuration.
        
        Args:
            name: Strategy identifier
            config: Dictionary containing all strategy parameters
        """
        self.name = name
        self.config = config
        self.position: float = 0.0
        self.pnl_history: list = []
        
    @abstractmethod
    def generate_signal(
        self,
        market_data: pd.DataFrame,
        timestamp: pd.Timestamp
    ) -> Signal:
        """
        Generate trading signal based on current market state.
        
        Args:
            market_data: Historical price data up to timestamp
            timestamp: Current decision point
            
        Returns:
            Signal: Trading signal with direction and confidence
            
        Critical:
            Must not use any data with timestamp > current timestamp.
            Use validate_no_lookahead() to enforce this.
        """
        pass
    
    @abstractmethod
    def calculate_position_size(
        self,
        signal: Signal,
        portfolio_value: float,
        current_volatility: float
    ) -> float:
        """
        Determine position size based on signal and risk parameters.
        
        Args:
            signal: Generated trading signal
            portfolio_value: Current portfolio NAV
            current_volatility: Realized or implied volatility
            
        Returns:
            float: Position size in notional units
            
        Mathematical Foundation:
            Should incorporate Kelly Criterion or volatility targeting.
        """
        pass
    
    def update_position(self, new_position: float, price: float) -> Dict[str, float]:
        """
        Execute position change and record transaction costs.
        
        Args:
            new_position: Target position size
            price: Execution price
            
        Returns:
            Dict with 'trade_size', 'slippage', 'commission'
        """
        trade_size = new_position - self.position
        
        # Transaction cost modeling
        spread_bps = self.config.get('spread_bps', 5.0)
        commission_bps = self.config.get('commission_bps', 1.0)
        
        slippage = abs(trade_size) * price * (spread_bps / 10000)
        commission = abs(trade_size) * price * (commission_bps / 10000)
        
        self.position = new_position
        
        return {
            'trade_size': trade_size,
            'slippage': slippage,
            'commission': commission,
            'total_cost': slippage + commission
        }
    
    @abstractmethod
    def calculate_risk_metrics(
        self,
        returns: pd.Series,
        risk_free_rate: float = 0.0
    ) -> RiskMetrics:
        """
        Compute risk-adjusted performance metrics.
        
        Args:
            returns: Time series of strategy returns
            risk_free_rate: Annual risk-free rate
            
        Returns:
            RiskMetrics: Comprehensive risk metrics
        """
        pass


class BaseRiskManager(ABC):
    """
    Abstract base class for risk management systems.
    
    Implements hard limits on portfolio risk exposure.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize risk manager with constraints.
        
        Args:
            config: Dictionary with risk limits
                   (max_drawdown, max_leverage, var_limit, etc.)
        """
        self.config = config
        self.breach_count: int = 0
        
    @abstractmethod
    def check_position_limit(
        self,
        proposed_position: float,
        portfolio_value: float
    ) -> Tuple[bool, Optional[str]]:
        """
        Verify position doesn't violate risk limits.
        
        Args:
            proposed_position: Intended position size
            portfolio_value: Current portfolio NAV
            
        Returns:
            Tuple of (is_allowed, reason_if_blocked)
        """
        pass
    
    @abstractmethod
    def calculate_var(
        self,
        returns: pd.Series,
        confidence: float = 0.95,
        method: str = "historical"
    ) -> float:
        """
        Calculate Value at Risk.
        
        Args:
            returns: Historical return series
            confidence: VaR confidence level (0.95 = 95%)
            method: 'historical', 'parametric', or 'monte_carlo'
            
        Returns:
            float: VaR estimate (positive number representing loss)
        """
        pass
    
    @abstractmethod
    def calculate_portfolio_greeks(
        self,
        positions: Dict[str, float],
        market_data: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate portfolio-level Greeks.
        
        Args:
            positions: Current holdings by instrument
            market_data: Latest market prices and volatilities
            
        Returns:
            Dict with 'delta', 'gamma', 'vega', 'theta', 'rho'
        """
        pass


def run_stationarity_tests(data: pd.Series) -> Dict[str, Any]:
    """
    Perform statistical tests for mean reversion properties.
    
    Args:
        data: Time series to test
        
    Returns:
        Dict containing:
            - adf_statistic: Augmented Dickey-Fuller test statistic
            - adf_pvalue: p-value for ADF test
            - is_stationary: bool (True if p < 0.05)
            - hurst_exponent: Hurst exponent (H < 0.5 = mean reverting)
    """
    from statsmodels.tsa.stattools import adfuller
    
    # ADF Test
    adf_result = adfuller(data.dropna(), autolag='AIC')
    
    # Hurst Exponent (simplified calculation)
    lags = range(2, min(100, len(data) // 2))
    tau = [np.std(np.subtract(data[lag:].values, data[:-lag].values)) 
           for lag in lags]
    
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    hurst = poly[0] * 2.0
    
    return {
        'adf_statistic': adf_result[0],
        'adf_pvalue': adf_result[1],
        'is_stationary': adf_result[1] < 0.05,
        'hurst_exponent': hurst,
        'is_mean_reverting': hurst < 0.5
    }


def check_outliers(data: pd.Series, n_std: float = 5.0) -> pd.Series:
    """
    Detect outliers using z-score method.
    
    Args:
        data: Time series to check
        n_std: Number of standard deviations for threshold
        
    Returns:
        pd.Series: Boolean mask (True = outlier)
    """
    z_scores = np.abs((data - data.mean()) / data.std())
    return z_scores > n_std
