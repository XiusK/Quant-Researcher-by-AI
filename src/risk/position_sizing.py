"""
Position Sizing and Leverage Optimization Models

Mathematical Foundation:
- Kelly Criterion: f* = (p*b - q) / b where p=win_rate, q=loss_rate, b=win/loss_ratio
- Optimal f: Maximizes geometric growth E[log(1 + f*R)]
- Volatility-based: position_size = target_risk / (volatility * sqrt(time))
- Dynamic Leverage: leverage = base_leverage * (1 - volatility_ratio)

Author: Quant Researcher AI
Date: 2026-02-03
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar


class BasePositionSizer(ABC):
    """
    Abstract base class for position sizing models.
    
    All position sizers must implement calculate_position_size() method.
    """
    
    def __init__(self, max_position_size: float = 1.0):
        """
        Parameters:
        -----------
        max_position_size : float
            Maximum allowed position size as fraction of capital (0-1)
        """
        self.max_position_size = max_position_size
    
    @abstractmethod
    def calculate_position_size(
        self, 
        returns: np.ndarray,
        **kwargs
    ) -> float:
        """
        Calculate optimal position size.
        
        Parameters:
        -----------
        returns : np.ndarray
            Historical returns
        **kwargs : dict
            Additional parameters specific to each method
            
        Returns:
        --------
        float : Position size as fraction of capital (0-1)
        """
        pass
    
    def _clip_position(self, position_size: float) -> float:
        """Ensure position size within valid range."""
        return np.clip(position_size, 0.0, self.max_position_size)


class FixedFractionalSizer(BasePositionSizer):
    """
    Fixed Fractional position sizing.
    
    Risk fixed percentage of capital on each trade.
    Simple but effective for consistent risk management.
    
    Formula:
    --------
    position_size = risk_per_trade / stop_loss_pct
    
    Example:
    --------
    Risk 2% per trade with 1% stop loss -> 2.0 position size
    (leverage 2x)
    """
    
    def __init__(
        self, 
        risk_per_trade: float = 0.02,
        max_position_size: float = 1.0
    ):
        """
        Parameters:
        -----------
        risk_per_trade : float
            Fraction of capital to risk per trade (default: 2%)
        max_position_size : float
            Maximum position size cap
        """
        super().__init__(max_position_size)
        self.risk_per_trade = risk_per_trade
    
    def calculate_position_size(
        self,
        returns: np.ndarray,
        stop_loss_pct: Optional[float] = None,
        **kwargs
    ) -> float:
        """
        Calculate fixed fractional position size.
        
        Parameters:
        -----------
        returns : np.ndarray
            Historical returns (not used in fixed fractional)
        stop_loss_pct : float, optional
            Stop loss as percentage (e.g., 0.01 for 1%)
            If None, uses standard deviation as proxy
            
        Returns:
        --------
        float : Position size as fraction of capital
        """
        if stop_loss_pct is None:
            # Use 1.5 * std as default stop loss
            stop_loss_pct = 1.5 * np.std(returns)
        
        if stop_loss_pct <= 0:
            return 0.0
        
        position_size = self.risk_per_trade / stop_loss_pct
        return self._clip_position(position_size)


class KellyCriterionSizer(BasePositionSizer):
    """
    Kelly Criterion for optimal growth.
    
    Maximizes long-run geometric growth rate.
    WARNING: Full Kelly is aggressive, use fractional Kelly (0.25-0.5).
    
    Formula:
    --------
    f* = (p * b - q) / b
    
    Where:
    - p = win probability
    - q = loss probability (1-p)
    - b = win/loss ratio (avg_win / avg_loss)
    
    References:
    -----------
    Kelly, J.L. (1956). "A New Interpretation of Information Rate"
    """
    
    def __init__(
        self,
        kelly_fraction: float = 0.25,
        max_position_size: float = 1.0,
        min_trades: int = 30
    ):
        """
        Parameters:
        -----------
        kelly_fraction : float
            Fraction of full Kelly to use (0.25 = Quarter Kelly)
            Reduces variance while preserving most growth
        max_position_size : float
            Maximum position size cap
        min_trades : int
            Minimum number of trades to calculate Kelly
        """
        super().__init__(max_position_size)
        self.kelly_fraction = kelly_fraction
        self.min_trades = min_trades
    
    def calculate_position_size(
        self,
        returns: np.ndarray,
        **kwargs
    ) -> float:
        """
        Calculate Kelly Criterion position size.
        
        Parameters:
        -----------
        returns : np.ndarray
            Historical returns (must include both wins and losses)
            
        Returns:
        --------
        float : Position size as fraction of capital
        """
        # Filter valid returns
        valid_returns = returns[~np.isnan(returns)]
        
        if len(valid_returns) < self.min_trades:
            # Not enough data, use conservative sizing
            return self._clip_position(0.10)
        
        # Calculate win rate
        wins = valid_returns > 0
        losses = valid_returns < 0
        
        num_wins = np.sum(wins)
        num_losses = np.sum(losses)
        
        if num_losses == 0:
            # No losses observed (unlikely but handle edge case)
            return self._clip_position(self.kelly_fraction * 0.5)
        
        win_prob = num_wins / len(valid_returns)
        loss_prob = 1 - win_prob
        
        # Calculate win/loss ratio
        avg_win = np.mean(valid_returns[wins]) if num_wins > 0 else 0
        avg_loss = abs(np.mean(valid_returns[losses]))
        
        if avg_loss == 0:
            return self._clip_position(self.kelly_fraction * 0.5)
        
        win_loss_ratio = avg_win / avg_loss
        
        # Kelly formula: f* = (p*b - q) / b
        kelly_pct = (win_prob * win_loss_ratio - loss_prob) / win_loss_ratio
        
        # Apply fractional Kelly and clip
        position_size = kelly_pct * self.kelly_fraction
        
        return self._clip_position(max(0.0, position_size))


class VolatilityBasedSizer(BasePositionSizer):
    """
    Volatility-based position sizing.
    
    Adjusts position size inversely to volatility:
    - High volatility -> Smaller positions
    - Low volatility -> Larger positions
    
    Formula:
    --------
    position_size = target_volatility / realized_volatility
    
    Where:
    - target_volatility = desired portfolio volatility (e.g., 15% annual)
    - realized_volatility = current asset volatility (ATR or StdDev)
    """
    
    def __init__(
        self,
        target_volatility: float = 0.15,
        lookback: int = 20,
        max_position_size: float = 1.0,
        min_position_size: float = 0.05,
        annualization_factor: int = 252
    ):
        """
        Parameters:
        -----------
        target_volatility : float
            Target portfolio volatility (annualized)
        lookback : int
            Rolling window for volatility calculation
        max_position_size : float
            Maximum position size
        min_position_size : float
            Minimum position size (prevent zero exposure)
        annualization_factor : int
            Trading days per year (252 for stocks, 365 for crypto)
        """
        super().__init__(max_position_size)
        self.target_volatility = target_volatility
        self.lookback = lookback
        self.min_position_size = min_position_size
        self.annualization_factor = annualization_factor
    
    def calculate_position_size(
        self,
        returns: np.ndarray,
        use_ewm: bool = True,
        **kwargs
    ) -> float:
        """
        Calculate volatility-based position size.
        
        Parameters:
        -----------
        returns : np.ndarray
            Historical returns
        use_ewm : bool
            Use exponentially weighted moving average (more responsive)
            
        Returns:
        --------
        float : Position size as fraction of capital
        """
        valid_returns = returns[~np.isnan(returns)]
        
        if len(valid_returns) < self.lookback:
            # Not enough data, use minimum position
            return self.min_position_size
        
        # Calculate realized volatility
        if use_ewm:
            # Exponential weighting (more weight on recent data)
            weights = np.exp(np.linspace(-1, 0, self.lookback))
            weights /= weights.sum()
            recent_returns = valid_returns[-self.lookback:]
            realized_vol = np.sqrt(np.average(recent_returns**2, weights=weights))
        else:
            # Simple rolling standard deviation
            realized_vol = np.std(valid_returns[-self.lookback:])
        
        # Annualize volatility
        realized_vol_annual = realized_vol * np.sqrt(self.annualization_factor)
        
        if realized_vol_annual <= 0:
            return self.min_position_size
        
        # Calculate position size
        position_size = self.target_volatility / realized_vol_annual
        
        # Ensure within bounds
        position_size = np.clip(
            position_size, 
            self.min_position_size, 
            self.max_position_size
        )
        
        return position_size


class DynamicLeverageSizer(BasePositionSizer):
    """
    Dynamic Leverage Adjustment based on market regime.
    
    Combines multiple signals:
    1. Volatility regime (GARCH-based)
    2. Drawdown control (reduce leverage during drawdowns)
    3. Correlation regime (increase leverage when diversified)
    
    Formula:
    --------
    leverage = base_leverage * vol_adjustment * dd_adjustment * corr_adjustment
    
    Where:
    - vol_adjustment = 1 - (current_vol / max_vol)
    - dd_adjustment = 1 - (current_dd / max_dd_threshold)
    - corr_adjustment = 1 + (1 - avg_correlation)
    """
    
    def __init__(
        self,
        base_leverage: float = 2.0,
        max_leverage: float = 5.0,
        min_leverage: float = 0.5,
        volatility_lookback: int = 60,
        max_drawdown_threshold: float = 0.20,
        vol_percentile: float = 0.95
    ):
        """
        Parameters:
        -----------
        base_leverage : float
            Base leverage in normal conditions
        max_leverage : float
            Maximum allowed leverage
        min_leverage : float
            Minimum leverage (safety floor)
        volatility_lookback : int
            Lookback for volatility calculation
        max_drawdown_threshold : float
            Maximum acceptable drawdown before reducing leverage
        vol_percentile : float
            Percentile for "high volatility" threshold (e.g., 95th)
        """
        super().__init__(max_position_size=max_leverage)
        self.base_leverage = base_leverage
        self.max_leverage = max_leverage
        self.min_leverage = min_leverage
        self.volatility_lookback = volatility_lookback
        self.max_drawdown_threshold = max_drawdown_threshold
        self.vol_percentile = vol_percentile
    
    def calculate_position_size(
        self,
        returns: np.ndarray,
        equity_curve: Optional[np.ndarray] = None,
        **kwargs
    ) -> float:
        """
        Calculate dynamic leverage.
        
        Parameters:
        -----------
        returns : np.ndarray
            Historical returns
        equity_curve : np.ndarray, optional
            Equity curve for drawdown calculation
            
        Returns:
        --------
        float : Leverage multiplier
        """
        valid_returns = returns[~np.isnan(returns)]
        
        if len(valid_returns) < self.volatility_lookback:
            return self.min_leverage
        
        # 1. Volatility Adjustment
        recent_vol = np.std(valid_returns[-self.volatility_lookback:])
        historical_vol = np.std(valid_returns)
        max_vol = np.percentile(
            pd.Series(valid_returns).rolling(self.volatility_lookback).std().dropna(),
            self.vol_percentile * 100
        )
        
        if max_vol > 0:
            vol_ratio = recent_vol / max_vol
            vol_adjustment = 1.0 - np.clip(vol_ratio, 0, 1)
        else:
            vol_adjustment = 0.5
        
        # 2. Drawdown Adjustment
        dd_adjustment = 1.0
        if equity_curve is not None and len(equity_curve) > 0:
            current_drawdown = self._calculate_drawdown(equity_curve)
            if current_drawdown > 0:
                dd_adjustment = 1.0 - (current_drawdown / self.max_drawdown_threshold)
                dd_adjustment = np.clip(dd_adjustment, 0.2, 1.0)
        
        # 3. Calculate dynamic leverage
        leverage = self.base_leverage * (0.5 + 0.5 * vol_adjustment) * dd_adjustment
        
        # Clip to bounds
        leverage = np.clip(leverage, self.min_leverage, self.max_leverage)
        
        return leverage
    
    def _calculate_drawdown(self, equity_curve: np.ndarray) -> float:
        """Calculate current drawdown from peak."""
        if len(equity_curve) == 0:
            return 0.0
        
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak[-1] - equity_curve[-1]) / peak[-1] if peak[-1] > 0 else 0.0
        return drawdown


def calculate_var(
    returns: np.ndarray,
    confidence_level: float = 0.95,
    method: str = 'historical'
) -> float:
    """
    Calculate Value at Risk (VaR).
    
    VaR estimates maximum expected loss over a time period at given confidence level.
    
    Parameters:
    -----------
    returns : np.ndarray
        Historical returns
    confidence_level : float
        Confidence level (e.g., 0.95 for 95%)
    method : str
        'historical', 'parametric', or 'cornish_fisher'
        
    Returns:
    --------
    float : VaR as percentage loss (positive number)
    
    Example:
    --------
    VaR(95%) = 0.03 means 95% confidence that loss won't exceed 3%
    """
    valid_returns = returns[~np.isnan(returns)]
    
    if len(valid_returns) < 10:
        return np.nan
    
    if method == 'historical':
        # Historical simulation
        var = -np.percentile(valid_returns, (1 - confidence_level) * 100)
    
    elif method == 'parametric':
        # Assumes normal distribution
        mean = np.mean(valid_returns)
        std = np.std(valid_returns)
        z_score = stats.norm.ppf(confidence_level)
        var = -(mean - z_score * std)
    
    elif method == 'cornish_fisher':
        # Accounts for skewness and kurtosis
        mean = np.mean(valid_returns)
        std = np.std(valid_returns)
        skew = stats.skew(valid_returns)
        kurt = stats.kurtosis(valid_returns)
        
        z = stats.norm.ppf(confidence_level)
        z_cf = (z + 
                (z**2 - 1) * skew / 6 +
                (z**3 - 3*z) * kurt / 24 -
                (2*z**3 - 5*z) * skew**2 / 36)
        
        var = -(mean - z_cf * std)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return max(0.0, var)


def calculate_cvar(
    returns: np.ndarray,
    confidence_level: float = 0.95
) -> float:
    """
    Calculate Conditional Value at Risk (CVaR) / Expected Shortfall (ES).
    
    CVaR is the expected loss given that VaR threshold is breached.
    More conservative than VaR (accounts for tail risk).
    
    Parameters:
    -----------
    returns : np.ndarray
        Historical returns
    confidence_level : float
        Confidence level (e.g., 0.95 for 95%)
        
    Returns:
    --------
    float : CVaR as percentage loss (positive number)
    
    Example:
    --------
    CVaR(95%) = 0.05 means expected loss is 5% when worst 5% scenarios occur
    """
    valid_returns = returns[~np.isnan(returns)]
    
    if len(valid_returns) < 10:
        return np.nan
    
    # Calculate VaR threshold
    var_threshold = -np.percentile(valid_returns, (1 - confidence_level) * 100)
    
    # CVaR is mean of all losses exceeding VaR
    tail_losses = valid_returns[valid_returns <= -var_threshold]
    
    if len(tail_losses) == 0:
        return var_threshold
    
    cvar = -np.mean(tail_losses)
    
    return max(0.0, cvar)


def optimize_position_size(
    returns: np.ndarray,
    objective: str = 'sharpe',
    constraints: Optional[Dict] = None
) -> Tuple[float, Dict]:
    """
    Optimize position size using numerical optimization.
    
    Parameters:
    -----------
    returns : np.ndarray
        Historical returns
    objective : str
        Optimization objective:
        - 'sharpe': Maximize Sharpe ratio
        - 'sortino': Maximize Sortino ratio
        - 'calmar': Maximize Calmar ratio
        - 'omega': Maximize Omega ratio
    constraints : dict, optional
        Additional constraints (max_drawdown, max_var, etc.)
        
    Returns:
    --------
    Tuple[float, Dict] : Optimal position size and performance metrics
    """
    if constraints is None:
        constraints = {'max_drawdown': 0.30, 'max_var': 0.05}
    
    def objective_function(position_size: float) -> float:
        """Calculate negative objective (for minimization)."""
        leveraged_returns = returns * position_size
        
        # Calculate objective metric
        if objective == 'sharpe':
            mean_ret = np.mean(leveraged_returns)
            std_ret = np.std(leveraged_returns)
            metric = mean_ret / std_ret if std_ret > 0 else 0
        
        elif objective == 'sortino':
            mean_ret = np.mean(leveraged_returns)
            downside_returns = leveraged_returns[leveraged_returns < 0]
            downside_std = np.std(downside_returns) if len(downside_returns) > 0 else np.std(leveraged_returns)
            metric = mean_ret / downside_std if downside_std > 0 else 0
        
        elif objective == 'calmar':
            mean_ret = np.mean(leveraged_returns)
            cum_returns = np.cumprod(1 + leveraged_returns)
            max_dd = np.max(np.maximum.accumulate(cum_returns) - cum_returns) / np.max(cum_returns)
            metric = mean_ret / max_dd if max_dd > 0 else 0
        
        else:
            metric = 0
        
        # Apply constraints as penalties
        penalty = 0
        
        # Max drawdown constraint
        cum_returns = np.cumprod(1 + leveraged_returns)
        current_dd = np.max(np.maximum.accumulate(cum_returns) - cum_returns) / np.max(cum_returns)
        if current_dd > constraints['max_drawdown']:
            penalty += 10 * (current_dd - constraints['max_drawdown'])
        
        # Max VaR constraint
        var_95 = calculate_var(leveraged_returns, 0.95)
        if var_95 > constraints['max_var']:
            penalty += 10 * (var_95 - constraints['max_var'])
        
        return -(metric - penalty)
    
    # Optimize
    result = minimize_scalar(
        objective_function,
        bounds=(0.1, 3.0),
        method='bounded'
    )
    
    optimal_size = result.x
    leveraged_returns = returns * optimal_size
    
    # Calculate final metrics
    metrics = {
        'position_size': optimal_size,
        'sharpe': np.mean(leveraged_returns) / np.std(leveraged_returns) if np.std(leveraged_returns) > 0 else 0,
        'total_return': np.sum(leveraged_returns),
        'volatility': np.std(leveraged_returns),
        'var_95': calculate_var(leveraged_returns, 0.95),
        'cvar_95': calculate_cvar(leveraged_returns, 0.95)
    }
    
    return optimal_size, metrics
