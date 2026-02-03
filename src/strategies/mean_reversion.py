"""
Mean Reversion Strategy using Ornstein-Uhlenbeck Process

Signal Generation Logic:
    Z-score = (Current Price - Theta) / Sigma
    
    Entry: |Z-score| > entry_threshold
    Exit: |Z-score| < exit_threshold or stop-loss triggered
    
Position Sizing: Kelly Criterion adjusted for estimation error
"""

from typing import Dict, Any
import numpy as np
import pandas as pd
from src.base import BaseStrategy, Signal, RiskMetrics
from src.models.ornstein_uhlenbeck import OrnsteinUhlenbeck, AssetClass


class MeanReversionStrategy(BaseStrategy):
    """
    OU-based mean reversion strategy with dynamic recalibration.
    
    Configuration Parameters:
        - entry_threshold: Z-score threshold for entry (e.g., 2.0)
        - exit_threshold: Z-score threshold for exit (e.g., 0.5)
        - stop_loss: Maximum loss per trade as fraction of capital (e.g., 0.02)
        - recalibration_window: Days between model recalibration (e.g., 30)
        - lookback_period: Historical data for calibration (e.g., 252)
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """Initialize mean reversion strategy."""
        super().__init__(name=name, config=config)
        
        # Validate required config
        required = ['entry_threshold', 'exit_threshold', 'lookback_period']
        for key in required:
            if key not in config:
                raise ValueError(f"Missing required config parameter: {key}")
        
        self.model = OrnsteinUhlenbeck(asset_class=AssetClass.FX)
        self.last_calibration: Optional[pd.Timestamp] = None
        
    def generate_signal(
        self,
        market_data: pd.DataFrame,
        timestamp: pd.Timestamp
    ) -> Signal:
        """
        Generate mean reversion signal based on OU process.
        
        Returns:
            Signal with direction: +1 (long), -1 (short), 0 (neutral)
        """
        # Validate no look-ahead bias
        available_data = market_data[market_data.index <= timestamp]
        
        if len(available_data) < self.config['lookback_period']:
            return Signal(
                timestamp=timestamp,
                direction=0,
                confidence=0.0,
                target_position=0.0,
                metadata={'reason': 'insufficient_data'}
            )
        
        # Recalibrate model if needed
        recalib_days = self.config.get('recalibration_window', 30)
        should_recalibrate = (
            self.last_calibration is None or
            (timestamp - self.last_calibration).days >= recalib_days
        )
        
        if should_recalibrate:
            lookback = available_data['close'].iloc[-self.config['lookback_period']:]
            
            # Use returns instead of price levels for stationarity
            lookback_returns = lookback.pct_change().dropna()
            
            try:
                self.model.calibrate(lookback_returns)
                self.last_calibration = timestamp
            except ValueError as e:
                # If calibration fails, use simple mean/std
                print(f"Warning: OU calibration failed, using simple statistics")
                self.model.is_calibrated = False
        
        # Calculate Z-score
        current_price = available_data['close'].iloc[-1]
        
        if self.model.is_calibrated:
            theta = self.model.params.params['theta']
            sigma = self.model.params.params['sigma']
        else:
            # Fallback to simple rolling statistics
            lookback = available_data['close'].iloc[-self.config['lookback_period']:]
            theta = lookback.mean()
            sigma = lookback.std()
        
        z_score = (current_price - theta) / sigma if sigma > 0 else 0.0
        
        # Signal logic
        entry_threshold = self.config['entry_threshold']
        exit_threshold = self.config.get('exit_threshold', 0.5)
        
        direction = 0
        confidence = 0.0
        
        if abs(z_score) > entry_threshold:
            # Price has deviated significantly -> bet on reversion
            direction = -1 if z_score > 0 else 1  # Short if overvalued, long if undervalued
            confidence = min(abs(z_score) / entry_threshold - 1.0, 1.0)
        elif abs(z_score) < exit_threshold and self.position != 0:
            # Close existing position
            direction = 0
            confidence = 0.0
        
        return Signal(
            timestamp=timestamp,
            direction=direction,
            confidence=confidence,
            target_position=direction,  # Will be scaled by position sizing
            metadata={
                'z_score': z_score,
                'theta': theta,
                'kappa': self.model.params.params['kappa'] if self.model.is_calibrated else None,
                'half_life': self.model.half_life() if self.model.is_calibrated else None
            }
        )
    
    def calculate_position_size(
        self,
        signal: Signal,
        portfolio_value: float,
        current_volatility: float
    ) -> float:
        """
        Kelly Criterion adjusted for parameter uncertainty.
        
        Formula:
            f = (p * b - q) / b * adjustment_factor
            where adjustment_factor accounts for estimation error
        """
        if signal.direction == 0:
            return 0.0
        
        # Volatility targeting
        target_vol = self.config.get('target_volatility', 0.15)  # 15% annualized
        vol_scalar = target_vol / (current_volatility + 1e-6)
        
        # Base position as fraction of portfolio
        base_position = self.config.get('max_position_pct', 0.2) * portfolio_value
        
        # Scale by confidence and volatility
        position = base_position * signal.confidence * vol_scalar
        
        # Apply leverage limit
        max_leverage = self.config.get('max_leverage', 1.0)
        position = np.clip(position, -max_leverage * portfolio_value, max_leverage * portfolio_value)
        
        return position * signal.direction
    
    def calculate_risk_metrics(
        self,
        returns: pd.Series,
        risk_free_rate: float = 0.0
    ) -> RiskMetrics:
        """Calculate comprehensive risk metrics."""
        if len(returns) == 0:
            raise ValueError("Empty returns series")
        
        # Annualization factor (assume daily returns)
        ann_factor = np.sqrt(252)
        
        # Sharpe Ratio
        excess_returns = returns - risk_free_rate / 252
        sharpe = (excess_returns.mean() / excess_returns.std()) * ann_factor if excess_returns.std() > 0 else 0.0
        
        # Sortino Ratio (downside deviation)
        downside = excess_returns[excess_returns < 0]
        sortino = (excess_returns.mean() / downside.std()) * ann_factor if len(downside) > 0 and downside.std() > 0 else 0.0
        
        # Maximum Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()
        
        # VaR and CVaR (95%)
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean()
        
        # Calmar Ratio
        annual_return = (1 + returns.mean()) ** 252 - 1
        calmar = annual_return / abs(max_dd) if max_dd != 0 else 0.0
        
        return RiskMetrics(
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            var_95=var_95,
            cvar_95=cvar_95,
            calmar_ratio=calmar
        )
