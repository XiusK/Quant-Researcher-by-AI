"""
Momentum/Trend Following Strategy

Signal Logic:
    Entry: Price crosses above/below moving average with confirmation
    Exit: Opposite signal or trailing stop
    
Position Sizing: Volatility-adjusted with pyramid potential
"""

from typing import Dict, Any
import numpy as np
import pandas as pd
from src.base import BaseStrategy, Signal, RiskMetrics


class MomentumStrategy(BaseStrategy):
    """
    Trend-following strategy using moving average crossover.
    
    Configuration:
        - fast_ma: Fast moving average period (e.g., 20)
        - slow_ma: Slow moving average period (e.g., 50)
        - ma_type: 'sma' or 'ema'
        - min_trend_strength: Minimum slope for trend confirmation
        - trailing_stop_atr: ATR multiplier for trailing stop
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """Initialize momentum strategy."""
        super().__init__(name=name, config=config)
        
        required = ['fast_ma', 'slow_ma']
        for key in required:
            if key not in config:
                raise ValueError(f"Missing required config: {key}")
        
        self.last_signal: int = 0
        self.entry_price: float = 0.0
        
    def generate_signal(
        self,
        market_data: pd.DataFrame,
        timestamp: pd.Timestamp
    ) -> Signal:
        """Generate momentum signal based on MA crossover."""
        available_data = market_data[market_data.index <= timestamp]
        
        min_lookback = max(self.config['fast_ma'], self.config['slow_ma']) + 10
        if len(available_data) < min_lookback:
            return Signal(
                timestamp=timestamp,
                direction=0,
                confidence=0.0,
                target_position=0.0,
                metadata={'reason': 'insufficient_data'}
            )
        
        # Calculate moving averages
        ma_type = self.config.get('ma_type', 'ema')
        fast_period = self.config['fast_ma']
        slow_period = self.config['slow_ma']
        
        if ma_type == 'sma':
            fast_ma = available_data['close'].rolling(window=fast_period).mean()
            slow_ma = available_data['close'].rolling(window=slow_period).mean()
        else:  # ema
            fast_ma = available_data['close'].ewm(span=fast_period, adjust=False).mean()
            slow_ma = available_data['close'].ewm(span=slow_period, adjust=False).mean()
        
        # Current values
        current_fast = fast_ma.iloc[-1]
        current_slow = slow_ma.iloc[-1]
        prev_fast = fast_ma.iloc[-2]
        prev_slow = slow_ma.iloc[-2]
        current_price = available_data['close'].iloc[-1]
        
        # Crossover detection
        bullish_cross = (prev_fast <= prev_slow) and (current_fast > current_slow)
        bearish_cross = (prev_fast >= prev_slow) and (current_fast < current_slow)
        
        # Trend strength (slope of slow MA)
        slow_ma_slope = (slow_ma.iloc[-1] - slow_ma.iloc[-5]) / slow_ma.iloc[-5]
        trend_strength = abs(slow_ma_slope)
        
        min_strength = self.config.get('min_trend_strength', 0.001)
        
        direction = 0
        confidence = 0.0
        
        if bullish_cross and slow_ma_slope > 0:
            direction = 1
            confidence = min(trend_strength / min_strength, 1.0)
            self.last_signal = 1
            self.entry_price = current_price
            
        elif bearish_cross and slow_ma_slope < 0:
            direction = -1
            confidence = min(trend_strength / min_strength, 1.0)
            self.last_signal = -1
            self.entry_price = current_price
            
        # Check trailing stop if in position
        elif self.position != 0:
            atr = available_data['atr_14'].iloc[-1] if 'atr_14' in available_data.columns else current_price * 0.02
            trailing_stop_mult = self.config.get('trailing_stop_atr', 2.0)
            
            if self.position > 0:  # Long position
                stop_price = current_price - (trailing_stop_mult * atr)
                if current_price < stop_price:
                    direction = 0  # Exit
            else:  # Short position
                stop_price = current_price + (trailing_stop_mult * atr)
                if current_price > stop_price:
                    direction = 0  # Exit
        
        return Signal(
            timestamp=timestamp,
            direction=direction,
            confidence=confidence,
            target_position=direction,
            metadata={
                'fast_ma': current_fast,
                'slow_ma': current_slow,
                'trend_strength': trend_strength,
                'slope': slow_ma_slope
            }
        )
    
    def calculate_position_size(
        self,
        signal: Signal,
        portfolio_value: float,
        current_volatility: float
    ) -> float:
        """Volatility-adjusted position sizing."""
        if signal.direction == 0:
            return 0.0
        
        target_vol = self.config.get('target_volatility', 0.15)
        vol_scalar = target_vol / (current_volatility + 1e-6)
        
        base_position = self.config.get('max_position_pct', 0.3) * portfolio_value
        position = base_position * signal.confidence * vol_scalar
        
        max_leverage = self.config.get('max_leverage', 1.0)
        position = np.clip(position, -max_leverage * portfolio_value, max_leverage * portfolio_value)
        
        return position * signal.direction
    
    def calculate_risk_metrics(
        self,
        returns: pd.Series,
        risk_free_rate: float = 0.0
    ) -> RiskMetrics:
        """Calculate risk metrics."""
        if len(returns) == 0:
            raise ValueError("Empty returns series")
        
        ann_factor = np.sqrt(252)
        
        excess_returns = returns - risk_free_rate / 252
        sharpe = (excess_returns.mean() / excess_returns.std()) * ann_factor if excess_returns.std() > 0 else 0.0
        
        downside = excess_returns[excess_returns < 0]
        sortino = (excess_returns.mean() / downside.std()) * ann_factor if len(downside) > 0 and downside.std() > 0 else 0.0
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()
        
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean()
        
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
