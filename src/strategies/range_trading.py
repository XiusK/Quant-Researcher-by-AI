"""
Range Trading Strategy

Signal Logic:
    Entry: Buy at support, sell at resistance (mean reversion within range)
    Exit: Profit target or range breakdown
    
Best for: Gold during low volatility consolidation periods
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from src.base import BaseStrategy, Signal, RiskMetrics


class RangeTradingStrategy(BaseStrategy):
    """
    Mean reversion within identified price ranges.
    
    Configuration:
        - range_lookback: Period to identify support/resistance (e.g., 50)
        - entry_threshold: Distance from support/resistance to enter (e.g., 0.02)
        - profit_target: Percentage profit target (e.g., 0.015)
        - breakout_exit: Exit if range breaks (True/False)
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """Initialize range trading strategy."""
        super().__init__(name=name, config=config)
        
        self.support: Optional[float] = None
        self.resistance: Optional[float] = None
        self.range_high: Optional[float] = None
        self.range_low: Optional[float] = None
        
    def generate_signal(
        self,
        market_data: pd.DataFrame,
        timestamp: pd.Timestamp
    ) -> Signal:
        """Generate range trading signal."""
        available_data = market_data[market_data.index <= timestamp]
        
        range_lookback = self.config.get('range_lookback', 50)
        
        if len(available_data) < range_lookback:
            return Signal(
                timestamp=timestamp,
                direction=0,
                confidence=0.0,
                target_position=0.0,
                metadata={'reason': 'insufficient_data'}
            )
        
        # Identify range
        lookback_data = available_data.iloc[-range_lookback:]
        self.range_high = lookback_data['high'].max()
        self.range_low = lookback_data['low'].min()
        
        # Calculate support and resistance (more conservative)
        # Use 75th and 25th percentile for more robust levels
        self.resistance = lookback_data['high'].quantile(0.90)
        self.support = lookback_data['low'].quantile(0.10)
        
        range_width = self.resistance - self.support
        current_price = available_data['close'].iloc[-1]
        
        # Check if in a valid range (not trending)
        price_std = lookback_data['close'].std()
        is_ranging = range_width < (current_price * 0.15)  # Range is < 15% of price
        
        direction = 0
        confidence = 0.0
        metadata = {
            'support': self.support,
            'resistance': self.resistance,
            'range_high': self.range_high,
            'range_low': self.range_low,
            'range_width': range_width,
            'is_ranging': is_ranging
        }
        
        if not is_ranging:
            return Signal(
                timestamp=timestamp,
                direction=0,
                confidence=0.0,
                target_position=0.0,
                metadata={**metadata, 'reason': 'trending_market'}
            )
        
        entry_threshold = self.config.get('entry_threshold', 0.02)
        
        # Distance from support/resistance as percentage
        dist_to_support = (current_price - self.support) / current_price
        dist_to_resistance = (self.resistance - current_price) / current_price
        
        # Entry near support (buy)
        if dist_to_support < entry_threshold and self.position <= 0:
            direction = 1
            confidence = 1.0 - (dist_to_support / entry_threshold)
            
        # Entry near resistance (sell)
        elif dist_to_resistance < entry_threshold and self.position >= 0:
            direction = -1
            confidence = 1.0 - (dist_to_resistance / entry_threshold)
            
        # Exit at profit target
        elif self.position != 0:
            profit_target = self.config.get('profit_target', 0.015)
            
            if self.position > 0:  # Long position
                target_price = self.resistance - (range_width * 0.1)
                if current_price >= target_price:
                    direction = 0
                    metadata['exit_reason'] = 'profit_target'
                    
            else:  # Short position
                target_price = self.support + (range_width * 0.1)
                if current_price <= target_price:
                    direction = 0
                    metadata['exit_reason'] = 'profit_target'
        
        # Breakout exit
        if self.config.get('breakout_exit', True) and self.position != 0:
            if current_price > self.range_high or current_price < self.range_low:
                direction = 0
                metadata['exit_reason'] = 'range_breakout'
        
        return Signal(
            timestamp=timestamp,
            direction=direction,
            confidence=confidence,
            target_position=direction,
            metadata=metadata
        )
    
    def calculate_position_size(
        self,
        signal: Signal,
        portfolio_value: float,
        current_volatility: float
    ) -> float:
        """Fixed fractional position sizing."""
        if signal.direction == 0:
            return 0.0
        
        # Use smaller positions for range trading (higher frequency)
        base_position = self.config.get('max_position_pct', 0.15) * portfolio_value
        position = base_position * signal.confidence
        
        max_leverage = self.config.get('max_leverage', 1.0)
        position = np.clip(position, 0, max_leverage * portfolio_value)
        
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
