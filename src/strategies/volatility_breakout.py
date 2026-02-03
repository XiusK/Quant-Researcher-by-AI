"""
Volatility Breakout Strategy

Signal Logic:
    Entry: Price breaks out of Bollinger Bands or ATR-based channel
    Exit: Return to mean or opposite breakout
    
Best for: Gold during high volatility regimes
"""

from typing import Dict, Any
import numpy as np
import pandas as pd
from src.base import BaseStrategy, Signal, RiskMetrics


class VolatilityBreakoutStrategy(BaseStrategy):
    """
    Breakout strategy using Bollinger Bands or ATR channels.
    
    Configuration:
        - method: 'bollinger' or 'atr'
        - bb_period: Bollinger Bands period (e.g., 20)
        - bb_std: Number of standard deviations (e.g., 2.0)
        - atr_period: ATR period (e.g., 14)
        - atr_mult: ATR multiplier for channel (e.g., 2.0)
        - min_volatility: Minimum volatility to trade
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """Initialize volatility breakout strategy."""
        super().__init__(name=name, config=config)
        
        self.method = config.get('method', 'bollinger')
        
    def generate_signal(
        self,
        market_data: pd.DataFrame,
        timestamp: pd.Timestamp
    ) -> Signal:
        """Generate breakout signal."""
        available_data = market_data[market_data.index <= timestamp]
        
        min_lookback = 50
        if len(available_data) < min_lookback:
            return Signal(
                timestamp=timestamp,
                direction=0,
                confidence=0.0,
                target_position=0.0,
                metadata={'reason': 'insufficient_data'}
            )
        
        current_price = available_data['close'].iloc[-1]
        
        if self.method == 'bollinger':
            signal_result = self._bollinger_signal(available_data, current_price)
        else:  # atr
            signal_result = self._atr_channel_signal(available_data, current_price)
        
        direction, confidence, metadata = signal_result
        
        # Check minimum volatility filter
        if 'realized_vol' in available_data.columns:
            current_vol = available_data['realized_vol'].iloc[-1]
            min_vol = self.config.get('min_volatility', 0.10)
            
            if current_vol < min_vol:
                direction = 0
                confidence = 0.0
                metadata['filtered'] = 'low_volatility'
        
        return Signal(
            timestamp=timestamp,
            direction=direction,
            confidence=confidence,
            target_position=direction,
            metadata=metadata
        )
    
    def _bollinger_signal(
        self,
        data: pd.DataFrame,
        current_price: float
    ) -> tuple:
        """Generate signal using Bollinger Bands."""
        bb_period = self.config.get('bb_period', 20)
        bb_std = self.config.get('bb_std', 2.0)
        
        bb_middle = data['close'].rolling(window=bb_period).mean()
        bb_std_dev = data['close'].rolling(window=bb_period).std()
        bb_upper = bb_middle + (bb_std * bb_std_dev)
        bb_lower = bb_middle - (bb_std * bb_std_dev)
        
        current_middle = bb_middle.iloc[-1]
        current_upper = bb_upper.iloc[-1]
        current_lower = bb_lower.iloc[-1]
        
        # Calculate position within bands
        bb_width = current_upper - current_lower
        distance_from_middle = current_price - current_middle
        
        direction = 0
        confidence = 0.0
        
        # Breakout above upper band
        if current_price > current_upper:
            direction = 1
            confidence = min(abs(distance_from_middle) / bb_width, 1.0)
            
        # Breakout below lower band
        elif current_price < current_lower:
            direction = -1
            confidence = min(abs(distance_from_middle) / bb_width, 1.0)
            
        # Exit if price returns to middle
        elif self.position != 0 and abs(distance_from_middle) < bb_width * 0.2:
            direction = 0
            confidence = 0.0
        
        metadata = {
            'bb_middle': current_middle,
            'bb_upper': current_upper,
            'bb_lower': current_lower,
            'bb_width': bb_width,
            'distance_from_middle': distance_from_middle
        }
        
        return direction, confidence, metadata
    
    def _atr_channel_signal(
        self,
        data: pd.DataFrame,
        current_price: float
    ) -> tuple:
        """Generate signal using ATR channels."""
        atr_period = self.config.get('atr_period', 14)
        atr_mult = self.config.get('atr_mult', 2.0)
        
        # Calculate ATR
        if 'atr_14' in data.columns:
            atr = data['atr_14'].iloc[-1]
        else:
            high_low = data['high'] - data['low']
            atr = high_low.rolling(window=atr_period).mean().iloc[-1]
        
        # Channel around recent average
        lookback = self.config.get('channel_lookback', 20)
        channel_middle = data['close'].rolling(window=lookback).mean().iloc[-1]
        channel_upper = channel_middle + (atr_mult * atr)
        channel_lower = channel_middle - (atr_mult * atr)
        
        direction = 0
        confidence = 0.0
        
        # Breakout above channel
        if current_price > channel_upper:
            direction = 1
            breakout_distance = current_price - channel_upper
            confidence = min(breakout_distance / atr, 1.0)
            
        # Breakout below channel
        elif current_price < channel_lower:
            direction = -1
            breakout_distance = channel_lower - current_price
            confidence = min(breakout_distance / atr, 1.0)
            
        # Exit if back in channel
        elif self.position != 0 and channel_lower <= current_price <= channel_upper:
            direction = 0
            confidence = 0.0
        
        metadata = {
            'channel_middle': channel_middle,
            'channel_upper': channel_upper,
            'channel_lower': channel_lower,
            'atr': atr
        }
        
        return direction, confidence, metadata
    
    def calculate_position_size(
        self,
        signal: Signal,
        portfolio_value: float,
        current_volatility: float
    ) -> float:
        """Inverse volatility position sizing."""
        if signal.direction == 0:
            return 0.0
        
        # Inverse volatility: trade smaller in high volatility
        base_position = self.config.get('max_position_pct', 0.25) * portfolio_value
        
        vol_adjustment = 0.15 / (current_volatility + 1e-6)  # Target 15% vol
        vol_adjustment = np.clip(vol_adjustment, 0.5, 2.0)  # Limit adjustment
        
        position = base_position * signal.confidence * vol_adjustment
        
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
