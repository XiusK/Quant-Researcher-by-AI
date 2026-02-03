"""
Microstructure-based Intraday Strategy

For high-frequency data (5m, 15m, 1h):
- Order flow imbalance
- Volume-weighted signals
- Time-of-day effects
- Session patterns (Asian/European/US)

Best for: Gold 5m-1h data from Kaggle
"""

from typing import Dict, Any
import numpy as np
import pandas as pd
from src.base import BaseStrategy, Signal, RiskMetrics


class MicrostructureStrategy(BaseStrategy):
    """
    Intraday strategy exploiting microstructure patterns.
    
    Configuration:
        - volume_imbalance_threshold: Volume spike threshold (e.g., 2.0)
        - price_impact_window: Window for price impact calculation
        - session_filter: Trade only specific sessions ('all', 'us', 'europe', 'asia')
        - bid_ask_proxy: Use high-low range as bid-ask proxy
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """Initialize microstructure strategy."""
        super().__init__(name=name, config=config)
        
    def generate_signal(
        self,
        market_data: pd.DataFrame,
        timestamp: pd.Timestamp
    ) -> Signal:
        """Generate signal based on microstructure indicators."""
        available_data = market_data[market_data.index <= timestamp]
        
        min_lookback = 100
        if len(available_data) < min_lookback:
            return Signal(
                timestamp=timestamp,
                direction=0,
                confidence=0.0,
                target_position=0.0,
                metadata={'reason': 'insufficient_data'}
            )
        
        # Session filter
        session_filter = self.config.get('session_filter', 'all')
        if not self._is_valid_session(timestamp, session_filter):
            return Signal(
                timestamp=timestamp,
                direction=0,
                confidence=0.0,
                target_position=0.0,
                metadata={'reason': 'outside_trading_session'}
            )
        
        current_price = available_data['close'].iloc[-1]
        
        # Volume imbalance
        if 'volume' in available_data.columns and available_data['volume'].sum() > 0:
            volume_signal = self._volume_imbalance_signal(available_data)
        else:
            volume_signal = (0, 0.0, {})
        
        # Price impact (using HL range as proxy for liquidity)
        impact_signal = self._price_impact_signal(available_data)
        
        # Combine signals
        direction, confidence, metadata = self._combine_signals(
            volume_signal,
            impact_signal
        )
        
        return Signal(
            timestamp=timestamp,
            direction=direction,
            confidence=confidence,
            target_position=direction,
            metadata=metadata
        )
    
    def _is_valid_session(self, timestamp: pd.Timestamp, session: str) -> bool:
        """Check if timestamp falls in valid trading session."""
        if session == 'all':
            return True
        
        hour = timestamp.hour
        
        if session == 'asia':
            return 0 <= hour < 9  # Asian session
        elif session == 'europe':
            return 7 <= hour < 16  # European session
        elif session == 'us':
            return 13 <= hour < 22  # US session
        
        return True
    
    def _volume_imbalance_signal(self, data: pd.DataFrame) -> tuple:
        """Detect abnormal volume patterns."""
        if 'volume' not in data.columns:
            return 0, 0.0, {}
        
        current_volume = data['volume'].iloc[-1]
        avg_volume = data['volume'].rolling(window=20).mean().iloc[-1]
        
        if avg_volume == 0:
            return 0, 0.0, {}
        
        volume_ratio = current_volume / avg_volume
        threshold = self.config.get('volume_imbalance_threshold', 2.0)
        
        # High volume with price direction
        price_change = data['close'].pct_change().iloc[-1]
        
        direction = 0
        confidence = 0.0
        
        if volume_ratio > threshold:
            if price_change > 0:
                direction = 1  # Volume buying pressure
                confidence = min((volume_ratio - threshold) / threshold, 1.0)
            elif price_change < 0:
                direction = -1  # Volume selling pressure
                confidence = min((volume_ratio - threshold) / threshold, 1.0)
        
        metadata = {
            'volume_ratio': volume_ratio,
            'price_change': price_change
        }
        
        return direction, confidence, metadata
    
    def _price_impact_signal(self, data: pd.DataFrame) -> tuple:
        """Detect price impact using high-low range."""
        window = self.config.get('price_impact_window', 10)
        
        # Average true range
        atr = data['atr_14'].iloc[-1] if 'atr_14' in data.columns else None
        
        if atr is None:
            hl_range = data['high'] - data['low']
            atr = hl_range.rolling(window=14).mean().iloc[-1]
        
        current_range = data['high'].iloc[-1] - data['low'].iloc[-1]
        
        # Tight range suggests low liquidity, avoid trading
        if current_range < atr * 0.3:
            return 0, 0.0, {'reason': 'low_liquidity'}
        
        # Wide range with directional close
        close = data['close'].iloc[-1]
        high = data['high'].iloc[-1]
        low = data['low'].iloc[-1]
        
        close_position = (close - low) / (high - low) if (high - low) > 0 else 0.5
        
        direction = 0
        confidence = 0.0
        
        # Close near high = bullish
        if close_position > 0.8:
            direction = 1
            confidence = (close_position - 0.8) / 0.2
        # Close near low = bearish
        elif close_position < 0.2:
            direction = -1
            confidence = (0.2 - close_position) / 0.2
        
        metadata = {
            'close_position': close_position,
            'range_atr_ratio': current_range / atr if atr > 0 else 0
        }
        
        return direction, confidence, metadata
    
    def _combine_signals(self, signal1: tuple, signal2: tuple) -> tuple:
        """Combine multiple signals with weights."""
        dir1, conf1, meta1 = signal1
        dir2, conf2, meta2 = signal2
        
        # Average direction weighted by confidence
        if conf1 + conf2 > 0:
            combined_direction = (dir1 * conf1 + dir2 * conf2) / (conf1 + conf2)
            
            # Convert to discrete direction
            if combined_direction > 0.3:
                direction = 1
            elif combined_direction < -0.3:
                direction = -1
            else:
                direction = 0
            
            confidence = (conf1 + conf2) / 2
        else:
            direction = 0
            confidence = 0.0
        
        metadata = {**meta1, **meta2}
        
        return direction, confidence, metadata
    
    def calculate_position_size(
        self,
        signal: Signal,
        portfolio_value: float,
        current_volatility: float
    ) -> float:
        """Position sizing for high-frequency strategy."""
        if signal.direction == 0:
            return 0.0
        
        # Smaller positions for intraday
        base_position = self.config.get('max_position_pct', 0.10) * portfolio_value
        
        position = base_position * signal.confidence
        
        max_leverage = self.config.get('max_leverage', 1.0)
        position = np.clip(position, 0, max_leverage * portfolio_value)
        
        return position * signal.direction
    
    def calculate_risk_metrics(
        self,
        returns: pd.Series,
        risk_free_rate: float = 0.0
    ) -> RiskMetrics:
        """Calculate risk metrics (annualize for intraday data)."""
        if len(returns) == 0:
            raise ValueError("Empty returns series")
        
        # Annualization factor depends on data frequency
        # For intraday, use higher frequency factor
        periods_per_day = self.config.get('periods_per_day', 24)  # For hourly
        ann_factor = np.sqrt(252 * periods_per_day)
        
        excess_returns = returns - risk_free_rate / (252 * periods_per_day)
        sharpe = (excess_returns.mean() / excess_returns.std()) * ann_factor if excess_returns.std() > 0 else 0.0
        
        downside = excess_returns[excess_returns < 0]
        sortino = (excess_returns.mean() / downside.std()) * ann_factor if len(downside) > 0 and downside.std() > 0 else 0.0
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()
        
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean()
        
        annual_return = (1 + returns.mean()) ** (252 * periods_per_day) - 1
        calmar = annual_return / abs(max_dd) if max_dd != 0 else 0.0
        
        return RiskMetrics(
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            var_95=var_95,
            cvar_95=cvar_95,
            calmar_ratio=calmar
        )
