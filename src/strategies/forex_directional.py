"""
Forex Directional Strategies for Position Sizing

Three Hypothesis-Driven Strategies:
1. Mean Reversion (Z-Score): Exploit EUR/USD mean-reverting behavior
2. Momentum (Dual MA): Capture trending periods
3. Combined Regime: Switch between MR and Momentum

Mathematical Foundation:
- Mean Reversion: Assumes Ornstein-Uhlenbeck process (dX = kappa*(theta - X)*dt + sigma*dW)
- Momentum: Exponential Moving Average with trend strength filter
- Regime Detection: Hurst Exponent < 0.5 (mean-reverting) vs > 0.5 (trending)
"""

from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import stats


class ZScoreMeanReversion:
    """
    Mean Reversion Strategy using Z-Score of price
    
    Hypothesis: EUR/USD reverts to mean over 20-60 day periods
    Entry: |Z-score| > entry_threshold (price far from mean)
    Exit: |Z-score| < exit_threshold (price near mean)
    
    Parameters
    ----------
    lookback : int
        Lookback period for mean/std calculation (default: 60)
    entry_threshold : float
        Z-score threshold for entry (default: 2.0 = 2 std devs)
    exit_threshold : float
        Z-score threshold for exit (default: 0.5)
    """
    
    def __init__(
        self,
        lookback: int = 60,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5
    ):
        self.lookback = lookback
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.name = "Z-Score Mean Reversion"
        
    def calculate_zscore(self, prices: pd.Series) -> pd.Series:
        """
        Calculate rolling Z-score
        
        Z = (X - mu) / sigma
        where mu = rolling mean, sigma = rolling std
        """
        rolling_mean = prices.rolling(window=self.lookback).mean()
        rolling_std = prices.rolling(window=self.lookback).std()
        
        zscore = (prices - rolling_mean) / rolling_std
        return zscore
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on Z-score
        
        Returns
        -------
        DataFrame with columns:
            - zscore: Calculated Z-score
            - signal: +1 (long), -1 (short), 0 (neutral)
            - position: Current position (+1, -1, 0)
        """
        df = data.copy()
        
        # Calculate Z-score
        df['zscore'] = self.calculate_zscore(df['close'])
        
        # Initialize signal
        df['signal'] = 0
        
        # Entry signals (vectorized)
        # Long when price is significantly below mean (negative Z-score)
        df.loc[df['zscore'] < -self.entry_threshold, 'signal'] = 1
        
        # Short when price is significantly above mean (positive Z-score)
        df.loc[df['zscore'] > self.entry_threshold, 'signal'] = -1
        
        # Track position state
        df['position'] = 0
        current_position = 0
        
        for i in range(len(df)):
            if pd.isna(df['zscore'].iloc[i]):
                df.loc[df.index[i], 'position'] = 0
                current_position = 0
                continue
            
            zscore_val = df['zscore'].iloc[i]
            signal_val = df['signal'].iloc[i]
            
            # Entry logic
            if current_position == 0 and signal_val != 0:
                current_position = signal_val
            
            # Exit logic (mean reversion)
            elif current_position != 0:
                if abs(zscore_val) < self.exit_threshold:
                    current_position = 0
                # Stop loss: opposite threshold breached
                elif (current_position == 1 and zscore_val > self.entry_threshold) or \
                     (current_position == -1 and zscore_val < -self.entry_threshold):
                    current_position = 0
            
            df.loc[df.index[i], 'position'] = current_position
        
        return df


class DualMovingAverageMomentum:
    """
    Momentum Strategy using Dual Moving Average Crossover
    
    Hypothesis: EUR/USD exhibits trending behavior during risk-on/risk-off periods
    Entry: Fast MA crosses above/below Slow MA
    Exit: Opposite crossover
    
    Parameters
    ----------
    fast_period : int
        Fast moving average period (default: 20)
    slow_period : int
        Slow moving average period (default: 50)
    use_ema : bool
        Use EMA instead of SMA (default: True)
    trend_filter : float
        Minimum slope for trend confirmation (default: None)
    """
    
    def __init__(
        self,
        fast_period: int = 20,
        slow_period: int = 50,
        use_ema: bool = True,
        trend_filter: Optional[float] = None
    ):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.use_ema = use_ema
        self.trend_filter = trend_filter
        self.name = "Dual MA Momentum"
        
    def calculate_ma(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate moving average (SMA or EMA)"""
        if self.use_ema:
            return prices.ewm(span=period, adjust=False).mean()
        else:
            return prices.rolling(window=period).mean()
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate momentum signals based on MA crossover
        
        Returns
        -------
        DataFrame with columns:
            - fast_ma: Fast moving average
            - slow_ma: Slow moving average
            - signal: +1 (bullish cross), -1 (bearish cross), 0 (no cross)
            - position: Current position
        """
        df = data.copy()
        
        # Calculate moving averages
        df['fast_ma'] = self.calculate_ma(df['close'], self.fast_period)
        df['slow_ma'] = self.calculate_ma(df['close'], self.slow_period)
        
        # Calculate trend strength (optional filter)
        if self.trend_filter is not None:
            df['slope'] = df['slow_ma'].diff(5) / df['slow_ma'].shift(5)
        
        # Initialize signal
        df['signal'] = 0
        df['position'] = 0
        
        # Detect crossovers
        df['ma_diff'] = df['fast_ma'] - df['slow_ma']
        df['ma_diff_prev'] = df['ma_diff'].shift(1)
        
        # Bullish crossover: fast crosses above slow
        bullish_cross = (df['ma_diff'] > 0) & (df['ma_diff_prev'] <= 0)
        
        # Bearish crossover: fast crosses below slow
        bearish_cross = (df['ma_diff'] < 0) & (df['ma_diff_prev'] >= 0)
        
        # Apply trend filter if specified
        if self.trend_filter is not None:
            bullish_cross = bullish_cross & (df['slope'] > self.trend_filter)
            bearish_cross = bearish_cross & (df['slope'] < -self.trend_filter)
        
        df.loc[bullish_cross, 'signal'] = 1
        df.loc[bearish_cross, 'signal'] = -1
        
        # Track position (forward-fill signals until next crossover)
        current_position = 0
        for i in range(len(df)):
            if pd.notna(df['fast_ma'].iloc[i]) and pd.notna(df['slow_ma'].iloc[i]):
                if df['signal'].iloc[i] != 0:
                    current_position = df['signal'].iloc[i]
                df.loc[df.index[i], 'position'] = current_position
        
        return df


class RegimeSwitchingStrategy:
    """
    Combined Strategy: Switch between Mean Reversion and Momentum
    
    Hypothesis: EUR/USD behavior changes across volatility regimes
    - Low volatility → Mean reversion
    - High volatility → Momentum
    
    Regime Detection:
    - Volatility percentile
    - Hurst Exponent (H < 0.5 = mean-reverting, H > 0.5 = trending)
    
    Parameters
    ----------
    vol_lookback : int
        Lookback for volatility calculation (default: 60)
    vol_threshold : float
        Volatility percentile threshold (default: 0.5 = median)
    mr_strategy : ZScoreMeanReversion
        Mean reversion strategy instance
    mom_strategy : DualMovingAverageMomentum
        Momentum strategy instance
    """
    
    def __init__(
        self,
        vol_lookback: int = 60,
        vol_threshold: float = 0.5,
        mr_strategy: Optional[ZScoreMeanReversion] = None,
        mom_strategy: Optional[DualMovingAverageMomentum] = None
    ):
        self.vol_lookback = vol_lookback
        self.vol_threshold = vol_threshold
        self.mr_strategy = mr_strategy or ZScoreMeanReversion()
        self.mom_strategy = mom_strategy or DualMovingAverageMomentum()
        self.name = "Regime Switching"
        
    def calculate_hurst_exponent(self, prices: pd.Series, window: int = 100) -> float:
        """
        Calculate Hurst Exponent using R/S analysis
        
        H < 0.5: Mean reverting
        H = 0.5: Random walk
        H > 0.5: Trending
        """
        if len(prices) < window:
            return 0.5
        
        prices = prices[-window:]
        
        # Calculate log returns
        log_returns = np.log(prices / prices.shift(1)).dropna()
        
        if len(log_returns) < 10:
            return 0.5
        
        # R/S analysis
        lags = range(2, min(20, len(log_returns) // 2))
        tau = []
        
        for lag in lags:
            # Partition into blocks
            blocks = [log_returns[i:i+lag] for i in range(0, len(log_returns), lag) if len(log_returns[i:i+lag]) == lag]
            
            rs_values = []
            for block in blocks:
                mean_adj = block - block.mean()
                cumsum = mean_adj.cumsum()
                R = cumsum.max() - cumsum.min()
                S = block.std()
                if S > 0:
                    rs_values.append(R / S)
            
            if len(rs_values) > 0:
                tau.append(np.mean(rs_values))
        
        if len(tau) < 2:
            return 0.5
        
        # Linear regression: log(R/S) = H * log(lag) + const
        log_lags = np.log(list(lags[:len(tau)]))
        log_tau = np.log(tau)
        
        hurst = np.polyfit(log_lags, log_tau, 1)[0]
        return np.clip(hurst, 0, 1)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals based on regime detection
        
        Returns
        -------
        DataFrame with columns:
            - volatility: Rolling volatility
            - vol_regime: 'low' or 'high'
            - hurst: Hurst exponent
            - regime: 'mean_reverting' or 'trending'
            - position_mr: Mean reversion position
            - position_mom: Momentum position
            - position: Final combined position
        """
        df = data.copy()
        
        # Calculate volatility regime
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=self.vol_lookback).std() * np.sqrt(252)
        
        vol_percentile = df['volatility'].rolling(window=252).quantile(self.vol_threshold)
        df['vol_regime'] = 'low'
        df.loc[df['volatility'] > vol_percentile, 'vol_regime'] = 'high'
        
        # Calculate Hurst exponent (rolling)
        df['hurst'] = np.nan
        for i in range(100, len(df)):
            df.loc[df.index[i], 'hurst'] = self.calculate_hurst_exponent(df['close'].iloc[:i+1])
        
        # Regime classification
        df['regime'] = 'mean_reverting'
        df.loc[df['hurst'] > 0.55, 'regime'] = 'trending'
        
        # Generate signals from both strategies
        df_mr = self.mr_strategy.generate_signals(df)
        df_mom = self.mom_strategy.generate_signals(df)
        
        df['position_mr'] = df_mr['position']
        df['position_mom'] = df_mom['position']
        df['zscore'] = df_mr['zscore']
        df['fast_ma'] = df_mom['fast_ma']
        df['slow_ma'] = df_mom['slow_ma']
        
        # Combined position based on regime
        df['position'] = 0
        
        # Low volatility → Mean reversion
        low_vol_mask = df['vol_regime'] == 'low'
        df.loc[low_vol_mask, 'position'] = df.loc[low_vol_mask, 'position_mr']
        
        # High volatility → Momentum
        high_vol_mask = df['vol_regime'] == 'high'
        df.loc[high_vol_mask, 'position'] = df.loc[high_vol_mask, 'position_mom']
        
        return df


def backtest_strategy(
    strategy_df: pd.DataFrame,
    position_sizer,
    transaction_cost: float = 0.0001
) -> Dict[str, float]:
    """
    Backtest strategy with position sizing
    
    Parameters
    ----------
    strategy_df : DataFrame
        DataFrame with 'position' and 'returns' columns
    position_sizer : PositionSizer
        Position sizing model from src.risk.position_sizing
    transaction_cost : float
        Transaction cost as percentage (default: 1 basis point)
        
    Returns
    -------
    dict : Performance metrics
    """
    df = strategy_df.copy()
    
    # Calculate position sizes
    returns = df['returns'].values
    position_sizes = []
    equity_curve = [1.0]
    
    for i in range(len(df)):
        if i < position_sizer.lookback:
            position_sizes.append(0)
            equity_curve.append(equity_curve[-1])
            continue
        
        window_returns = returns[max(0, i-position_sizer.lookback):i]
        equity_array = np.array(equity_curve[-position_sizer.lookback:])
        
        pos_size = position_sizer.calculate_position_size(
            returns=window_returns,
            equity_curve=equity_array
        )
        
        position_sizes.append(pos_size)
        
        # Calculate strategy return
        directional_signal = df['position'].iloc[i]
        
        if i > 0:
            # Apply transaction costs on position changes
            position_change = abs(df['position'].iloc[i] - df['position'].iloc[i-1])
            tc_cost = position_change * transaction_cost
            
            # Strategy return = position * position_size * market_return - transaction_cost
            strategy_return = directional_signal * pos_size * returns[i] - tc_cost
        else:
            strategy_return = 0
        
        new_equity = equity_curve[-1] * (1 + strategy_return)
        equity_curve.append(new_equity)
    
    df['position_size'] = position_sizes
    df['equity'] = equity_curve[1:]
    
    # Calculate metrics
    equity_series = pd.Series(equity_curve[1:], index=df.index)
    equity_returns = equity_series.pct_change().dropna()
    
    total_return = (equity_series.iloc[-1] / equity_series.iloc[0] - 1) * 100
    
    if len(equity_returns) > 0 and equity_returns.std() > 0:
        sharpe_ratio = (equity_returns.mean() / equity_returns.std()) * np.sqrt(252)
    else:
        sharpe_ratio = 0
    
    # Calculate max drawdown
    cummax = equity_series.cummax()
    drawdown = (equity_series - cummax) / cummax
    max_drawdown = drawdown.min() * 100
    
    # Win rate
    winning_trades = (equity_returns > 0).sum()
    total_trades = len(equity_returns[equity_returns != 0])
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    # Number of trades (position changes)
    num_trades = (df['position'].diff() != 0).sum()
    
    metrics = {
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'num_trades': num_trades,
        'avg_position_size': np.mean(position_sizes),
        'final_equity': equity_series.iloc[-1]
    }
    
    return df, metrics
