"""
Backtesting Engine

Vectorized backtesting with comprehensive transaction cost modeling.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
from dataclasses import dataclass
from src.base import BaseStrategy, RiskMetrics


@dataclass
class BacktestResult:
    """Container for backtest results."""
    strategy_name: str
    total_return: float
    annualized_return: float
    risk_metrics: RiskMetrics
    trades: pd.DataFrame
    equity_curve: pd.Series
    positions: pd.Series
    metadata: Dict[str, Any]


class BacktestEngine:
    """
    Event-driven backtesting engine with transaction costs.
    
    Features:
    - No look-ahead bias (strict temporal ordering)
    - Realistic transaction costs (spread + commission + slippage)
    - Position tracking and risk management
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        spread_bps: float = 5.0,
        commission_bps: float = 1.0,
        slippage_bps: float = 2.0
    ):
        """
        Initialize backtest engine.
        
        Args:
            initial_capital: Starting portfolio value
            spread_bps: Bid-ask spread in basis points
            commission_bps: Commission in basis points
            slippage_bps: Market impact slippage in basis points
        """
        self.initial_capital = initial_capital
        self.spread_bps = spread_bps
        self.commission_bps = commission_bps
        self.slippage_bps = slippage_bps
        
    def run(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        warmup_period: int = 0
    ) -> BacktestResult:
        """
        Run backtest on historical data.
        
        Args:
            strategy: Strategy instance to test
            data: OHLC data with features
            warmup_period: Number of initial bars to skip
            
        Returns:
            BacktestResult with comprehensive metrics
        """
        print(f"\n{'='*60}")
        print(f"Backtesting: {strategy.name}")
        print(f"{'='*60}")
        print(f"Data: {data.index[0]} to {data.index[-1]} ({len(data)} bars)")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        
        # Initialize tracking
        portfolio_value = self.initial_capital
        cash = self.initial_capital
        position_qty = 0.0
        position_value = 0.0
        
        trades: List[Dict] = []
        equity_curve: List[float] = []
        positions: List[float] = []
        timestamps: List[pd.Timestamp] = []
        
        # Run simulation
        for i, timestamp in enumerate(data.index):
            if i < warmup_period:
                equity_curve.append(portfolio_value)
                positions.append(0.0)
                timestamps.append(timestamp)
                continue
            
            # Get market data up to current timestamp
            available_data = data.iloc[:i+1]
            current_price = available_data['close'].iloc[-1]
            
            # Calculate current volatility
            if 'realized_vol' in available_data.columns:
                current_vol = available_data['realized_vol'].iloc[-1]
            else:
                returns = available_data['close'].pct_change()
                current_vol = returns.std() * np.sqrt(252)
            
            current_vol = max(current_vol, 0.01)  # Floor at 1%
            
            # Generate signal
            signal = strategy.generate_signal(available_data, timestamp)
            
            # Calculate target position
            target_position_value = strategy.calculate_position_size(
                signal,
                portfolio_value,
                current_vol
            )
            
            target_qty = target_position_value / current_price if current_price > 0 else 0.0
            
            # Execute trade if position change
            if abs(target_qty - position_qty) > 0.01:  # Minimum trade size
                trade_qty = target_qty - position_qty
                trade_value = abs(trade_qty * current_price)
                
                # Transaction costs
                total_cost_bps = self.spread_bps + self.commission_bps + self.slippage_bps
                transaction_cost = trade_value * (total_cost_bps / 10000)
                
                # Update position and cash
                position_qty = target_qty
                cash -= (trade_qty * current_price + transaction_cost)
                
                # Record trade
                trades.append({
                    'timestamp': timestamp,
                    'price': current_price,
                    'quantity': trade_qty,
                    'value': trade_qty * current_price,
                    'cost': transaction_cost,
                    'signal': signal.direction,
                    'confidence': signal.confidence
                })
            
            # Update portfolio value
            position_value = position_qty * current_price
            portfolio_value = cash + position_value
            
            # Track
            equity_curve.append(portfolio_value)
            positions.append(position_qty)
            timestamps.append(timestamp)
        
        # Create DataFrames
        equity_series = pd.Series(equity_curve, index=timestamps)
        position_series = pd.Series(positions, index=timestamps)
        trades_df = pd.DataFrame(trades)
        
        # Calculate returns
        returns = equity_series.pct_change().dropna()
        
        # Calculate metrics
        total_return = (equity_series.iloc[-1] - self.initial_capital) / self.initial_capital
        
        trading_days = len(returns)
        years = trading_days / 252
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0
        
        risk_metrics = strategy.calculate_risk_metrics(returns)
        
        # Trade statistics
        n_trades = len(trades_df)
        avg_trade_cost = trades_df['cost'].mean() if n_trades > 0 else 0.0
        total_costs = trades_df['cost'].sum() if n_trades > 0 else 0.0
        
        metadata = {
            'n_trades': n_trades,
            'avg_trade_cost': avg_trade_cost,
            'total_costs': total_costs,
            'final_value': equity_series.iloc[-1],
            'peak_value': equity_series.max(),
            'trading_days': trading_days
        }
        
        # Print summary
        print(f"\nResults:")
        print(f"  Total Return: {total_return:.2%}")
        print(f"  Annualized Return: {annualized_return:.2%}")
        print(f"  Sharpe Ratio: {risk_metrics.sharpe_ratio:.2f}")
        print(f"  Max Drawdown: {risk_metrics.max_drawdown:.2%}")
        print(f"  Number of Trades: {n_trades}")
        print(f"  Total Costs: ${total_costs:,.2f}")
        
        return BacktestResult(
            strategy_name=strategy.name,
            total_return=total_return,
            annualized_return=annualized_return,
            risk_metrics=risk_metrics,
            trades=trades_df,
            equity_curve=equity_series,
            positions=position_series,
            metadata=metadata
        )
