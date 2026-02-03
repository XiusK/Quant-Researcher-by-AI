"""
XAUUSD Strategy Hunt - Find Edge in Gold Trading

This script tests multiple strategies on Gold data to identify profitable edges.

Strategies Tested:
1. Mean Reversion (OU-based)
2. Momentum (MA Crossover)
3. Volatility Breakout (Bollinger/ATR)
4. Range Trading (Support/Resistance)

Workflow:
1. Download XAUUSD data
2. Calculate technical features
3. Split train/test
4. Backtest all strategies
5. Compare risk-adjusted returns
6. Identify best performer
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

from src.data import load_xauusd_from_kaggle, calculate_features, split_train_test, list_available_timeframes
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.momentum import MomentumStrategy
from src.strategies.volatility_breakout import VolatilityBreakoutStrategy
from src.strategies.range_trading import RangeTradingStrategy
from src.backtest import BacktestEngine, BacktestResult


def download_and_prepare_data(timeframe: str = "1d"):
    """Load XAUUSD data from Kaggle files and calculate features."""
    print("\n" + "="*60)
    print("STEP 1: DATA PREPARATION")
    print("="*60)
    
    # List available timeframes
    available = list_available_timeframes()
    print(f"Available timeframes: {', '.join(available)}")
    
    # Load data from local Kaggle files
    data = load_xauusd_from_kaggle(
        timeframe=timeframe,
        data_folder="Kaggles_XAUUSD_Data"
    )
    
    # Calculate features
    print("\nCalculating technical features...")
    data = calculate_features(data)
    
    # Remove NaN rows from feature calculation
    data = data.dropna()
    
    print(f"Final dataset: {len(data)} rows")
    
    return data


def create_strategy_configs() -> Dict[str, Dict]:
    """Define configuration for each strategy."""
    
    configs = {
        'Mean_Reversion': {
            'class': MeanReversionStrategy,
            'params': {
                'entry_threshold': 2.0,
                'exit_threshold': 0.5,
                'lookback_period': 252,
                'recalibration_window': 30,
                'max_position_pct': 0.20,
                'target_volatility': 0.15,
                'max_leverage': 1.0,
                'spread_bps': 5.0,
                'commission_bps': 1.0
            }
        },
        
        'Momentum_Fast': {
            'class': MomentumStrategy,
            'params': {
                'fast_ma': 10,
                'slow_ma': 30,
                'ma_type': 'ema',
                'min_trend_strength': 0.001,
                'trailing_stop_atr': 2.0,
                'max_position_pct': 0.30,
                'target_volatility': 0.15,
                'max_leverage': 1.0,
                'spread_bps': 5.0,
                'commission_bps': 1.0
            }
        },
        
        'Momentum_Slow': {
            'class': MomentumStrategy,
            'params': {
                'fast_ma': 50,
                'slow_ma': 200,
                'ma_type': 'sma',
                'min_trend_strength': 0.002,
                'trailing_stop_atr': 3.0,
                'max_position_pct': 0.30,
                'target_volatility': 0.15,
                'max_leverage': 1.0,
                'spread_bps': 5.0,
                'commission_bps': 1.0
            }
        },
        
        'Volatility_Breakout_BB': {
            'class': VolatilityBreakoutStrategy,
            'params': {
                'method': 'bollinger',
                'bb_period': 20,
                'bb_std': 2.0,
                'min_volatility': 0.10,
                'max_position_pct': 0.25,
                'max_leverage': 1.0,
                'spread_bps': 5.0,
                'commission_bps': 1.0
            }
        },
        
        'Volatility_Breakout_ATR': {
            'class': VolatilityBreakoutStrategy,
            'params': {
                'method': 'atr',
                'atr_period': 14,
                'atr_mult': 2.0,
                'channel_lookback': 20,
                'min_volatility': 0.10,
                'max_position_pct': 0.25,
                'max_leverage': 1.0,
                'spread_bps': 5.0,
                'commission_bps': 1.0
            }
        },
        
        'Range_Trading': {
            'class': RangeTradingStrategy,
            'params': {
                'range_lookback': 50,
                'entry_threshold': 0.02,
                'profit_target': 0.015,
                'breakout_exit': True,
                'max_position_pct': 0.15,
                'max_leverage': 1.0,
                'spread_bps': 5.0,
                'commission_bps': 1.0
            }
        }
    }
    
    return configs


def run_backtests(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    configs: Dict
) -> Dict[str, BacktestResult]:
    """Run backtests for all strategies."""
    
    print("\n" + "="*60)
    print("STEP 2: BACKTESTING STRATEGIES")
    print("="*60)
    
    engine = BacktestEngine(
        initial_capital=100000.0,
        spread_bps=5.0,
        commission_bps=1.0,
        slippage_bps=2.0
    )
    
    results = {}
    
    for strategy_name, config in configs.items():
        print(f"\n[Testing: {strategy_name}]")
        
        # Create strategy
        strategy = config['class'](
            name=strategy_name,
            config=config['params']
        )
        
        # Run on test data (out-of-sample)
        result = engine.run(
            strategy=strategy,
            data=test_data,
            warmup_period=50  # Skip first 50 bars for warmup
        )
        
        results[strategy_name] = result
    
    return results


def compare_strategies(results: Dict[str, BacktestResult]) -> pd.DataFrame:
    """Create comparison table of all strategies."""
    
    print("\n" + "="*60)
    print("STEP 3: STRATEGY COMPARISON")
    print("="*60)
    
    comparison_data = []
    
    for name, result in results.items():
        comparison_data.append({
            'Strategy': name,
            'Total Return': result.total_return,
            'Annual Return': result.annualized_return,
            'Sharpe': result.risk_metrics.sharpe_ratio,
            'Sortino': result.risk_metrics.sortino_ratio,
            'Max DD': result.risk_metrics.max_drawdown,
            'Calmar': result.risk_metrics.calmar_ratio,
            'Trades': result.metadata['n_trades'],
            'Total Costs': result.metadata['total_costs']
        })
    
    df = pd.DataFrame(comparison_data)
    df = df.sort_values('Sharpe', ascending=False)
    
    # Print formatted table
    print("\nPerformance Comparison (Sorted by Sharpe Ratio):")
    print("-" * 120)
    
    for idx, row in df.iterrows():
        print(f"{row['Strategy']:30s} | "
              f"Return: {row['Total Return']:>7.2%} | "
              f"Sharpe: {row['Sharpe']:>5.2f} | "
              f"MaxDD: {row['Max DD']:>7.2%} | "
              f"Trades: {row['Trades']:>4.0f}")
    
    return df


def visualize_results(
    data: pd.DataFrame,
    results: Dict[str, BacktestResult],
    comparison_df: pd.DataFrame
):
    """Create comprehensive visualization."""
    
    print("\n" + "="*60)
    print("STEP 4: VISUALIZATION")
    print("="*60)
    
    fig = plt.figure(figsize=(16, 12))
    
    # Layout: 3 rows x 2 columns
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Equity Curves
    ax1 = fig.add_subplot(gs[0, :])
    for name, result in results.items():
        normalized = (result.equity_curve / result.equity_curve.iloc[0] - 1) * 100
        ax1.plot(normalized.index, normalized.values, label=name, alpha=0.8, linewidth=1.5)
    
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax1.set_title('Equity Curves (Normalized to 0%)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Return (%)')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 2. Sharpe Ratio Comparison
    ax2 = fig.add_subplot(gs[1, 0])
    colors = ['green' if x > 0 else 'red' for x in comparison_df['Sharpe']]
    ax2.barh(comparison_df['Strategy'], comparison_df['Sharpe'], color=colors, alpha=0.7)
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax2.axvline(x=1, color='blue', linestyle='--', alpha=0.5, label='Sharpe = 1')
    ax2.set_xlabel('Sharpe Ratio')
    ax2.set_title('Sharpe Ratio Comparison', fontweight='bold')
    ax2.legend()
    ax2.grid(True, axis='x', alpha=0.3)
    
    # 3. Max Drawdown Comparison
    ax3 = fig.add_subplot(gs[1, 1])
    colors = ['green' if x > -0.15 else 'red' for x in comparison_df['Max DD']]
    ax3.barh(comparison_df['Strategy'], comparison_df['Max DD'] * 100, color=colors, alpha=0.7)
    ax3.axvline(x=-15, color='red', linestyle='--', alpha=0.5, label='DD = -15%')
    ax3.set_xlabel('Maximum Drawdown (%)')
    ax3.set_title('Maximum Drawdown Comparison', fontweight='bold')
    ax3.legend()
    ax3.grid(True, axis='x', alpha=0.3)
    
    # 4. Return vs Risk Scatter
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.scatter(
        comparison_df['Max DD'] * 100,
        comparison_df['Annual Return'] * 100,
        s=200,
        alpha=0.6,
        c=comparison_df['Sharpe'],
        cmap='RdYlGn'
    )
    
    for idx, row in comparison_df.iterrows():
        ax4.annotate(
            row['Strategy'],
            (row['Max DD'] * 100, row['Annual Return'] * 100),
            fontsize=7,
            ha='center'
        )
    
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax4.set_xlabel('Max Drawdown (%)')
    ax4.set_ylabel('Annualized Return (%)')
    ax4.set_title('Return vs Risk', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # 5. Number of Trades
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.bar(comparison_df['Strategy'], comparison_df['Trades'], alpha=0.7, color='steelblue')
    ax5.set_ylabel('Number of Trades')
    ax5.set_title('Trading Frequency', fontweight='bold')
    ax5.tick_params(axis='x', rotation=45)
    ax5.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('xauusd_strategy_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: xauusd_strategy_comparison.png")
    
    # Additional: Best strategy details
    best_strategy_name = comparison_df.iloc[0]['Strategy']
    best_result = results[best_strategy_name]
    
    fig2, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Price and Positions
    axes[0].plot(data.index, data['close'], label='XAUUSD Price', alpha=0.7)
    axes[0].set_ylabel('Price (USD)')
    axes[0].set_title(f'Best Strategy: {best_strategy_name}', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Position overlay
    ax0_twin = axes[0].twinx()
    position_colors = best_result.positions.apply(lambda x: 'green' if x > 0 else ('red' if x < 0 else 'gray'))
    ax0_twin.fill_between(
        best_result.positions.index,
        0,
        best_result.positions.values,
        alpha=0.3,
        color='blue'
    )
    ax0_twin.set_ylabel('Position Size')
    
    # Equity Curve
    axes[1].plot(best_result.equity_curve.index, best_result.equity_curve.values, linewidth=2)
    axes[1].fill_between(
        best_result.equity_curve.index,
        best_result.equity_curve.values,
        100000,
        alpha=0.3,
        color='green' if best_result.total_return > 0 else 'red'
    )
    axes[1].axhline(y=100000, color='black', linestyle='--', alpha=0.5, label='Initial Capital')
    axes[1].set_ylabel('Portfolio Value (USD)')
    axes[1].set_xlabel('Date')
    axes[1].set_title('Equity Curve', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('xauusd_best_strategy_detail.png', dpi=150, bbox_inches='tight')
    print("Saved: xauusd_best_strategy_detail.png")


def main(timeframe: str = "1d"):
    """Main execution flow."""
    
    print("\n" + "="*60)
    print("XAUUSD STRATEGY HUNT - FIND EDGE IN GOLD TRADING")
    print(f"Timeframe: {timeframe}")
    print("="*60)
    
    # Step 1: Data
    data = download_and_prepare_data(timeframe=timeframe)
    
    # Split train/test
    train_data, test_data = split_train_test(data, train_ratio=0.7)
    
    # Step 2: Define strategies
    configs = create_strategy_configs()
    print(f"\nTesting {len(configs)} strategies:")
    for name in configs.keys():
        print(f"  - {name}")
    
    # Step 3: Run backtests
    results = run_backtests(train_data, test_data, configs)
    
    # Step 4: Compare
    comparison_df = compare_strategies(results)
    
    # Step 5: Visualize
    visualize_results(test_data, results, comparison_df)
    
    # Step 6: Final recommendation
    print("\n" + "="*60)
    print("FINAL RECOMMENDATION")
    print("="*60)
    
    best = comparison_df.iloc[0]
    print(f"\nBest Strategy: {best['Strategy']}")
    print(f"  Sharpe Ratio: {best['Sharpe']:.2f}")
    print(f"  Annual Return: {best['Annual Return']:.2%}")
    print(f"  Max Drawdown: {best['Max DD']:.2%}")
    print(f"  Calmar Ratio: {best['Calmar']:.2f}")
    
    # Edge detection
    if best['Sharpe'] > 1.0:
        print(f"\n✓ EDGE DETECTED: Strategy shows statistical edge (Sharpe > 1.0)")
    elif best['Sharpe'] > 0.5:
        print(f"\n⚠ WEAK EDGE: Strategy shows marginal edge (Sharpe > 0.5)")
    else:
        print(f"\n✗ NO EDGE: No strategy shows consistent profitability")
    
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)


if __name__ == "__main__":
    import sys
    
    # Check if timeframe argument provided
    if len(sys.argv) > 1:
        timeframe = sys.argv[1]
    else:
        timeframe = "1d"  # Default to daily
    
    main(timeframe=timeframe)
