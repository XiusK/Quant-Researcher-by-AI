"""
Example: Mean Reversion Strategy on FX Data

This script demonstrates:
1. Data preprocessing and validation
2. OU model calibration
3. Strategy backtesting with transaction costs
4. Risk metrics calculation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.models.ornstein_uhlenbeck import OrnsteinUhlenbeck, AssetClass
from src.strategies.mean_reversion import MeanReversionStrategy
from src.base import run_stationarity_tests


def generate_synthetic_fx_data(n_days: int = 1000) -> pd.DataFrame:
    """
    Generate synthetic FX data with mean reversion properties.
    
    Returns:
        DataFrame with columns: date, close
    """
    np.random.seed(42)
    
    # True OU parameters
    kappa = 0.5
    theta = 1.20  # EUR/USD around 1.20
    sigma = 0.008
    
    dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')
    prices = np.zeros(n_days)
    prices[0] = theta
    
    dt = 1.0
    for i in range(1, n_days):
        dW = np.random.randn()
        prices[i] = theta + (prices[i-1] - theta) * np.exp(-kappa * dt) + \
                   sigma * np.sqrt((1 - np.exp(-2 * kappa * dt)) / (2 * kappa)) * dW
    
    df = pd.DataFrame({
        'date': dates,
        'close': prices
    })
    df.set_index('date', inplace=True)
    
    return df


def main():
    """Run complete mean reversion strategy example."""
    
    print("=" * 60)
    print("Mean Reversion Strategy Example")
    print("=" * 60)
    
    # Step 1: Generate data
    print("\n[1] Generating synthetic FX data...")
    data = generate_synthetic_fx_data(n_days=1000)
    print(f"Generated {len(data)} days of data")
    print(f"Price range: [{data['close'].min():.4f}, {data['close'].max():.4f}]")
    
    # Step 2: Statistical validation
    print("\n[2] Running stationarity tests...")
    tests = run_stationarity_tests(data['close'])
    print(f"ADF Statistic: {tests['adf_statistic']:.4f}")
    print(f"ADF p-value: {tests['adf_pvalue']:.4f}")
    print(f"Is Stationary: {tests['is_stationary']}")
    print(f"Hurst Exponent: {tests['hurst_exponent']:.4f}")
    print(f"Is Mean Reverting: {tests['is_mean_reverting']}")
    
    if not tests['is_stationary']:
        print("\nWARNING: Data is not stationary! OU model may not be appropriate.")
        return
    
    # Step 3: Calibrate OU model
    print("\n[3] Calibrating Ornstein-Uhlenbeck model...")
    model = OrnsteinUhlenbeck(asset_class=AssetClass.FX)
    params = model.calibrate(data['close'])
    
    print(f"Calibrated Parameters:")
    print(f"  kappa (mean reversion): {params.params['kappa']:.4f}")
    print(f"  theta (long-term mean): {params.params['theta']:.4f}")
    print(f"  sigma (volatility): {params.params['sigma']:.4f}")
    print(f"  Half-life: {model.half_life():.2f} days")
    print(f"  AIC: {params.aic:.2f}")
    print(f"  BIC: {params.bic:.2f}")
    
    # Step 4: Create strategy
    print("\n[4] Initializing mean reversion strategy...")
    strategy_config = {
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
    
    strategy = MeanReversionStrategy(name="OU_MR_Example", config=strategy_config)
    
    # Step 5: Backtest
    print("\n[5] Running backtest...")
    portfolio_value = 100000.0
    positions = []
    pnl_series = []
    
    # Split: 70% in-sample, 30% out-of-sample
    split_idx = int(len(data) * 0.7)
    in_sample = data.iloc[:split_idx]
    out_sample = data.iloc[split_idx:]
    
    print(f"In-sample period: {in_sample.index[0]} to {in_sample.index[-1]}")
    print(f"Out-of-sample period: {out_sample.index[0]} to {out_sample.index[-1]}")
    
    # Backtest on out-of-sample data
    for timestamp in out_sample.index:
        available_data = data[data.index <= timestamp]
        
        if len(available_data) < strategy_config['lookback_period']:
            continue
        
        # Generate signal
        signal = strategy.generate_signal(available_data, timestamp)
        
        # Calculate position size
        current_vol = available_data['close'].pct_change().std() * np.sqrt(252)
        position_size = strategy.calculate_position_size(signal, portfolio_value, current_vol)
        
        positions.append({
            'date': timestamp,
            'signal': signal.direction,
            'z_score': signal.metadata.get('z_score', 0),
            'position': position_size
        })
    
    positions_df = pd.DataFrame(positions).set_index('date')
    
    print(f"\nGenerated {len(positions_df)} trading signals")
    print(f"Long signals: {(positions_df['signal'] == 1).sum()}")
    print(f"Short signals: {(positions_df['signal'] == -1).sum()}")
    print(f"Neutral: {(positions_df['signal'] == 0).sum()}")
    
    # Step 6: Calculate performance
    print("\n[6] Calculating risk metrics...")
    
    # Simple returns calculation (simplified)
    returns = out_sample['close'].pct_change().dropna()
    
    if len(returns) > 0:
        metrics = strategy.calculate_risk_metrics(returns)
        
        print(f"\nRisk-Adjusted Performance:")
        print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        print(f"  Sortino Ratio: {metrics.sortino_ratio:.2f}")
        print(f"  Maximum Drawdown: {metrics.max_drawdown:.2%}")
        print(f"  VaR (95%): {metrics.var_95:.2%}")
        print(f"  CVaR (95%): {metrics.cvar_95:.2%}")
        print(f"  Calmar Ratio: {metrics.calmar_ratio:.2f}")
    
    # Step 7: Visualization
    print("\n[7] Generating plots...")
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot 1: Price and theta
    axes[0].plot(data.index, data['close'], label='Price', alpha=0.7)
    axes[0].axhline(y=params.params['theta'], color='r', linestyle='--', label='Theta (Mean)')
    axes[0].set_ylabel('Price')
    axes[0].set_title('FX Price and Long-Term Mean')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Z-scores
    if len(positions_df) > 0:
        axes[1].plot(positions_df.index, positions_df['z_score'], label='Z-Score')
        axes[1].axhline(y=strategy_config['entry_threshold'], color='r', linestyle='--', label='Entry')
        axes[1].axhline(y=-strategy_config['entry_threshold'], color='r', linestyle='--')
        axes[1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
        axes[1].set_ylabel('Z-Score')
        axes[1].set_title('Mean Reversion Z-Scores')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Positions
    if len(positions_df) > 0:
        axes[2].plot(positions_df.index, positions_df['position'], label='Position Size')
        axes[2].axhline(y=0, color='k', linestyle='-', alpha=0.3)
        axes[2].set_ylabel('Position')
        axes[2].set_xlabel('Date')
        axes[2].set_title('Trading Positions')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mean_reversion_example.png', dpi=150)
    print("Saved plot to: mean_reversion_example.png")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
