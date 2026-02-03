"""
Advanced Strategy Testing Suite for XAUUSD

Tests additional models:
1. Jump-Diffusion (event-driven)
2. Regime-Switching HMM
3. Microstructure (intraday)

Run: python test_advanced_strategies.py [timeframe]
Examples:
    python test_advanced_strategies.py 1d   # Daily with Jump-Diffusion + HMM
    python test_advanced_strategies.py 1h   # Hourly with Microstructure
    python test_advanced_strategies.py 15m  # 15-min with Microstructure
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from src.data import load_xauusd_from_kaggle, calculate_features, split_train_test
from src.models.jump_diffusion import JumpDiffusionModel
from src.models.regime_hmm import MarketRegimeHMM
from src.strategies.microstructure import MicrostructureStrategy
from src.backtest import BacktestEngine
from src.base import AssetClass


def test_jump_diffusion(data: pd.DataFrame):
    """Test Jump-Diffusion model for event detection."""
    print("\n" + "="*60)
    print("Testing Jump-Diffusion Model")
    print("="*60)
    
    # Calibrate model
    model = JumpDiffusionModel(asset_class=AssetClass.COMMODITIES)
    
    try:
        params = model.calibrate(data['close'])
        
        print("\nCalibrated Parameters:")
        print(f"  Drift (mu): {params.params['mu']:.4f}")
        print(f"  Diffusion Vol (sigma): {params.params['sigma']:.4f}")
        print(f"  Jump Intensity (lambda): {params.params['lambda_jump']:.2f} jumps/year")
        print(f"  Jump Mean (mu_J): {params.params['mu_jump']:.4f}")
        print(f"  Jump Vol (sigma_J): {params.params['sigma_jump']:.4f}")
        print(f"\nExpected Jump Size: {model.expected_jump_size():.2%}")
        print(f"AIC: {params.aic:.2f}")
        print(f"BIC: {params.bic:.2f}")
        
        # Simulate paths
        print("\nSimulating 100 paths...")
        S0 = data['close'].iloc[-1]
        paths = model.simulate(S0=S0, T=1.0, n_steps=252, n_paths=100, seed=42)
        
        # Plot simulation
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i in range(min(20, len(paths))):
            ax.plot(paths[i], alpha=0.3, color='blue')
        
        ax.plot(paths.mean(axis=0), color='red', linewidth=2, label='Mean Path')
        ax.axhline(y=S0, color='black', linestyle='--', alpha=0.5, label='Current Price')
        ax.set_title('Jump-Diffusion Simulation (1 Year Forward)')
        ax.set_xlabel('Trading Days')
        ax.set_ylabel('Gold Price (USD)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('jump_diffusion_simulation.png', dpi=150)
        print("Saved: jump_diffusion_simulation.png")
        
        return True
        
    except ValueError as e:
        print(f"\nJump-Diffusion Test Failed: {e}")
        print("This data may not exhibit sufficient jump behavior.")
        return False


def test_regime_hmm(data: pd.DataFrame):
    """Test HMM regime detection."""
    print("\n" + "="*60)
    print("Testing Hidden Markov Model - Regime Detection")
    print("="*60)
    
    # Fit HMM
    hmm_model = MarketRegimeHMM(n_regimes=3)
    
    try:
        hmm_model.fit(data, features=['returns', 'realized_vol', 'hl_range'])
        
        # Get regime statistics
        regime_stats = hmm_model.get_regime_statistics(data)
        print("\nRegime Statistics:")
        print(regime_stats.to_string())
        
        # Predict regimes
        regimes = hmm_model.predict_regime(data, features=['returns', 'realized_vol', 'hl_range'])
        regime_probs = hmm_model.predict_proba(data, features=['returns', 'realized_vol', 'hl_range'])
        
        # Visualize regimes
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        # Price with regime coloring
        regime_colors = {0: 'green', 1: 'blue', 2: 'red'}
        
        for regime_id in range(3):
            mask = regimes == regime_id
            regime_data = data.loc[data[['returns', 'realized_vol', 'hl_range']].dropna().index][mask]
            
            if len(regime_data) > 0:
                regime_name = hmm_model.regime_names.get(regime_id, f"Regime_{regime_id}")
                axes[0].scatter(
                    regime_data.index,
                    regime_data['close'],
                    c=regime_colors[regime_id],
                    s=10,
                    alpha=0.6,
                    label=regime_name
                )
        
        axes[0].set_ylabel('Gold Price')
        axes[0].set_title('Market Regimes Detected by HMM')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Regime probabilities
        valid_index = data[['returns', 'realized_vol', 'hl_range']].dropna().index
        for regime_id in range(3):
            regime_name = hmm_model.regime_names.get(regime_id, f"Regime_{regime_id}")
            axes[1].plot(
                valid_index,
                regime_probs[:, regime_id],
                label=regime_name,
                alpha=0.7
            )
        
        axes[1].set_ylabel('Probability')
        axes[1].set_title('Regime Probabilities Over Time')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Volatility by regime
        axes[2].plot(valid_index, data.loc[valid_index, 'realized_vol'], label='Realized Volatility', alpha=0.7)
        axes[2].set_ylabel('Volatility')
        axes[2].set_xlabel('Date')
        axes[2].set_title('Volatility Over Time')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('regime_hmm_analysis.png', dpi=150)
        print("\nSaved: regime_hmm_analysis.png")
        
        return True, hmm_model
        
    except Exception as e:
        print(f"\nHMM Test Failed: {e}")
        return False, None


def test_microstructure_strategy(data: pd.DataFrame, timeframe: str):
    """Test microstructure strategy on intraday data."""
    print("\n" + "="*60)
    print(f"Testing Microstructure Strategy ({timeframe} data)")
    print("="*60)
    
    # Configure based on timeframe
    periods_per_day_map = {
        '5m': 288,   # 24h * 60 / 5
        '15m': 96,   # 24h * 60 / 15
        '30m': 48,
        '1h': 24,
        '4h': 6
    }
    
    periods_per_day = periods_per_day_map.get(timeframe, 24)
    
    config = {
        'volume_imbalance_threshold': 2.0,
        'price_impact_window': 10,
        'session_filter': 'us',  # Trade US session only
        'max_position_pct': 0.10,
        'max_leverage': 1.0,
        'spread_bps': 8.0,  # Higher for intraday
        'commission_bps': 2.0,
        'periods_per_day': periods_per_day
    }
    
    strategy = MicrostructureStrategy(
        name=f"Microstructure_{timeframe}",
        config=config
    )
    
    # Split data
    train_data, test_data = split_train_test(data, train_ratio=0.7)
    
    # Backtest
    engine = BacktestEngine(
        initial_capital=100000.0,
        spread_bps=8.0,
        commission_bps=2.0,
        slippage_bps=3.0
    )
    
    result = engine.run(
        strategy=strategy,
        data=test_data,
        warmup_period=100
    )
    
    print(f"\nMicrostructure Strategy Results:")
    print(f"  Sharpe Ratio: {result.risk_metrics.sharpe_ratio:.2f}")
    print(f"  Total Return: {result.total_return:.2%}")
    print(f"  Max Drawdown: {result.risk_metrics.max_drawdown:.2%}")
    print(f"  Number of Trades: {result.metadata['n_trades']}")
    
    return result


def main(timeframe: str = "1d"):
    """Main execution."""
    print("\n" + "="*60)
    print("ADVANCED STRATEGY TESTING FOR XAUUSD")
    print(f"Timeframe: {timeframe}")
    print("="*60)
    
    # Load data
    data = load_xauusd_from_kaggle(timeframe=timeframe)
    data = calculate_features(data)
    
    # Test 1: Jump-Diffusion (for all timeframes)
    if timeframe in ['1d', '1w', '1M']:
        test_jump_diffusion(data)
    
    # Test 2: HMM Regime Detection (for daily/weekly)
    if timeframe in ['1d', '1w']:
        success, hmm_model = test_regime_hmm(data)
    
    # Test 3: Microstructure Strategy (for intraday)
    if timeframe in ['5m', '15m', '30m', '1h', '4h']:
        test_microstructure_strategy(data, timeframe)
    
    print("\n" + "="*60)
    print("Advanced Testing Complete!")
    print("="*60)
    
    print("\nRecommendations:")
    print("1. For Daily Data: Use Jump-Diffusion + HMM-based regime switching")
    print("2. For Intraday Data: Use Microstructure strategy during US session")
    print("3. Combine regime detection with existing strategies for better timing")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        timeframe = sys.argv[1]
    else:
        timeframe = "1d"
    
    main(timeframe=timeframe)
