# Quant Researcher & Financial Engineer - Project Structure

This document describes the organized folder structure of the quantitative trading framework.

---

## 📁 Project Structure

```
Quant Researcher By AI/
│
├── 📂 src/                          # Core source code
│   ├── base.py                      # Abstract base classes (BaseStochasticModel, BaseStrategy, BaseRiskManager)
│   ├── models/                      # Stochastic models
│   │   ├── ornstein_uhlenbeck.py   # OU process for mean reversion
│   │   ├── jump_diffusion.py       # Merton jump-diffusion model
│   │   └── regime_hmm.py            # Hidden Markov Model for regimes
│   ├── strategies/                  # Trading strategies
│   │   ├── mean_reversion.py       # OU-based mean reversion
│   │   ├── momentum.py              # MA crossover (fast/slow)
│   │   ├── volatility_breakout.py  # Bollinger Bands & ATR channels
│   │   ├── range_trading.py         # Support/resistance trading
│   │   └── microstructure.py        # Intraday order flow
│   ├── risk/                        # Risk management
│   │   └── portfolio_risk.py        # VaR, CVaR, position limits
│   ├── data/                        # Data loading & preprocessing
│   │   └── data_loader.py           # Kaggle XAUUSD CSV loader
│   └── backtest/                    # Backtesting engine
│       └── engine.py                # Event-driven backtester
│
├── 📂 tests/                        # Test scripts & strategy experiments
│   ├── test_kaggle_data.py         # Test data loading from Kaggle
│   ├── test_advanced_strategies.py # Test Jump-Diffusion, HMM, Microstructure
│   └── xauusd_strategy_hunt.py     # Main strategy comparison script (6 strategies)
│
├── 📂 examples/                     # Educational guides & demonstrations
│   ├── volatility_breakout_guide.py       # Deep dive into winning strategy
│   └── channel_comparison_guide.py        # Compare BB vs ATR vs Keltner vs Donchian
│
├── 📂 deployment/                   # Production-ready code for live trading
│   ├── tradingview/
│   │   └── volatility_breakout_atr.pine   # TradingView Pine Script indicator
│   ├── metatrader5/
│   │   └── VolatilityBreakoutATR.mq5      # MT5 Expert Advisor (EA)
│   ├── INSTALLATION_GUIDE.md              # How to install on TV & MT5
│   └── DEPLOYMENT_CHECKLIST.md            # Pre-live trading checklist
│
├── 📂 results/                      # Output files from backtests & analysis
│   ├── strategy_backtests/
│   │   ├── xauusd_strategy_comparison.png      # 6 strategies comparison
│   │   └── xauusd_best_strategy_detail.png     # Best strategy equity curve
│   └── analysis/
│       ├── volatility_breakout_detailed_analysis.png  # ATR channel signals
│       └── channel_types_comparison.png              # 4 channel types visualized
│
├── 📂 configs/                      # YAML configuration files
│   ├── strategy_config.yaml         # Strategy parameters
│   ├── model_config.yaml            # Model parameters (OU, Jump-Diffusion, HMM)
│   └── xauusd_strategies.yaml       # XAUUSD-specific strategy configs
│
├── 📂 docs/                         # Documentation & architecture decisions
│   ├── MODEL_ZOO.md                 # Catalog of all implemented models
│   └── adr/                         # Architecture Decision Records
│       └── 001-ou-process-for-fx.md # Why use OU for FX mean reversion
│
├── 📂 Kaggles_XAUUSD_Data/          # Historical data (9 timeframes)
│   ├── XAU_1M_data (1).csv          # Monthly bars
│   ├── XAU_1w_data (1).csv          # Weekly bars
│   ├── XAU_1d_data (1).csv          # Daily bars (main)
│   ├── XAU_4h_data (1).csv          # 4-hour bars
│   ├── XAU_1h_data (1).csv          # Hourly bars
│   ├── XAU_30m_data (1).csv         # 30-minute bars
│   ├── XAU_15m_data (1).csv         # 15-minute bars
│   ├── XAU_5m_data (1).csv          # 5-minute bars
│   └── XAU_1m_data (1).csv          # 1-minute bars
│
├── 📄 README.md                     # Main project documentation
├── 📄 README_XAUUSD.md              # XAUUSD-specific guide
├── 📄 requirements.txt              # Core Python dependencies
├── 📄 requirements_advanced.txt     # Advanced packages (hmmlearn, arch)
└── 📄 .gitignore                    # Git ignore rules

```

---

## 🎯 Key Files by Purpose

### For Learning & Understanding:
1. **[examples/volatility_breakout_guide.py](examples/volatility_breakout_guide.py)** - Understand the winning strategy
2. **[examples/channel_comparison_guide.py](examples/channel_comparison_guide.py)** - Compare different channel types
3. **[docs/MODEL_ZOO.md](docs/MODEL_ZOO.md)** - Overview of all models

### For Testing & Development:
1. **[tests/xauusd_strategy_hunt.py](tests/xauusd_strategy_hunt.py)** - Run full strategy comparison
2. **[tests/test_kaggle_data.py](tests/test_kaggle_data.py)** - Verify data loading
3. **[tests/test_advanced_strategies.py](tests/test_advanced_strategies.py)** - Test Jump-Diffusion & HMM

### For Live Trading:
1. **[deployment/tradingview/volatility_breakout_atr.pine](deployment/tradingview/volatility_breakout_atr.pine)** - TradingView indicator
2. **[deployment/metatrader5/VolatilityBreakoutATR.mq5](deployment/metatrader5/VolatilityBreakoutATR.mq5)** - MT5 Expert Advisor
3. **[deployment/INSTALLATION_GUIDE.md](deployment/INSTALLATION_GUIDE.md)** - Installation instructions
4. **[deployment/DEPLOYMENT_CHECKLIST.md](deployment/DEPLOYMENT_CHECKLIST.md)** - Pre-live checklist

### For Analysis:
1. **[results/strategy_backtests/](results/strategy_backtests/)** - Strategy comparison charts
2. **[results/analysis/](results/analysis/)** - Technical analysis visualizations

---

## 🚀 Quick Start Guide

### 1. Run Strategy Backtest
```powershell
python tests/xauusd_strategy_hunt.py
```
Output: PNG charts in `results/strategy_backtests/`

### 2. Understand the Winning Strategy
```powershell
python examples/volatility_breakout_guide.py
```
Output: Detailed analysis in `results/analysis/`

### 3. Compare Channel Types
```powershell
python examples/channel_comparison_guide.py
```
Output: 4 channel types comparison chart

### 4. Deploy to TradingView
1. Open [deployment/tradingview/volatility_breakout_atr.pine](deployment/tradingview/volatility_breakout_atr.pine)
2. Follow [deployment/INSTALLATION_GUIDE.md](deployment/INSTALLATION_GUIDE.md)

### 5. Deploy to MetaTrader 5
1. Open [deployment/metatrader5/VolatilityBreakoutATR.mq5](deployment/metatrader5/VolatilityBreakoutATR.mq5)
2. Compile in MetaEditor
3. Follow [deployment/INSTALLATION_GUIDE.md](deployment/INSTALLATION_GUIDE.md)

---

## 📊 Data Structure

### Kaggle XAUUSD Data Format
```
Date;Time;Open;High;Low;Close;Volume
2004.06.11;00:00;389.40;395.95;386.70;392.80;0
```
- Delimiter: Semicolon (`;`)
- Date Format: `YYYY.MM.DD HH:MM`
- Timeframes: 9 different resolutions (1M to 1m)
- Date Range: 2004-06-11 to 2025-12-01 (22 years)

---

## 🔧 Configuration

All strategy parameters are in `configs/`:
- **strategy_config.yaml**: Base strategy settings
- **model_config.yaml**: Stochastic model parameters
- **xauusd_strategies.yaml**: XAUUSD-specific configs

Example (Volatility Breakout ATR):
```yaml
volatility_breakout_atr:
  channel_length: 20        # MA period
  atr_period: 14            # ATR period
  atr_multiplier: 2.5       # Channel width
  min_volatility: 0.12      # 12% annual vol filter
  target_volatility: 0.15   # 15% target portfolio vol
```

---

## 🧪 Testing Workflow

1. **Data Verification**: Run `tests/test_kaggle_data.py`
2. **Strategy Hunt**: Run `tests/xauusd_strategy_hunt.py`
3. **Advanced Models**: Run `tests/test_advanced_strategies.py`
4. **Analysis**: Review charts in `results/`
5. **Deployment**: Follow `deployment/INSTALLATION_GUIDE.md`

---

## 📈 Performance Results

### Best Strategy: Volatility Breakout ATR
- **File**: [src/strategies/volatility_breakout.py](src/strategies/volatility_breakout.py)
- **Test Period**: 2019-08-05 to 2025-12-01 (1,579 days)
- **Sharpe Ratio**: 0.37
- **Total Return**: +6.34%
- **Max Drawdown**: -3.53%
- **Trades**: 365

### All Strategies Tested:
1. Volatility_Breakout_ATR (Winner)
2. Momentum_Fast
3. Volatility_Breakout_BB
4. Momentum_Slow
5. Range_Trading
6. Mean_Reversion

See full results: [results/strategy_backtests/xauusd_strategy_comparison.png](results/strategy_backtests/xauusd_strategy_comparison.png)

---

## 🛠️ Development Workflow

### Adding New Strategy:
1. Create file in `src/strategies/`
2. Inherit from `BaseStrategy` in `src/base.py`
3. Implement `generate_signal()` method
4. Add config to `configs/xauusd_strategies.yaml`
5. Test in `tests/xauusd_strategy_hunt.py`

### Adding New Model:
1. Create file in `src/models/`
2. Inherit from `BaseStochasticModel` in `src/base.py`
3. Implement `calibrate()` and `simulate()` methods
4. Document in `docs/MODEL_ZOO.md`
5. Create ADR in `docs/adr/`

---

## 📚 Resources

### Internal Documentation:
- [README.md](README.md) - Main project overview
- [README_XAUUSD.md](README_XAUUSD.md) - XAUUSD strategy guide
- [docs/MODEL_ZOO.md](docs/MODEL_ZOO.md) - All models catalog
- [deployment/INSTALLATION_GUIDE.md](deployment/INSTALLATION_GUIDE.md) - Deployment guide

### External References:
- Ornstein-Uhlenbeck Process: Mean reversion modeling
- ATR (Average True Range): Volatility measurement
- Bollinger Bands: Statistical channels
- Turtle Trading: Donchian breakout system

---

## 🚨 Important Notes

### Before Live Trading:
1. ✅ Read [deployment/DEPLOYMENT_CHECKLIST.md](deployment/DEPLOYMENT_CHECKLIST.md)
2. ✅ Backtest minimum 3 years of data
3. ✅ Paper trade 1-2 months
4. ✅ Start with 1% risk per trade
5. ✅ Monitor daily for first month

### Risk Warnings:
- Strategy Sharpe: 0.37 (modest returns)
- Not suitable for aggressive trading
- Transaction costs significant
- Requires trending markets
- Past performance ≠ future results

---

## 🤝 Contributing

To maintain code quality:
1. Follow type hints (Python 3.9+)
2. Document all functions with docstrings
3. Include mathematical formulas in comments
4. Add unit tests in `tests/`
5. Update `docs/MODEL_ZOO.md` for new models
6. Create ADR for major architectural decisions

---

## 📞 Support

For questions or issues:
1. Check [deployment/INSTALLATION_GUIDE.md](deployment/INSTALLATION_GUIDE.md)
2. Review example scripts in `examples/`
3. Check error logs in MT5/TradingView
4. Verify data loading with `tests/test_kaggle_data.py`

---

**Last Updated**: February 3, 2026  
**Framework Version**: 1.0  
**Python Version**: 3.14.2  
**Best Strategy**: Volatility Breakout ATR (Sharpe 0.37)
