# Quantitative Trading Framework

A rigorous quantitative research framework for systematic trading strategies, emphasizing mathematical correctness, strict risk management, and bias prevention.

## Philosophy

**Mathematical Rigor over Intuition**

This framework enforces:
- Statistical validation of all hypotheses (ADF, Hurst, AIC/BIC)
- Explicit stochastic modeling with calibrated parameters
- Out-of-sample testing and walk-forward analysis
- Comprehensive transaction cost modeling
- Automated detection of look-ahead bias

## Project Structure

```
src/
├── base.py                    # Abstract base classes
├── models/                    # Stochastic processes
│   ├── ornstein_uhlenbeck.py # Mean reversion (FX, rates)
│   ├── gbm.py                # Geometric Brownian Motion
│   └── jump_diffusion.py    # Heavy tails (crypto)
├── strategies/               # Alpha generation
│   ├── mean_reversion.py    # OU-based strategy
│   └── momentum.py          # Trend following
└── risk/                     # Risk management
    └── portfolio_risk.py    # VaR, Greeks, limits

configs/
├── strategy_config.yaml      # Strategy parameters
└── model_config.yaml        # Model calibration settings

docs/
├── MODEL_ZOO.md             # Model catalog
└── adr/                     # Architecture Decision Records
    └── 001-ou-process-for-fx.md
```

## Core Principles

### 1. Hypothesis-Driven Research
Every strategy must articulate:
- The market inefficiency being exploited
- Mathematical foundation (stochastic process)
- Statistical evidence (p-values, AIC/BIC)

### 2. Bias Guardrails
Automated checks for:
- Look-ahead bias (using future data)
- Survivorship bias (delisted stocks)
- Overfitting (walk-forward validation)

### 3. Transaction Cost Reality
All backtests include:
- Bid-ask spread
- Market impact (slippage)
- Commission fees

### 4. Risk-First Mentality
Optimization targets:
- Sharpe Ratio (not net profit)
- Maximum Drawdown limits
- VaR constraints

## Quick Start

### Installation

```bash
# Clone repository
git clone <repo_url>
cd quant-research

# Install dependencies
pip install -r requirements.txt
```

### Example: Mean Reversion Strategy

```python
from src.models.ornstein_uhlenbeck import OrnsteinUhlenbeck, AssetClass
from src.strategies.mean_reversion import MeanReversionStrategy
import pandas as pd

# Load data
data = pd.read_csv('eurusd_daily.csv', index_col='date', parse_dates=True)

# Calibrate OU model
model = OrnsteinUhlenbeck(asset_class=AssetClass.FX)
params = model.calibrate(data['close'])

print(f"Mean reversion speed: {params.params['kappa']:.3f}")
print(f"Half-life: {model.half_life():.2f} days")

# Create strategy
config = {
    'entry_threshold': 2.0,
    'exit_threshold': 0.5,
    'lookback_period': 252,
    'max_position_pct': 0.2
}

strategy = MeanReversionStrategy(name="EUR/USD MR", config=config)

# Generate signal
signal = strategy.generate_signal(
    market_data=data,
    timestamp=data.index[-1]
)

print(f"Signal: {signal.direction}, Confidence: {signal.confidence:.2f}")
```

## Stochastic Model Selection

| Asset Class | Primary Model | Rationale |
|-------------|---------------|-----------|
| FX / Rates | Ornstein-Uhlenbeck | Mean reversion property |
| Equities | GBM / GARCH | Log-normal returns, vol clustering |
| Crypto | Jump-Diffusion | Heavy tails, extreme events |
| Macro | Hidden Markov | Regime transitions |

See [MODEL_ZOO.md](docs/MODEL_ZOO.md) for detailed specifications.

## Configuration Management

All parameters must be defined in YAML files (no hardcoding):

**configs/strategy_config.yaml**:
```yaml
strategy:
  name: "OU_Mean_Reversion"
  entry_threshold: 2.0
  max_position_pct: 0.20
  
risk:
  max_drawdown: 0.15
  var_limit: 0.05
```

## Testing Requirements

Before deploying any strategy:

1. **Statistical Validation**
   - [ ] ADF test (p < 0.05 for stationary models)
   - [ ] Hurst exponent calculated
   - [ ] Parameter stability across rolling windows

2. **Backtesting**
   - [ ] In-sample vs out-of-sample results separated
   - [ ] Walk-forward analysis completed
   - [ ] Transaction costs included

3. **Risk Checks**
   - [ ] Maximum drawdown < 20%
   - [ ] Sharpe ratio > 1.0 (out-of-sample)
   - [ ] VaR limits enforced

## Documentation Standards

Every model requires:
1. Entry in `MODEL_ZOO.md` with calibration results
2. ADR (Architecture Decision Record) explaining choice
3. Type hints and docstrings with mathematical formulas

## Performance Metrics

We optimize for **risk-adjusted returns**:

- **Primary**: Sharpe Ratio
- **Secondary**: Sortino Ratio (downside risk)
- **Constraint**: Maximum Drawdown < threshold

*Net profit without risk adjustment is meaningless.*

## Contributing

1. All code must pass `mypy` type checking
2. Use vectorized NumPy operations (no loops)
3. Write ADR for significant decisions
4. ASCII-only (no emojis in code)

## References

1. Shreve, S. (2004). *Stochastic Calculus for Finance II*
2. Chan, E. (2009). *Quantitative Trading*
3. Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*

---

**Last Updated**: 2026-02-03  
**License**: Proprietary  
**Contact**: quant-research@example.com
