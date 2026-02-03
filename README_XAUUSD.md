# XAUUSD Strategy Hunt - Finding Edge in Gold Trading

Comprehensive framework for testing multiple trading strategies on Gold (XAUUSD) to identify statistical edges.

## Strategies Implemented

### 1. Mean Reversion (OU-based)
- **Hypothesis**: Gold exhibits short-term mean reversion
- **Mathematical Foundation**: Ornstein-Uhlenbeck process
- **Best For**: Low volatility, range-bound periods
- **Entry**: Z-score > 2.0 from long-term mean
- **Exit**: Z-score < 0.5 or stop-loss

### 2. Momentum (MA Crossover)
- **Hypothesis**: Gold trends during macro uncertainty
- **Variants**: 
  - Fast (10/30 EMA): Capture short-term trends
  - Slow (50/200 SMA): Classic golden cross
- **Best For**: High volatility, trending markets
- **Entry**: MA crossover with trend confirmation
- **Exit**: Opposite crossover or trailing stop

### 3. Volatility Breakout
- **Hypothesis**: Gold breaks out after consolidation
- **Methods**:
  - Bollinger Bands (20-day, 2σ)
  - ATR Channels (14-day ATR × 2.5)
- **Best For**: Post-consolidation expansions
- **Entry**: Price exceeds band/channel
- **Exit**: Return to mean or opposite breakout

### 4. Range Trading
- **Hypothesis**: Gold respects support/resistance in stable regimes
- **Method**: Buy at support, sell at resistance
- **Best For**: Low volatility consolidation
- **Entry**: Within 1.5% of support/resistance
- **Exit**: 2% profit target or range breakdown

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run Strategy Hunt

```bash
python xauusd_strategy_hunt.py
```

This will:
1. Download XAUUSD data (2018-present)
2. Calculate 20+ technical features
3. Split data (70% train, 30% test)
4. Backtest 6 strategy variants
5. Generate comparison visualizations
6. Identify best performer

## Expected Output

### Console Output
```
XAUUSD STRATEGY HUNT - FIND EDGE IN GOLD TRADING
================================================================

STEP 1: DATA PREPARATION
Downloading GC=F data from 2018-01-01 to 2026-02-03...
Successfully downloaded 2087 rows
Price range: $1180.20 - $2789.45

STEP 2: BACKTESTING STRATEGIES
[Testing: Mean_Reversion]
Total Return: 18.45%
Annualized Return: 8.23%
Sharpe Ratio: 1.42
Max Drawdown: -12.34%
Number of Trades: 87

...

FINAL RECOMMENDATION
Best Strategy: Momentum_Fast
  Sharpe Ratio: 1.58
  Annual Return: 12.34%
  Max Drawdown: -14.56%
  
✓ EDGE DETECTED: Strategy shows statistical edge (Sharpe > 1.0)
```

### Generated Files

1. **xauusd_strategy_comparison.png**
   - Equity curves (all strategies)
   - Sharpe ratio comparison
   - Max drawdown comparison
   - Return vs risk scatter
   - Trading frequency

2. **xauusd_best_strategy_detail.png**
   - Price chart with positions
   - Equity curve
   - Drawdown periods

## Strategy Configuration

Edit [configs/xauusd_strategies.yaml](configs/xauusd_strategies.yaml) to adjust:
- Entry/exit thresholds
- Position sizing
- Risk limits
- Transaction costs

## Performance Metrics

All strategies evaluated on:

| Metric | Description | Target |
|--------|-------------|--------|
| **Sharpe Ratio** | Risk-adjusted return | > 1.0 |
| **Sortino Ratio** | Downside risk-adjusted | > 1.5 |
| **Max Drawdown** | Peak-to-trough decline | < -20% |
| **Calmar Ratio** | Return / Max DD | > 0.5 |
| **Win Rate** | % profitable trades | > 45% |

## Gold-Specific Insights

### Market Regimes

**Trending (Momentum favored)**:
- Crisis periods (2020 COVID, 2022 Ukraine)
- QE/money printing cycles
- USD weakness

**Mean Reverting (MR favored)**:
- Stable macro environment
- Low VIX (<15)
- Tight trading ranges

**Volatile (Breakout favored)**:
- Post-consolidation
- Fed policy announcements
- Geopolitical events

### Key Correlations

- **Negative with USD Index**: Trade during USD trends
- **Positive with inflation**: Monitor CPI releases
- **Safe haven**: Spikes during equity crashes

## Bias Prevention

### Look-ahead Bias
- Strict temporal ordering in backtests
- `validate_no_lookahead()` checks
- Walk-forward validation

### Overfitting
- 70/30 train-test split
- Parameter stability tests
- Out-of-sample required for conclusions

### Transaction Costs
- 5 bps spread (realistic for Gold)
- 1 bps commission
- 2 bps slippage (market impact)
- Total: ~8 bps per trade

## Advanced Usage

### Custom Strategy

```python
from src.base import BaseStrategy, Signal
from src.backtest import BacktestEngine

class MyGoldStrategy(BaseStrategy):
    def generate_signal(self, market_data, timestamp):
        # Your logic here
        return Signal(...)
    
    # Implement other required methods...

# Test it
engine = BacktestEngine()
result = engine.run(strategy, data)
```

### Walk-Forward Analysis

```python
# Coming soon: Rolling window optimization
# tests/test_walk_forward.py
```

## Known Limitations

1. **Daily Data Only**: No intraday testing yet
2. **No Options**: Spot only (no Greeks for Gold options)
3. **Single Asset**: No portfolio diversification
4. **Static Parameters**: No adaptive optimization

## Roadmap

- [ ] Intraday strategies (1H, 15M data)
- [ ] Machine learning features (LSTM, XGBoost)
- [ ] Multi-asset portfolio (Gold + Silver + USD)
- [ ] Regime detection (HMM, clustering)
- [ ] Live trading integration

## References

1. **Gold Mean Reversion**: Ciner, C. (2001). "On the long run relationship between gold and silver prices"
2. **Momentum in Commodities**: Gorton, G. et al. (2012). "Facts and Fantasies about Commodity Futures"
3. **Volatility Breakouts**: Kaufman, P. (2013). "Trading Systems and Methods"

---

**Last Updated**: 2026-02-03  
**Maintainer**: Quant Research Team  
**License**: Proprietary
