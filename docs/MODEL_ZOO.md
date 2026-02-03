# Model Zoo

This document catalogs all stochastic models implemented in the framework, their mathematical foundations, recommended use cases, and calibration results.

## Table of Contents
1. [Ornstein-Uhlenbeck Process](#ornstein-uhlenbeck-process)
2. [Geometric Brownian Motion](#geometric-brownian-motion)
3. [Jump-Diffusion Model](#jump-diffusion-model)
4. [GARCH Models](#garch-models)
5. [Model Selection Guidelines](#model-selection-guidelines)

---

## Ornstein-Uhlenbeck Process

### Mathematical Form
```
dX_t = κ(θ - X_t)dt + σdW_t
```

**Parameters:**
- κ (kappa): Mean reversion speed
- θ (theta): Long-term equilibrium level
- σ (sigma): Volatility parameter

**Properties:**
- Mean: E[X_t] = θ
- Variance: Var(X_t) = σ²/(2κ)
- Half-life: t₁/₂ = ln(2)/κ

### Recommended Asset Classes
- **FX Pairs**: Natural mean reversion in exchange rates
- **Interest Rates**: Short-term rates tend to revert to central bank targets
- **Pairs Trading**: Spread between cointegrated assets

### Calibration Method
Maximum Likelihood Estimation (MLE) via exact discretization:

```
X_{t+Δt} = θ + (X_t - θ)e^{-κΔt} + σ√((1-e^{-2κΔt})/(2κ)) · Z
```

### Statistical Tests Required
1. **ADF Test**: p-value < 0.05 (stationarity)
2. **Hurst Exponent**: H < 0.5 (mean reversion)
3. **KPSS Test**: Confirm stationarity

### Example Calibration Results

| Asset Pair | κ | θ | σ | Half-Life | AIC | In-Sample Sharpe |
|------------|---|---|---|-----------|-----|------------------|
| EUR/USD | 0.52 | 1.1850 | 0.0082 | 1.33 days | -1245.3 | 1.82 |
| US 2Y-10Y Spread | 0.31 | 1.45 | 0.22 | 2.24 days | -856.7 | 1.45 |

### Implementation Status
- **Status**: ✅ Implemented
- **File**: `src/models/ornstein_uhlenbeck.py`
- **Tests**: `tests/test_ou_model.py`
- **Last Updated**: 2026-02-03

---

## Geometric Brownian Motion

### Mathematical Form
```
dS_t = μS_t dt + σS_t dW_t
```

**Properties:**
- Log-normal distribution of prices
- Constant drift and volatility
- Used in Black-Scholes model

### Recommended Asset Classes
- **Equities**: Stock prices (non-mean-reverting)
- **Commodities**: Energy products

### Implementation Status
- **Status**: ⏳ Planned
- **Priority**: Medium

---

## Jump-Diffusion Model

### Mathematical Form
```
dS_t = μS_t dt + σS_t dW_t + S_t dJ_t
```

Where J_t is a compound Poisson process with intensity λ.

### Recommended Asset Classes
- **Cryptocurrencies**: Extreme moves and heavy tails
- **Event-driven**: Earnings announcements, central bank decisions

### Implementation Status
- **Status**: ⏳ Planned
- **Priority**: High (for crypto strategies)

---

## Model Selection Guidelines

### Decision Tree

```
Is data stationary (ADF test)?
├─ YES: Consider OU or GARCH
│   └─ Hurst < 0.5? → OU Process
│   └─ Volatility clustering? → GARCH
│
└─ NO: Consider GBM or Jump-Diffusion
    └─ Heavy tails / extreme events? → Jump-Diffusion
    └─ Log-normal returns? → GBM
```

### Validation Checklist

Before deploying any model:

- [ ] ADF test performed (p < 0.05 for OU)
- [ ] Hurst exponent calculated
- [ ] AIC/BIC compared across models
- [ ] Parameter stability tested (rolling windows)
- [ ] Out-of-sample validation completed
- [ ] Look-ahead bias checked
- [ ] Transaction costs included

---

## References

1. Uhlenbeck, G. E., & Ornstein, L. S. (1930). "On the Theory of Brownian Motion"
2. Merton, R. C. (1976). "Option Pricing when Underlying Stock Returns are Discontinuous"
3. Bollerslev, T. (1986). "Generalized Autoregressive Conditional Heteroskedasticity"

---

**Last Updated**: 2026-02-03  
**Maintainer**: Quant Research Team
