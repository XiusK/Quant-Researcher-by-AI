# Deployment Checklist - Volatility Breakout ATR

Use this checklist before deploying the strategy to live trading.

---

## 📋 Pre-Deployment Testing

### 1. Backtest Verification
- [ ] Historical data covers minimum 3 years (2019-2024)
- [ ] Out-of-sample period shows positive Sharpe (> 0.2)
- [ ] Maximum drawdown within acceptable limits (< 10%)
- [ ] Transaction costs properly modeled (spread + commission + slippage)
- [ ] No look-ahead bias (only using data available at entry time)

**Expected Results (XAUUSD Daily):**
```
Sharpe Ratio: 0.35 - 0.40
Annual Return: 1% - 3%
Max Drawdown: 3% - 5%
Win Rate: 45% - 55%
Trades/Year: ~50-70
```

### 2. Parameter Validation
- [ ] ATR Multiplier tested range: 2.0 - 3.0
- [ ] Minimum volatility appropriate for asset (12% for Gold)
- [ ] Channel length matches timeframe (20 for Daily)
- [ ] Position sizing matches risk tolerance (1-2% per trade)
- [ ] Parameters stable across different time periods

### 3. Code Verification

**TradingView Pine Script:**
- [ ] Code compiles without errors
- [ ] Visual elements display correctly
- [ ] Info table shows accurate data
- [ ] Alerts trigger at correct times
- [ ] Strategy backtester matches expectations

**MetaTrader 5 MQL5:**
- [ ] Code compiles with 0 errors, 0 warnings
- [ ] EA attaches to chart successfully
- [ ] Indicators calculate correctly (ATR, MA, channels)
- [ ] Position opening/closing works in Strategy Tester
- [ ] Money management calculates proper lot sizes

---

## 🔧 Platform Configuration

### TradingView Setup
- [ ] Account: [ ] Free [ ] Pro [ ] Premium
- [ ] Chart timeframe: Daily (D)
- [ ] Symbol: XAUUSD or equivalent
- [ ] Indicator added to chart
- [ ] Settings configured (see INSTALLATION_GUIDE.md)
- [ ] Alerts set up for:
  - [ ] Long Entry
  - [ ] Short Entry  
  - [ ] Exit Long
  - [ ] Exit Short
- [ ] Notification method configured: [ ] Email [ ] App [ ] SMS [ ] Webhook

### MetaTrader 5 Setup
- [ ] Platform version: MT5 (build 3xxx or higher)
- [ ] EA compiled successfully (VolatilityBreakoutATR.ex5)
- [ ] EA attached to chart: XAUUSD Daily
- [ ] "Allow Algo Trading" enabled (green button visible)
- [ ] AutoTrading settings configured:
  - [ ] Allow automated trading
  - [ ] Allow DLL imports (if needed)
  - [ ] Disable "Ask manual confirmation"
- [ ] Risk parameters set:
  - Risk Per Trade: ____%
  - Max Position Size: ____%
  - Fixed Lot Size: _____
  - Enable Money Management: [ ] Yes [ ] No

---

## 🎯 Risk Management Configuration

### Position Sizing
- [ ] Risk per trade: 1-2% (recommended)
- [ ] Maximum position: 25% of account (default)
- [ ] Inverse volatility sizing: [ ] Enabled [ ] Disabled
- [ ] Target portfolio volatility: 15% (default)

**Calculate Your Position Size:**
```
Account Balance: $__________
Risk Per Trade (%): __________
Risk Amount ($): $__________
ATR (Gold): $__________
ATR Multiplier: 2.5
Stop Distance: $__________ (ATR × Multiplier)
Position Size (lots): __________
```

### Filters
- [ ] Volatility filter enabled
- [ ] Minimum volatility: 12% (for Gold) / 8% (for FX)
- [ ] Trading hours filter: [ ] Enabled [ ] Disabled
  - Start hour: _____
  - End hour: _____
- [ ] Time-based exit: [ ] Enabled [ ] Disabled
  - Exit after N bars: _____

---

## 📊 Broker Verification

### Broker Requirements
- [ ] Regulation: [ ] FCA [ ] ASIC [ ] CySEC [ ] Other: __________
- [ ] Account type: [ ] Standard [ ] ECN [ ] Pro
- [ ] Leverage: 1:_____ (recommended max 1:20)
- [ ] Minimum deposit met: $__________

### Trading Conditions
- [ ] Symbol available: XAUUSD (or equivalent: GOLD, XAUUSD.a, etc.)
- [ ] Spread (typical): _____ (should be < $0.30 for Gold)
- [ ] Commission per lot: _____ (should be < $10/lot)
- [ ] Minimum lot size: _____ (typically 0.01)
- [ ] Maximum lot size: _____
- [ ] Lot step: _____ (typically 0.01)
- [ ] Margin requirements understood
- [ ] Swap rates reviewed (overnight fees)

### Execution Quality
- [ ] Tested order execution speed (< 100ms)
- [ ] Slippage tested (should be < 5 pips on average)
- [ ] No requotes during normal conditions
- [ ] Server location: _____ (closer = better)
- [ ] Backup trading method available

---

## 🧪 Paper Trading Phase

### Week 1-2: Observation Only
- [ ] EA running on demo account
- [ ] All signals logged manually
- [ ] Performance tracking started
- [ ] No issues observed with order execution

**Metrics to Track:**
```
Trades Taken: _____
Wins: _____
Losses: _____
Win Rate: _____%
Average Win: $_____
Average Loss: $_____
Profit Factor: _____
Max Drawdown: _____%
```

### Week 3-4: Small Position Size
- [ ] Reduce risk to 0.5% per trade
- [ ] Minimum lot size only
- [ ] Monitor for 20+ trades
- [ ] Results match backtest expectations (+/- 20%)
- [ ] No technical issues

**Go/No-Go Decision:**
- [ ] Sharpe ratio > 0.2 in paper trading
- [ ] Max drawdown < 10%
- [ ] No major technical issues
- [ ] Execution slippage acceptable
- [ ] Emotionally comfortable with strategy

---

## 🚀 Live Deployment

### Initial Live Trading (Month 1)
- [ ] Start date: __________
- [ ] Initial capital: $__________
- [ ] Risk per trade: 1% (conservative)
- [ ] Maximum daily loss limit: _____%
- [ ] Maximum weekly loss limit: _____%
- [ ] Maximum drawdown tolerance: _____%

### Monitoring Schedule

**Daily:**
- [ ] Check open positions
- [ ] Review unrealized P&L
- [ ] Verify EA is running (MT5 expert log)
- [ ] Check ATR level vs minimum volatility
- [ ] Log any unusual events

**Weekly:**
- [ ] Calculate win rate
- [ ] Update trade journal
- [ ] Review average win/loss
- [ ] Check if max drawdown exceeded
- [ ] Compare to backtest metrics

**Monthly:**
- [ ] Calculate Sharpe ratio
- [ ] Review parameter stability
- [ ] Check correlation with other strategies
- [ ] Decide if adjustments needed
- [ ] Update optimization if needed

---

## 📈 Performance Thresholds

### Green Zone (Continue Trading)
- [ ] Sharpe ratio > 0.2
- [ ] Max drawdown < 5%
- [ ] Win rate 40-60%
- [ ] Profit factor > 1.2
- [ ] Results within 1 std dev of backtest

### Yellow Zone (Caution)
- [ ] Sharpe ratio 0.0 - 0.2
- [ ] Max drawdown 5-10%
- [ ] Win rate 30-40% or 60-70%
- [ ] Unusual market conditions
- [ ] Results deviate from backtest

**Action:** Reduce position size by 50%, increase monitoring

### Red Zone (Stop Trading)
- [ ] Sharpe ratio < 0.0 (negative)
- [ ] Max drawdown > 10%
- [ ] Win rate < 30%
- [ ] Profit factor < 1.0
- [ ] Technical issues with EA/broker

**Action:** Stop EA immediately, investigate cause, re-optimize

---

## 🔍 Regular Maintenance

### Weekly Tasks
- [ ] Check for MT5/TradingView updates
- [ ] Verify EA is running (MT5 experts log)
- [ ] Review spread costs (should be stable)
- [ ] Check broker announcements
- [ ] Backup trade history

### Monthly Tasks
- [ ] Calculate performance metrics
- [ ] Update trade journal with lessons learned
- [ ] Review strategy parameters (walk-forward)
- [ ] Check for market regime changes
- [ ] Consider parameter re-optimization

### Quarterly Tasks
- [ ] Full performance review vs backtest
- [ ] Out-of-sample test with new data
- [ ] Parameter stability analysis
- [ ] Consider strategy improvements
- [ ] Tax preparation (if needed)

---

## 📞 Emergency Procedures

### If Max Drawdown Exceeded
1. [ ] Stop EA immediately
2. [ ] Close all open positions (if > 8% DD)
3. [ ] Review trade history for anomalies
4. [ ] Check if market regime changed
5. [ ] Re-run backtest with recent data
6. [ ] Consider parameter adjustment
7. [ ] Do not resume until cause identified

### If Technical Issue Occurs
1. [ ] Document the issue (screenshots)
2. [ ] Check EA log (MT5 experts tab)
3. [ ] Verify broker connection
4. [ ] Test order execution manually
5. [ ] Contact broker if execution issue
6. [ ] Reload EA if necessary
7. [ ] Resume only after issue resolved

### If Strategy Performance Degrades
1. [ ] Compare recent metrics to backtest
2. [ ] Check if market volatility changed
3. [ ] Verify parameters still optimal
4. [ ] Run walk-forward analysis
5. [ ] Consider reducing position size
6. [ ] Pause trading if Sharpe < 0 for 3+ months

---

## ✅ Final Authorization

**Strategy Owner:** __________________________

**Date of Deployment:** __________________________

**Initial Capital:** $__________________________

**Risk Tolerance:** __________________________

**Expected Annual Return:** __________% (realistic: 1-3%)

**Maximum Acceptable Drawdown:** __________% (recommended: 5-10%)

**Review Schedule:** 
- Daily: ___:___ (check positions)
- Weekly: ___day ___ (performance review)
- Monthly: ___day ___ (full analysis)

**Emergency Contact:**
- Broker Support: __________________________
- Trading Partner: __________________________

---

**Signatures:**

Strategy Approved: __________________________ Date: __________

Risk Manager: __________________________ Date: __________

---

**Notes:**
_____________________________________________________________
_____________________________________________________________
_____________________________________________________________
_____________________________________________________________

---

**Remember:**
- This strategy has Sharpe 0.37 (modest risk-adjusted returns)
- Not suitable for get-rich-quick expectations
- Best used as part of diversified portfolio
- Past performance ≠ future results
- Always trade with capital you can afford to lose

**Good luck and trade responsibly!** 🎯
