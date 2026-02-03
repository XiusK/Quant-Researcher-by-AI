# Volatility Breakout ATR Strategy - Installation Guide

Complete guide for deploying the winning strategy (Sharpe 0.37) on TradingView and MetaTrader 5.

---

## 📊 TradingView Pine Script

### Installation Steps:

1. **Open TradingView**
   - Go to https://www.tradingview.com
   - Login to your account

2. **Create New Indicator**
   - Click "Pine Editor" at bottom of chart
   - Click "Open" → "New blank indicator"

3. **Copy Code**
   - Open file: `tradingview/volatility_breakout_atr.pine`
   - Select all (Ctrl+A) and copy (Ctrl+C)

4. **Paste and Compile**
   - Paste code into Pine Editor
   - Click "Save" (give it a name: "Volatility Breakout ATR")
   - Click "Add to Chart"

5. **Configure Settings**
   - Click ⚙️ (Settings) on the indicator
   - Adjust parameters:
     ```
     Channel Length: 20 (default)
     ATR Period: 14 (default)
     ATR Multiplier: 2.5 (recommended for Gold)
     Min Volatility: 12% (filter low-vol periods)
     ```

### Usage:

**Visual Elements:**
- Blue line = Channel Middle (SMA 20)
- Green dashed = Upper Band (MA + 2.5×ATR)
- Red dashed = Lower Band (MA - 2.5×ATR)
- Green triangles ▲ = LONG signals
- Red triangles ▼ = SHORT signals
- Orange X = Exit signals

**Info Table (Top Right):**
- ATR value
- Realized Volatility (must be > 12%)
- Channel Width (%)
- Current Position Size
- Current Position (LONG/SHORT/FLAT)
- Confidence Score (in ATR units)
- Vol Filter Status (PASS/FAIL)

### Setting Alerts:

1. Click "Alert" (clock icon) on chart
2. Select "Volatility Breakout ATR" indicator
3. Choose condition:
   - "Long Entry" = notification when LONG signal
   - "Short Entry" = notification when SHORT signal
   - "Exit Long" = notification to close LONG
   - "Exit Short" = notification to close SHORT

4. Configure notification method:
   - Email
   - SMS (premium)
   - Webhook (for automation)

5. Set alert frequency: "Once Per Bar Close"

---

## 🤖 MetaTrader 5 Expert Advisor

### Installation Steps:

1. **Open MetaEditor**
   - In MT5: Tools → MetaQuotes Language Editor (F4)
   - Or click "MetaEditor" icon in toolbar

2. **Create New EA**
   - File → New → Expert Advisor (template)
   - Or: File → Open → Navigate to file location

3. **Copy Code**
   - Open file: `metatrader5/VolatilityBreakoutATR.mq5`
   - Copy all code
   - Paste into MetaEditor

4. **Compile**
   - Click "Compile" button (F7)
   - Check for 0 errors, 0 warnings
   - File will be saved in: `MQL5/Experts/VolatilityBreakoutATR.ex5`

5. **Attach to Chart**
   - In MT5 Navigator panel: Experts → VolatilityBreakoutATR
   - Drag & drop onto XAUUSD chart
   - Check "Allow Algo Trading" checkbox
   - Click OK

### Configuration:

#### Channel Settings:
```
Channel Length: 20        // Period for moving average
ATR Period: 14            // Period for volatility calculation
ATR Multiplier: 2.5       // Width of channel (2.5x ATR)
```

#### Filter Settings:
```
Enable Volatility Filter: true    // Only trade when vol > min
Minimum Volatility: 12.0%         // Annualized vol threshold
Enable Time-Based Exit: false     // Exit after N bars
Exit After N Bars: 5              // Hold duration limit
```

#### Risk Management:
```
Risk Per Trade: 2.0%              // % of account per trade
Max Position Size: 25.0%          // Max exposure (% of account)
Target Portfolio Volatility: 15.0%  // For inverse vol sizing
Inverse Volatility Sizing: true   // Scale down when vol high
```

#### Trading Settings:
```
Fixed Lot Size: 0.1               // If not using money management
Enable Money Management: true     // Dynamic position sizing
Magic Number: 202602              // Unique EA identifier
Trade Comment: "VB-ATR"           // Order comment
Slippage: 30 points               // Max slippage allowed
```

#### Trading Hours (Optional):
```
Enable Trading Hours Filter: false  // Limit trading time
Trading Start Hour: 0               // Start hour (GMT)
Trading End Hour: 23                // End hour (GMT)
```

### Important Settings in MT5:

**Enable Algo Trading:**
1. Tools → Options → Expert Advisors
2. Check "Allow automated trading"
3. Check "Allow DLL imports" (if needed)
4. Check "Allow WebRequest for listed URL" (for webhooks)

**One Chart One EA:**
- Only attach EA to ONE chart (e.g., XAUUSD Daily)
- EA will manage positions on that symbol only
- Do not run multiple instances on same symbol

**Backtest Before Live:**
1. View → Strategy Tester (Ctrl+R)
2. Select: VolatilityBreakoutATR
3. Symbol: XAUUSD
4. Period: D1 (Daily)
5. Date range: 2019-01-01 to 2024-12-31
6. Optimization: None (use default parameters)
7. Click "Start"

Expected Results (XAUUSD Daily 2019-2025):
- Total Return: ~6-7%
- Sharpe Ratio: 0.35-0.40
- Max Drawdown: 3-5%
- Trades: 300-400

---

## 📈 Parameter Optimization Guide

### For Different Assets:

| Asset | ATR Multiplier | Min Volatility | Timeframe |
|-------|----------------|----------------|-----------|
| XAUUSD (Gold) | 2.5 | 12% | Daily |
| EURUSD (FX) | 2.0 | 8% | H4 |
| BTCUSD (Crypto) | 3.0 | 20% | H1 |
| SPX500 (Index) | 2.2 | 10% | Daily |
| USDJPY (FX) | 2.0 | 8% | H4 |

### Tuning Guidelines:

**ATR Multiplier (1.5 - 3.5):**
- **Lower (1.5-2.0)**: More signals, higher risk of whipsaw
  - Use for: Mean reversion, ranging markets
- **Higher (2.5-3.5)**: Fewer signals, higher quality
  - Use for: Breakout, trending markets (our choice for Gold)

**Minimum Volatility (8% - 20%):**
- **Lower (8-10%)**: Trade more often, accept lower vol
  - Use for: FX pairs, stable assets
- **Higher (15-20%)**: Only trade explosive moves
  - Use for: Crypto, commodities during events

**Channel Length (10 - 50):**
- **Shorter (10-15)**: Responsive to recent price action
  - Use for: Intraday, volatile markets
- **Longer (30-50)**: Smooth trend following
  - Use for: Daily, slower markets

### Walk-Forward Optimization:

1. **In-Sample Period**: 2019-2022 (3 years)
   - Optimize parameters
   - Find best ATR multiplier

2. **Out-of-Sample Test**: 2023-2024 (2 years)
   - Test with optimized parameters
   - Verify Sharpe > 0.3

3. **Re-optimize**: Every 6 months
   - Check if parameters still work
   - Adjust if market regime changed

---

## 🚨 Risk Warnings

### Before Going Live:

✅ **DO:**
- Backtest on historical data (min 3 years)
- Paper trade for 1-2 months
- Start with minimum lot size
- Monitor first 10 trades closely
- Keep detailed trade journal
- Set realistic expectations (Sharpe ~0.3-0.4)

❌ **DON'T:**
- Over-optimize parameters (curve fitting)
- Trade with more than 2% risk per trade
- Use maximum leverage
- Ignore volatility regime changes
- Trade without stop loss (use ATR-based)
- Expect every trade to win (win rate ~45-55%)

### Known Limitations:

1. **Low Sharpe Ratio**: Strategy shows Sharpe 0.37 (below 1.0)
   - Not suitable for aggressive traders
   - Better for portfolio diversification
   - Combines well with other uncorrelated strategies

2. **Trending Market Dependency**:
   - Performs best in trending environments
   - Underperforms in choppy/ranging markets
   - Volatility filter helps but not perfect

3. **Transaction Costs**:
   - Sensitive to spreads and commissions
   - Use ECN broker with tight spreads
   - Gold spreads should be < 30 cents ($0.30)

4. **Slippage**:
   - Breakout entries may experience slippage
   - Limit orders not used (market orders)
   - Higher risk during news events

---

## 📊 Performance Monitoring

### Key Metrics to Track:

**Daily:**
- Open positions
- Unrealized P&L
- ATR level (should be > min volatility)
- Channel width (wider = more volatile)

**Weekly:**
- Number of trades
- Win rate (expect 45-55%)
- Average win vs average loss (expect 1.5:1)
- Maximum drawdown (alarm if > 5%)

**Monthly:**
- Total return
- Sharpe ratio (expect 0.3-0.4)
- Correlation with other strategies
- Parameter stability

### When to Stop Trading:

🛑 **Stop Conditions:**
1. Drawdown exceeds 10% (2x historical max)
2. Win rate drops below 35% for 30+ trades
3. Sharpe ratio negative for 3+ months
4. Market regime change (e.g., prolonged ranging)
5. Broker spread widens significantly

---

## 🔧 Troubleshooting

### TradingView Issues:

**Q: Indicator not showing on chart**
- A: Check if indicator is added to chart (should see name in top left)
- Try removing and re-adding

**Q: Signals not matching backtest**
- A: Ensure "Recalculate on every tick" is OFF
- Use "On bar close" for alerts

**Q: Strategy shows negative results**
- A: Check commission and slippage settings
- Default: 0.08% commission, 3 ticks slippage

### MT5 Issues:

**Q: EA not taking trades**
- A: Check "Allow Algo Trading" is enabled (green button in toolbar)
- Verify symbol is correct (XAUUSD vs GOLD vs XAUUSD.a)
- Check trading hours filter

**Q: Position size too large/small**
- A: Adjust "Risk Per Trade" or "Fixed Lot Size"
- Check account balance and leverage
- Verify "Enable Money Management" setting

**Q: "Not enough money" error**
- A: Reduce lot size or risk percentage
- Check margin requirements for symbol
- Ensure sufficient free margin

**Q: Orders rejected**
- A: Check minimum volume (usually 0.01 lots)
- Verify price is within trading hours
- Check if market is closed

---

## 📞 Support & Resources

### Documentation:
- Python backtest code: `xauusd_strategy_hunt.py`
- Strategy explanation: `volatility_breakout_guide.py`
- Channel comparison: `channel_comparison_guide.py`

### Recommended Brokers:

**For MT5:**
- IC Markets (low spreads, ECN)
- Pepperstone (good execution)
- FXTM (regulated, reliable)

**For TradingView:**
- Use with any broker that supports webhooks
- Or manual trading based on alerts

### Further Optimization:
Contact for advanced features:
- Machine learning parameter adaptation
- Multi-timeframe confirmation
- Regime detection integration
- Portfolio-level risk management

---

## ✅ Final Checklist

Before going live, confirm:

- [ ] Backtested on 3+ years of data
- [ ] Parameters match your asset/timeframe
- [ ] Risk per trade ≤ 2%
- [ ] Volatility filter enabled (if using)
- [ ] Broker spreads < 0.1% (10 bps)
- [ ] "Allow Algo Trading" enabled (MT5)
- [ ] Alert notifications set up (TradingView)
- [ ] Trade journal ready for logging
- [ ] Stop-loss rules defined
- [ ] Capital you can afford to lose

---

**Disclaimer**: This strategy is provided for educational purposes. Past performance does not guarantee future results. Always test thoroughly and never risk more than you can afford to lose. Trading involves substantial risk of loss.

**Strategy Performance (XAUUSD 2019-2025)**:
- Sharpe Ratio: 0.37
- Total Return: +6.34%
- Max Drawdown: -3.53%
- Trades: 365
- Win Rate: ~48%

Good luck and trade responsibly! 🎯
