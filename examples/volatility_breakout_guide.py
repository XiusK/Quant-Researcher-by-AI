"""
Volatility Breakout (ATR) Strategy - Deep Dive

กลยุทธ์ที่ชนะการทดสอบ XAUUSD ด้วย Sharpe 0.37
เอกสารนี้อธิบายเงื่อนไขและเครื่องมือที่ใช้ในทางปฏิบัติ
"""

import sys
sys.path.insert(0, 'e:/Python Project/Quant Researcher By AI')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from src.data import load_xauusd_from_kaggle, calculate_features
from src.strategies.volatility_breakout import VolatilityBreakoutStrategy

# Set style
sns.set_style('darkgrid')
plt.rcParams['figure.figsize'] = (14, 10)

print("="*70)
print("VOLATILITY BREAKOUT (ATR) - STRATEGY BREAKDOWN")
print("="*70)

# ============================================================================
# PART 1: เงื่อนไขของกลยุทธ์ (Strategy Rules)
# ============================================================================

print("\n" + "="*70)
print("PART 1: เงื่อนไขการเข้า-ออกของกลยุทธ์")
print("="*70)

strategy_rules = """
กลยุทธ์ Volatility Breakout (ATR) ใช้หลักการ:
"ราคาที่ breakout จาก channel = โอกาสเทรด"

📊 เครื่องมือหลัก:
1. ATR (Average True Range) - วัดความผันผวนของราคา
2. Dynamic Channel - ช่องราคาที่ปรับตาม volatility
3. Lookback Window - ระยะเวลาที่ใช้คำนวณ channel

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔵 LONG SIGNAL (ซื้อ):
   Condition: Price > Upper Channel
   
   Upper Channel = MA(20) + 2.5 × ATR(14)
   
   ├─ MA(20) = Moving Average 20 วัน (แกนกลาง)
   ├─ ATR(14) = Average True Range 14 วัน (วัด volatility)
   └─ Multiplier = 2.5 (ความกว้างของ channel)

   Confidence = (Price - Upper Channel) / ATR
   → ยิ่งราคา breakout ไกล = ยิ่ง confident

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔴 SHORT SIGNAL (ขาย):
   Condition: Price < Lower Channel
   
   Lower Channel = MA(20) - 2.5 × ATR(14)
   
   Confidence = (Lower Channel - Price) / ATR
   → ยิ่งราคาทะลุลึก = ยิ่ง confident

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚪ EXIT SIGNAL (ออก):
   Condition 1: ราคากลับเข้า channel
   Condition 2: สัญญาณตรงข้ามเกิดขึ้น
   
   เช่น: ถือ Long → ราคากลับเข้าใน channel → Exit

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🛡️ FILTERS (ตัวกรอง):
   1. Minimum Volatility: ATR > 12% annualized
      → ไม่เทรดในตลาดที่เงียบเกินไป
   
   2. Inverse Volatility Sizing:
      Position Size = Base × (Target Vol / Current Vol)
      → เทรดน้อยลงเมื่อ volatility สูง

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

print(strategy_rules)

# ============================================================================
# PART 2: คำนวณและแสดง Indicators
# ============================================================================

print("\n" + "="*70)
print("PART 2: การคำนวณ Indicators (ตัวอย่างจริงจากข้อมูล)")
print("="*70)

# Load recent data
data = load_xauusd_from_kaggle(timeframe="1d")
data = calculate_features(data)

# Get last 100 days for visualization
recent_data = data.tail(100).copy()

# Calculate ATR Channel
atr_period = 14
atr_mult = 2.5
channel_lookback = 20

# ATR
if 'atr_14' not in recent_data.columns:
    high_low = recent_data['high'] - recent_data['low']
    recent_data['atr_14'] = high_low.rolling(window=atr_period).mean()

# Channel middle (MA)
recent_data['channel_middle'] = recent_data['close'].rolling(window=channel_lookback).mean()

# Upper and Lower bands
recent_data['channel_upper'] = recent_data['channel_middle'] + (atr_mult * recent_data['atr_14'])
recent_data['channel_lower'] = recent_data['channel_middle'] - (atr_mult * recent_data['atr_14'])

# Identify breakouts
recent_data['signal'] = 0
recent_data.loc[recent_data['close'] > recent_data['channel_upper'], 'signal'] = 1  # Long
recent_data.loc[recent_data['close'] < recent_data['channel_lower'], 'signal'] = -1  # Short

print("\nตัวอย่างการคำนวณ (5 วันล่าสุด):")
print("-" * 70)

display_cols = ['close', 'atr_14', 'channel_middle', 'channel_upper', 'channel_lower', 'signal']
recent_sample = recent_data[display_cols].tail()

for idx, row in recent_sample.iterrows():
    print(f"\nDate: {idx.strftime('%Y-%m-%d')}")
    print(f"  Close:          ${row['close']:,.2f}")
    print(f"  ATR(14):        ${row['atr_14']:,.2f}")
    print(f"  Channel Middle: ${row['channel_middle']:,.2f}")
    print(f"  Upper Band:     ${row['channel_upper']:,.2f}")
    print(f"  Lower Band:     ${row['channel_lower']:,.2f}")
    
    signal_text = {1: "🔵 LONG", -1: "🔴 SHORT", 0: "⚪ NEUTRAL"}
    print(f"  Signal:         {signal_text[row['signal']]}")
    
    # Distance from bands
    if row['close'] > row['channel_upper']:
        distance = row['close'] - row['channel_upper']
        print(f"  Breakout:       +${distance:.2f} above upper band")
    elif row['close'] < row['channel_lower']:
        distance = row['channel_lower'] - row['close']
        print(f"  Breakout:       -${distance:.2f} below lower band")

# ============================================================================
# PART 3: Visualization
# ============================================================================

print("\n" + "="*70)
print("PART 3: สร้างกราฟแสดง Breakout Signals")
print("="*70)

fig, axes = plt.subplots(3, 1, figsize=(16, 12))

# ---- Chart 1: Price with ATR Channel ----
ax1 = axes[0]

# Plot price
ax1.plot(recent_data.index, recent_data['close'], 
         label='Gold Price', color='black', linewidth=2)

# Plot channel
ax1.plot(recent_data.index, recent_data['channel_middle'], 
         label='Channel Middle (MA 20)', color='blue', linestyle='--', alpha=0.7)
ax1.plot(recent_data.index, recent_data['channel_upper'], 
         label='Upper Band (MA + 2.5×ATR)', color='green', linestyle='--', alpha=0.7)
ax1.plot(recent_data.index, recent_data['channel_lower'], 
         label='Lower Band (MA - 2.5×ATR)', color='red', linestyle='--', alpha=0.7)

# Fill channel
ax1.fill_between(recent_data.index, 
                  recent_data['channel_upper'], 
                  recent_data['channel_lower'],
                  alpha=0.1, color='gray')

# Mark breakout signals
long_signals = recent_data[recent_data['signal'] == 1]
short_signals = recent_data[recent_data['signal'] == -1]

ax1.scatter(long_signals.index, long_signals['close'], 
           marker='^', color='green', s=100, label='LONG Signal', zorder=5)
ax1.scatter(short_signals.index, short_signals['close'], 
           marker='v', color='red', s=100, label='SHORT Signal', zorder=5)

ax1.set_ylabel('Price (USD)', fontsize=12, fontweight='bold')
ax1.set_title('Volatility Breakout (ATR) - Last 100 Days', 
              fontsize=14, fontweight='bold')
ax1.legend(loc='best', fontsize=10)
ax1.grid(True, alpha=0.3)

# ---- Chart 2: ATR Evolution ----
ax2 = axes[1]

ax2.plot(recent_data.index, recent_data['atr_14'], 
         label='ATR(14)', color='purple', linewidth=2)
ax2.axhline(y=recent_data['atr_14'].mean(), 
           color='orange', linestyle='--', alpha=0.7,
           label=f'Average ATR: ${recent_data["atr_14"].mean():.2f}')

ax2.set_ylabel('ATR (USD)', fontsize=12, fontweight='bold')
ax2.set_title('Average True Range - Volatility Measure', 
              fontsize=13, fontweight='bold')
ax2.legend(loc='best', fontsize=10)
ax2.grid(True, alpha=0.3)

# ---- Chart 3: Channel Width (as % of price) ----
ax3 = axes[2]

channel_width = (recent_data['channel_upper'] - recent_data['channel_lower']) / recent_data['close'] * 100

ax3.plot(recent_data.index, channel_width, 
         label='Channel Width (%)', color='teal', linewidth=2)
ax3.axhline(y=channel_width.mean(), 
           color='red', linestyle='--', alpha=0.7,
           label=f'Average: {channel_width.mean():.2f}%')

ax3.set_ylabel('Channel Width (%)', fontsize=12, fontweight='bold')
ax3.set_xlabel('Date', fontsize=12, fontweight='bold')
ax3.set_title('Dynamic Channel Width - Adapts to Market Volatility', 
              fontsize=13, fontweight='bold')
ax3.legend(loc='best', fontsize=10)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('volatility_breakout_detailed_analysis.png', dpi=150, bbox_inches='tight')
print("\n✅ Saved: volatility_breakout_detailed_analysis.png")

# ============================================================================
# PART 4: Performance Statistics
# ============================================================================

print("\n" + "="*70)
print("PART 4: สถิติการทำงานของกลยุทธ์")
print("="*70)

# Count signals
n_long = (recent_data['signal'] == 1).sum()
n_short = (recent_data['signal'] == -1).sum()
n_neutral = (recent_data['signal'] == 0).sum()

print(f"\n📊 Signal Distribution (Last 100 days):")
print(f"  LONG signals:    {n_long:3d} days ({n_long/len(recent_data)*100:.1f}%)")
print(f"  SHORT signals:   {n_short:3d} days ({n_short/len(recent_data)*100:.1f}%)")
print(f"  NEUTRAL:         {n_neutral:3d} days ({n_neutral/len(recent_data)*100:.1f}%)")

# ATR statistics
print(f"\n📈 ATR Statistics:")
print(f"  Current ATR:     ${recent_data['atr_14'].iloc[-1]:,.2f}")
print(f"  Average ATR:     ${recent_data['atr_14'].mean():,.2f}")
print(f"  Max ATR:         ${recent_data['atr_14'].max():,.2f}")
print(f"  Min ATR:         ${recent_data['atr_14'].min():,.2f}")
print(f"  ATR % of price:  {recent_data['atr_14'].iloc[-1]/recent_data['close'].iloc[-1]*100:.2f}%")

# Channel statistics
print(f"\n📏 Channel Statistics:")
print(f"  Current Width:   {channel_width.iloc[-1]:.2f}%")
print(f"  Average Width:   {channel_width.mean():.2f}%")
print(f"  Max Width:       {channel_width.max():.2f}%")
print(f"  Min Width:       {channel_width.min():.2f}%")

# ============================================================================
# PART 5: Trading Rules Summary
# ============================================================================

print("\n" + "="*70)
print("PART 5: กฎการเทรดในทางปฏิบัติ")
print("="*70)

trading_rules = """
┌─────────────────────────────────────────────────────────────────────┐
│  ENTRY CHECKLIST (เงื่อนไขเข้า)                                     │
├─────────────────────────────────────────────────────────────────────┤
│  1. ✅ Check ATR Level                                              │
│     → Current ATR > 12% annualized volatility                      │
│     → ถ้า ATR ต่ำเกินไป = ตลาดเงียบ = ไม่เทรด                     │
│                                                                     │
│  2. ✅ Calculate Channel                                            │
│     → MA(20) = แกนกลาง                                             │
│     → Upper = MA + 2.5×ATR                                         │
│     → Lower = MA - 2.5×ATR                                         │
│                                                                     │
│  3. ✅ Wait for Breakout                                            │
│     LONG:  Close > Upper Band                                      │
│     SHORT: Close < Lower Band                                      │
│                                                                     │
│  4. ✅ Calculate Confidence                                         │
│     Distance = |Price - Band|                                      │
│     Confidence = Distance / ATR                                    │
│     → ยิ่งทะลุไกล = ยิ่ง confident                                 │
│                                                                     │
│  5. ✅ Position Sizing                                              │
│     Base = 25% of portfolio                                        │
│     Adjusted = Base × (15% / Current_Vol) × Confidence             │
│     → ลดขนาด position เมื่อ vol สูง                                │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  EXIT CHECKLIST (เงื่อนไขออก)                                       │
├─────────────────────────────────────────────────────────────────────┤
│  1. ✅ Price Returns to Channel                                     │
│     → ราคากลับเข้าใน channel = แนวโน้ม breakout สิ้นสุด            │
│                                                                     │
│  2. ✅ Opposite Signal                                              │
│     → ถือ LONG แล้วเกิด SHORT signal = ออกทันที                    │
│                                                                     │
│  3. ✅ Time-Based (Optional)                                        │
│     → ถือเกิน 5 วันไม่มี momentum = พิจารณาออก                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  RISK MANAGEMENT (การจัดการความเสี่ยง)                             │
├─────────────────────────────────────────────────────────────────────┤
│  1. ✅ Max Position: 25% of portfolio                               │
│  2. ✅ Max Leverage: 1.0 (ไม่ใช้ leverage)                          │
│  3. ✅ Stop Loss: ไม่ใช้ hard stop (ใช้ channel return)             │
│  4. ✅ Volatility Adjustment: ลดขนาดเมื่อ vol > 20%                 │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  TOOLS & INDICATORS (เครื่องมือที่ใช้)                              │
├─────────────────────────────────────────────────────────────────────┤
│  📊 Primary:                                                        │
│     • ATR(14) - Average True Range                                 │
│     • MA(20) - Simple Moving Average                               │
│                                                                     │
│  📈 Secondary:                                                      │
│     • Realized Volatility (20-day rolling)                         │
│     • Price-to-Channel Position                                    │
│                                                                     │
│  🔧 Platform:                                                       │
│     • TradingView: Built-in ATR indicator                          │
│     • MetaTrader: Custom ATR Channel EA                            │
│     • Python: pandas, numpy สำหรับ backtest                        │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  PARAMETER OPTIMIZATION (การปรับพารามิเตอร์)                        │
├─────────────────────────────────────────────────────────────────────┤
│  Current Settings (ที่ชนะ):                                        │
│     • ATR Period: 14 days                                          │
│     • ATR Multiplier: 2.5                                          │
│     • Channel Lookback: 20 days                                    │
│     • Min Volatility: 12% annualized                               │
│                                                                     │
│  การปรับ:                                                           │
│     • เพิ่ม ATR Mult → น้อย signal แต่แม่นขึ้น                     │
│     • ลด ATR Mult → เยอะ signal แต่ whipsaw มากขึ้น                │
│     • เพิ่ม Lookback → ช้าลง แต่เสถียรขึ้น                          │
└─────────────────────────────────────────────────────────────────────┘
"""

print(trading_rules)

# ============================================================================
# PART 6: Real-World Example
# ============================================================================

print("\n" + "="*70)
print("PART 6: ตัวอย่างการเทรดจริง (Recent Signal)")
print("="*70)

# Find most recent signal
recent_signals = recent_data[recent_data['signal'] != 0].tail(3)

if len(recent_signals) > 0:
    print("\nสัญญาณล่าสุด 3 ครั้ง:")
    print("-" * 70)
    
    for idx, row in recent_signals.iterrows():
        signal_type = "🔵 LONG" if row['signal'] == 1 else "🔴 SHORT"
        
        print(f"\n{signal_type} Signal")
        print(f"  Date:            {idx.strftime('%Y-%m-%d')}")
        print(f"  Price:           ${row['close']:,.2f}")
        print(f"  Channel Middle:  ${row['channel_middle']:,.2f}")
        print(f"  Upper Band:      ${row['channel_upper']:,.2f}")
        print(f"  Lower Band:      ${row['channel_lower']:,.2f}")
        print(f"  ATR:             ${row['atr_14']:,.2f}")
        
        if row['signal'] == 1:
            breakout_dist = row['close'] - row['channel_upper']
            confidence = breakout_dist / row['atr_14']
            print(f"  Breakout:        +${breakout_dist:.2f} ({confidence:.2f}×ATR)")
            print(f"  Entry Reason:    Price broke above upper band")
        else:
            breakout_dist = row['channel_lower'] - row['close']
            confidence = breakout_dist / row['atr_14']
            print(f"  Breakout:        -${breakout_dist:.2f} ({confidence:.2f}×ATR)")
            print(f"  Entry Reason:    Price broke below lower band")

print("\n" + "="*70)
print("✅ Analysis Complete!")
print("="*70)

print("""
📚 สรุปความรู้:

1. ATR = เครื่องมือวัด Volatility ที่ดีกว่า Standard Deviation
   → รวม gap และ trend moves

2. Dynamic Channel = ปรับความกว้างตาม market condition
   → กว้างขึ้นเมื่อ volatile, แคบลงเมื่อเงียบ

3. Breakout = สัญญาณว่ามี momentum
   → แต่ต้อง filter ด้วย minimum volatility

4. Inverse Vol Sizing = ควบคุมความเสี่ยง
   → เทรดน้อยลงเมื่อตลาดผันผวนมาก

5. Channel Return = Exit signal ที่ดีกว่า fixed stop loss
   → ให้ profit run แต่ cut เมื่อ momentum หาย
""")
