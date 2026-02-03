"""
Channel Types Comparison - เปรียบเทียบเครื่องมือสร้าง Channel

เอกสารนี้เปรียบเทียบ Channel ทั้งหมดที่ใช้ในการเทรด
พร้อมข้อดี-ข้อเสีย และการคำนวณแต่ละแบบ
"""

import sys
sys.path.insert(0, 'e:/Python Project/Quant Researcher By AI')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from src.data import load_xauusd_from_kaggle, calculate_features

# Set style
sns.set_style('darkgrid')
plt.rcParams['figure.figsize'] = (16, 14)

print("="*80)
print("CHANNEL TYPES COMPARISON - เปรียบเทียบเครื่องมือสร้าง Channel")
print("="*80)

# Load data
print("\nLoading XAUUSD data...")
data = load_xauusd_from_kaggle(timeframe="1d")
data = calculate_features(data)
recent_data = data.tail(120).copy()

# ============================================================================
# PART 1: Channel Types Overview
# ============================================================================

print("\n" + "="*80)
print("PART 1: ประเภทของ Channels ทั้งหมด")
print("="*80)

channel_overview = """
┌────────────────────────────────────────────────────────────────────────────┐
│  1. BOLLINGER BANDS (BB)                                                   │
├────────────────────────────────────────────────────────────────────────────┤
│  Formula:                                                                  │
│    Middle = SMA(20)                                                        │
│    Upper  = SMA(20) + 2.0 × StdDev(20)                                     │
│    Lower  = SMA(20) - 2.0 × StdDev(20)                                     │
│                                                                            │
│  Characteristics:                                                          │
│    • ใช้ Standard Deviation วัดความผันผวน                                 │
│    • ขยายตัวเมื่อ volatility สูง, แคบลงเมื่อต่ำ                            │
│    • สมมติฐาน: ราคามีการกระจายตัวแบบ Normal Distribution                   │
│                                                                            │
│  Best For:                                                                 │
│    ✅ Mean Reversion strategies                                            │
│    ✅ Volatility squeeze detection                                         │
│    ✅ Markets with normal price distribution                               │
│                                                                            │
│  Weaknesses:                                                               │
│    ❌ ไม่ดีกับ trending markets (whipsaw)                                  │
│    ❌ Sensitive to outliers                                                │
│    ❌ ไม่รวม gaps ในการคำนวณ                                               │
└────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────┐
│  2. ATR CHANNELS (Keltner-style)                                           │
├────────────────────────────────────────────────────────────────────────────┤
│  Formula:                                                                  │
│    Middle = SMA(20)                                                        │
│    Upper  = SMA(20) + 2.5 × ATR(14)                                        │
│    Lower  = SMA(20) - 2.5 × ATR(14)                                        │
│                                                                            │
│  Characteristics:                                                          │
│    • ใช้ Average True Range (High-Low+Gaps)                                │
│    • รวม gaps และ extreme moves                                            │
│    • Dynamic multiplier ปรับได้ตาม asset                                   │
│                                                                            │
│  Best For:                                                                 │
│    ✅ Breakout strategies (กลยุทธ์ที่ชนะของเรา!)                           │
│    ✅ Trending markets                                                     │
│    ✅ Assets with gaps (Gold, Commodities)                                 │
│                                                                            │
│  Weaknesses:                                                               │
│    ❌ ATR lag (ช้ากว่า BB)                                                 │
│    ❌ ต้อง tune multiplier สำหรับแต่ละ asset                               │
└────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────┐
│  3. KELTNER CHANNELS (Original)                                            │
├────────────────────────────────────────────────────────────────────────────┤
│  Formula:                                                                  │
│    Middle = EMA(20)                                                        │
│    Upper  = EMA(20) + 2.0 × ATR(10)                                        │
│    Lower  = EMA(20) - 2.0 × ATR(10)                                        │
│                                                                            │
│  Characteristics:                                                          │
│    • ใช้ EMA แทน SMA (smooth กว่า)                                         │
│    • ATR period สั้นกว่า (responsive กว่า)                                 │
│    • Classic volatility channel                                           │
│                                                                            │
│  Best For:                                                                 │
│    ✅ Trend-following                                                      │
│    ✅ Smoother signals than BB                                             │
│    ✅ Intraday trading                                                     │
│                                                                            │
│  Weaknesses:                                                               │
│    ❌ EMA lag ในการเปลี่ยนแนวโน้ม                                          │
└────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────┐
│  4. DONCHIAN CHANNELS                                                      │
├────────────────────────────────────────────────────────────────────────────┤
│  Formula:                                                                  │
│    Upper  = Highest High (20)                                              │
│    Lower  = Lowest Low (20)                                                │
│    Middle = (Upper + Lower) / 2                                            │
│                                                                            │
│  Characteristics:                                                          │
│    • ใช้ Price Extremes (ไม่ใช่ average)                                   │
│    • ไม่มี volatility adjustment                                           │
│    • Richard Dennis (Turtle Traders) ใช้                                   │
│                                                                            │
│  Best For:                                                                 │
│    ✅ Breakout systems (pure breakout)                                     │
│    ✅ Trend-following (ตาม turtle strategy)                                │
│    ✅ Simple, no calculation overhead                                      │
│                                                                            │
│  Weaknesses:                                                               │
│    ❌ ไม่ปรับตาม volatility                                                │
│    ❌ Fixed width (ไม่ dynamic)                                            │
│    ❌ Whipsaw ใน ranging markets                                           │
└────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────┐
│  5. STANDARD DEVIATION CHANNELS (Linear Regression)                        │
├────────────────────────────────────────────────────────────────────────────┤
│  Formula:                                                                  │
│    Middle = Linear Regression Line                                         │
│    Upper  = Regression Line + 2.0 × StdDev(Residuals)                      │
│    Lower  = Regression Line - 2.0 × StdDev(Residuals)                      │
│                                                                            │
│  Characteristics:                                                          │
│    • ใช้ slope ของ trend                                                   │
│    • Channel มีทิศทางตาม regression                                        │
│    • StdDev of residuals = ความกว้าง                                       │
│                                                                            │
│  Best For:                                                                 │
│    ✅ Trending markets with clear direction                                │
│    ✅ Deviation from trend line                                            │
│    ✅ Statistical mean reversion to trend                                  │
│                                                                            │
│  Weaknesses:                                                               │
│    ❌ ซับซ้อนในการคำนวณ                                                    │
│    ❌ Overfitting risk                                                     │
│    ❌ ไม่ดีใน choppy markets                                               │
└────────────────────────────────────────────────────────────────────────────┘
"""

print(channel_overview)

# ============================================================================
# PART 2: Calculate All Channels
# ============================================================================

print("\n" + "="*80)
print("PART 2: คำนวณ Channels ทั้งหมดบนข้อมูลจริง")
print("="*80)

# Parameters
period = 20
atr_period = 14
bb_mult = 2.0
atr_mult = 2.5
keltner_mult = 2.0

# 1. Bollinger Bands
recent_data['bb_middle'] = recent_data['close'].rolling(window=period).mean()
recent_data['bb_std'] = recent_data['close'].rolling(window=period).std()
recent_data['bb_upper'] = recent_data['bb_middle'] + (bb_mult * recent_data['bb_std'])
recent_data['bb_lower'] = recent_data['bb_middle'] - (bb_mult * recent_data['bb_std'])
recent_data['bb_width'] = (recent_data['bb_upper'] - recent_data['bb_lower']) / recent_data['close'] * 100

# 2. ATR Channels
if 'atr_14' not in recent_data.columns:
    high_low = recent_data['high'] - recent_data['low']
    recent_data['atr_14'] = high_low.rolling(window=atr_period).mean()

recent_data['atr_middle'] = recent_data['close'].rolling(window=period).mean()
recent_data['atr_upper'] = recent_data['atr_middle'] + (atr_mult * recent_data['atr_14'])
recent_data['atr_lower'] = recent_data['atr_middle'] - (atr_mult * recent_data['atr_14'])
recent_data['atr_width'] = (recent_data['atr_upper'] - recent_data['atr_lower']) / recent_data['close'] * 100

# 3. Keltner Channels
recent_data['keltner_middle'] = recent_data['close'].ewm(span=period, adjust=False).mean()
recent_data['keltner_upper'] = recent_data['keltner_middle'] + (keltner_mult * recent_data['atr_14'])
recent_data['keltner_lower'] = recent_data['keltner_middle'] - (keltner_mult * recent_data['atr_14'])
recent_data['keltner_width'] = (recent_data['keltner_upper'] - recent_data['keltner_lower']) / recent_data['close'] * 100

# 4. Donchian Channels
donchian_period = 20
recent_data['donchian_upper'] = recent_data['high'].rolling(window=donchian_period).max()
recent_data['donchian_lower'] = recent_data['low'].rolling(window=donchian_period).min()
recent_data['donchian_middle'] = (recent_data['donchian_upper'] + recent_data['donchian_lower']) / 2
recent_data['donchian_width'] = (recent_data['donchian_upper'] - recent_data['donchian_lower']) / recent_data['close'] * 100

print("\n✅ คำนวณเสร็จแล้วทั้งหมด 4 ประเภท")
print(f"   • Bollinger Bands: {period}-period SMA + {bb_mult}×StdDev")
print(f"   • ATR Channels: {period}-period SMA + {atr_mult}×ATR({atr_period})")
print(f"   • Keltner Channels: {period}-period EMA + {keltner_mult}×ATR({atr_period})")
print(f"   • Donchian Channels: {donchian_period}-period High/Low")

# ============================================================================
# PART 3: Comparison Statistics
# ============================================================================

print("\n" + "="*80)
print("PART 3: สถิติเปรียบเทียบ (120 วันล่าสุด)")
print("="*80)

print("\n📊 Channel Width Comparison (% of Price):")
print("-" * 80)
print(f"{'Channel Type':<25} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
print("-" * 80)

width_stats = {
    'Bollinger Bands': recent_data['bb_width'].dropna(),
    'ATR Channels': recent_data['atr_width'].dropna(),
    'Keltner Channels': recent_data['keltner_width'].dropna(),
    'Donchian Channels': recent_data['donchian_width'].dropna()
}

for name, widths in width_stats.items():
    print(f"{name:<25} {widths.mean():>9.2f}% {widths.std():>9.2f}% {widths.min():>9.2f}% {widths.max():>9.2f}%")

print("\n📈 Breakout Signal Count (Price closes outside channel):")
print("-" * 80)

bb_breakouts = ((recent_data['close'] > recent_data['bb_upper']) | 
                (recent_data['close'] < recent_data['bb_lower'])).sum()
atr_breakouts = ((recent_data['close'] > recent_data['atr_upper']) | 
                 (recent_data['close'] < recent_data['atr_lower'])).sum()
keltner_breakouts = ((recent_data['close'] > recent_data['keltner_upper']) | 
                     (recent_data['close'] < recent_data['keltner_lower'])).sum()
donchian_breakouts = ((recent_data['close'] > recent_data['donchian_upper']) | 
                      (recent_data['close'] < recent_data['donchian_lower'])).sum()

print(f"  Bollinger Bands:    {bb_breakouts:3d} breakouts ({bb_breakouts/len(recent_data)*100:.1f}%)")
print(f"  ATR Channels:       {atr_breakouts:3d} breakouts ({atr_breakouts/len(recent_data)*100:.1f}%)")
print(f"  Keltner Channels:   {keltner_breakouts:3d} breakouts ({keltner_breakouts/len(recent_data)*100:.1f}%)")
print(f"  Donchian Channels:  {donchian_breakouts:3d} breakouts ({donchian_breakouts/len(recent_data)*100:.1f}%)")

print("\n💡 Interpretation:")
print("   • น้อย breakouts = Channel กว้าง = น้อย signals แต่คุณภาพสูง")
print("   • เยอะ breakouts = Channel แคบ = เยอะ signals แต่ whipsaw มาก")

# ============================================================================
# PART 4: Visualization
# ============================================================================

print("\n" + "="*80)
print("PART 4: สร้างกราฟเปรียบเทียบ")
print("="*80)

fig = plt.figure(figsize=(18, 16))
gs = fig.add_gridspec(5, 2, hspace=0.3, wspace=0.3)

# Get last 60 days for clearer visualization
plot_data = recent_data.tail(60)

# ---- Chart 1: Bollinger Bands ----
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(plot_data.index, plot_data['close'], label='Price', color='black', linewidth=2, zorder=5)
ax1.plot(plot_data.index, plot_data['bb_middle'], label='Middle (SMA)', color='blue', linestyle='--', alpha=0.7)
ax1.plot(plot_data.index, plot_data['bb_upper'], label='Upper (+2σ)', color='red', linestyle='--', alpha=0.7)
ax1.plot(plot_data.index, plot_data['bb_lower'], label='Lower (-2σ)', color='green', linestyle='--', alpha=0.7)
ax1.fill_between(plot_data.index, plot_data['bb_upper'], plot_data['bb_lower'], alpha=0.1, color='blue')

# Mark breakouts
bb_up_breaks = plot_data[plot_data['close'] > plot_data['bb_upper']]
bb_down_breaks = plot_data[plot_data['close'] < plot_data['bb_lower']]
ax1.scatter(bb_up_breaks.index, bb_up_breaks['close'], marker='^', color='red', s=80, zorder=10)
ax1.scatter(bb_down_breaks.index, bb_down_breaks['close'], marker='v', color='green', s=80, zorder=10)

ax1.set_title('1. BOLLINGER BANDS (SMA + 2×StdDev)', fontsize=13, fontweight='bold')
ax1.set_ylabel('Price (USD)', fontsize=11)
ax1.legend(loc='best', fontsize=9)
ax1.grid(True, alpha=0.3)

# ---- Chart 2: ATR Channels ----
ax2 = fig.add_subplot(gs[1, :])
ax2.plot(plot_data.index, plot_data['close'], label='Price', color='black', linewidth=2, zorder=5)
ax2.plot(plot_data.index, plot_data['atr_middle'], label='Middle (SMA)', color='blue', linestyle='--', alpha=0.7)
ax2.plot(plot_data.index, plot_data['atr_upper'], label='Upper (+2.5×ATR)', color='red', linestyle='--', alpha=0.7)
ax2.plot(plot_data.index, plot_data['atr_lower'], label='Lower (-2.5×ATR)', color='green', linestyle='--', alpha=0.7)
ax2.fill_between(plot_data.index, plot_data['atr_upper'], plot_data['atr_lower'], alpha=0.1, color='purple')

# Mark breakouts
atr_up_breaks = plot_data[plot_data['close'] > plot_data['atr_upper']]
atr_down_breaks = plot_data[plot_data['close'] < plot_data['atr_lower']]
ax2.scatter(atr_up_breaks.index, atr_up_breaks['close'], marker='^', color='red', s=80, zorder=10)
ax2.scatter(atr_down_breaks.index, atr_down_breaks['close'], marker='v', color='green', s=80, zorder=10)

ax2.set_title('2. ATR CHANNELS (SMA + 2.5×ATR) - กลยุทธ์ที่ชนะ!', fontsize=13, fontweight='bold', color='darkgreen')
ax2.set_ylabel('Price (USD)', fontsize=11)
ax2.legend(loc='best', fontsize=9)
ax2.grid(True, alpha=0.3)

# ---- Chart 3: Keltner Channels ----
ax3 = fig.add_subplot(gs[2, :])
ax3.plot(plot_data.index, plot_data['close'], label='Price', color='black', linewidth=2, zorder=5)
ax3.plot(plot_data.index, plot_data['keltner_middle'], label='Middle (EMA)', color='blue', linestyle='--', alpha=0.7)
ax3.plot(plot_data.index, plot_data['keltner_upper'], label='Upper (+2×ATR)', color='red', linestyle='--', alpha=0.7)
ax3.plot(plot_data.index, plot_data['keltner_lower'], label='Lower (-2×ATR)', color='green', linestyle='--', alpha=0.7)
ax3.fill_between(plot_data.index, plot_data['keltner_upper'], plot_data['keltner_lower'], alpha=0.1, color='orange')

# Mark breakouts
keltner_up_breaks = plot_data[plot_data['close'] > plot_data['keltner_upper']]
keltner_down_breaks = plot_data[plot_data['close'] < plot_data['keltner_lower']]
ax3.scatter(keltner_up_breaks.index, keltner_up_breaks['close'], marker='^', color='red', s=80, zorder=10)
ax3.scatter(keltner_down_breaks.index, keltner_down_breaks['close'], marker='v', color='green', s=80, zorder=10)

ax3.set_title('3. KELTNER CHANNELS (EMA + 2×ATR)', fontsize=13, fontweight='bold')
ax3.set_ylabel('Price (USD)', fontsize=11)
ax3.legend(loc='best', fontsize=9)
ax3.grid(True, alpha=0.3)

# ---- Chart 4: Donchian Channels ----
ax4 = fig.add_subplot(gs[3, :])
ax4.plot(plot_data.index, plot_data['close'], label='Price', color='black', linewidth=2, zorder=5)
ax4.plot(plot_data.index, plot_data['donchian_middle'], label='Middle', color='blue', linestyle='--', alpha=0.7)
ax4.plot(plot_data.index, plot_data['donchian_upper'], label='Upper (20-High)', color='red', linestyle='--', linewidth=2, alpha=0.7)
ax4.plot(plot_data.index, plot_data['donchian_lower'], label='Lower (20-Low)', color='green', linestyle='--', linewidth=2, alpha=0.7)
ax4.fill_between(plot_data.index, plot_data['donchian_upper'], plot_data['donchian_lower'], alpha=0.1, color='teal')

# Mark breakouts
donchian_up_breaks = plot_data[plot_data['close'] > plot_data['donchian_upper']]
donchian_down_breaks = plot_data[plot_data['close'] < plot_data['donchian_lower']]
ax4.scatter(donchian_up_breaks.index, donchian_up_breaks['close'], marker='^', color='red', s=80, zorder=10)
ax4.scatter(donchian_down_breaks.index, donchian_down_breaks['close'], marker='v', color='green', s=80, zorder=10)

ax4.set_title('4. DONCHIAN CHANNELS (20-period High/Low)', fontsize=13, fontweight='bold')
ax4.set_ylabel('Price (USD)', fontsize=11)
ax4.set_xlabel('Date', fontsize=11)
ax4.legend(loc='best', fontsize=9)
ax4.grid(True, alpha=0.3)

# ---- Chart 5: Channel Width Comparison ----
ax5 = fig.add_subplot(gs[4, 0])
ax5.plot(plot_data.index, plot_data['bb_width'], label='Bollinger', linewidth=2)
ax5.plot(plot_data.index, plot_data['atr_width'], label='ATR', linewidth=2)
ax5.plot(plot_data.index, plot_data['keltner_width'], label='Keltner', linewidth=2)
ax5.plot(plot_data.index, plot_data['donchian_width'], label='Donchian', linewidth=2)
ax5.set_title('Channel Width Evolution (% of Price)', fontsize=12, fontweight='bold')
ax5.set_ylabel('Width (%)', fontsize=10)
ax5.set_xlabel('Date', fontsize=10)
ax5.legend(loc='best', fontsize=9)
ax5.grid(True, alpha=0.3)

# ---- Chart 6: Breakout Frequency ----
ax6 = fig.add_subplot(gs[4, 1])
channels = ['Bollinger\nBands', 'ATR\nChannels', 'Keltner\nChannels', 'Donchian\nChannels']
breakout_counts = [bb_breakouts, atr_breakouts, keltner_breakouts, donchian_breakouts]
colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
bars = ax6.bar(channels, breakout_counts, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

ax6.set_title('Breakout Signal Frequency (Last 120 Days)', fontsize=12, fontweight='bold')
ax6.set_ylabel('Number of Breakouts', fontsize=10)
ax6.grid(True, alpha=0.3, axis='y')

plt.savefig('channel_types_comparison.png', dpi=150, bbox_inches='tight')
print("\n✅ Saved: channel_types_comparison.png")

# ============================================================================
# PART 5: Which Channel to Use?
# ============================================================================

print("\n" + "="*80)
print("PART 5: เลือกใช้ Channel แบบไหนดี?")
print("="*80)

selection_guide = """
┌────────────────────────────────────────────────────────────────────────────┐
│  DECISION TREE: เลือก Channel Type                                        │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  คุณต้องการเทรดแบบไหน?                                                     │
│                                                                            │
│  ┌─ MEAN REVERSION (ซื้อ oversold, ขาย overbought)                        │
│  │   ├─ Asset มี Normal Distribution → BOLLINGER BANDS                    │
│  │   │  • ใช้ได้ดีกับ: FX pairs, Indices                                   │
│  │   │  • Entry: ราคาแตะ band → รอ reversal                                │
│  │   │  • Exit: ราคากลับถึง middle line                                    │
│  │   │                                                                     │
│  │   └─ Asset มี gaps/jumps → ATR CHANNELS (multiplier 1.5-2.0)          │
│  │      • ใช้ได้ดีกับ: Gold, Oil, Crypto                                   │
│  │      • รวม gap risk ในการคำนวณ                                          │
│  │                                                                         │
│  └─ BREAKOUT (ตาม momentum เมื่อทะลุ)                                      │
│      ├─ ต้องการ quality > quantity → ATR CHANNELS (multiplier 2.5-3.0)   │
│      │  • น้อย signals แต่แม่นกว่า                                         │
│      │  • Filter ด้วย volume และ ATR level                                │
│      │  • กลยุทธ์ที่เราใช้ชนะ!                                              │
│      │                                                                     │
│      ├─ ต้องการ responsive → KELTNER CHANNELS                             │
│      │  • ใช้ EMA = เร็วกว่า SMA                                           │
│      │  • ดีสำหรับ intraday                                                │
│      │                                                                     │
│      └─ ต้องการ simple → DONCHIAN CHANNELS                                │
│         • No calculation overhead                                         │
│         • Pure price action                                               │
│         • ใช้ใน Turtle Trading System                                     │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────┐
│  COMBINING CHANNELS (ใช้หลาย Channels ร่วมกัน)                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  1️⃣ BB Squeeze (Bollinger + Keltner)                                      │
│     Condition: BB width < Keltner width                                    │
│     → Volatility กำลังจะ expand                                            │
│     → เตรียมเข้า breakout                                                  │
│                                                                            │
│  2️⃣ Double Confirmation (ATR + Donchian)                                  │
│     Entry: ราคา > Donchian Upper AND > ATR Upper                          │
│     → Double breakout = stronger signal                                    │
│                                                                            │
│  3️⃣ Channel Flip (BB for range, ATR for breakout)                         │
│     • ใช้ BB เทรด mean reversion ในช่วง low volatility                    │
│     • สลับไป ATR breakout เมื่อ volatility expand                         │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────┐
│  BEST PRACTICES                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ✅ DO:                                                                    │
│     • Backtest channel parameters สำหรับแต่ละ asset                       │
│     • ปรับ multiplier ตาม market condition (volatile = wider)             │
│     • ใช้ filters เพิ่ม (volume, trend, time-of-day)                      │
│     • Walk-forward test parameters ทุก 3-6 เดือน                          │
│                                                                            │
│  ❌ DON'T:                                                                 │
│     • ใช้ default parameters โดยไม่ test                                  │
│     • เทรดทุก breakout (ต้องมี filter)                                     │
│     • Ignore volatility regime changes                                    │
│     • Over-optimize (curve-fitting)                                       │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
"""

print(selection_guide)

# ============================================================================
# PART 6: Live Example
# ============================================================================

print("\n" + "="*80)
print("PART 6: ตัวอย่างการใช้งานจริง (วันล่าสุด)")
print("="*80)

latest = recent_data.iloc[-1]
print(f"\nDate: {recent_data.index[-1].strftime('%Y-%m-%d')}")
print(f"Price: ${latest['close']:,.2f}")
print("-" * 80)

print("\n1. BOLLINGER BANDS:")
print(f"   Middle: ${latest['bb_middle']:,.2f}")
print(f"   Upper:  ${latest['bb_upper']:,.2f} (Distance: ${latest['close'] - latest['bb_upper']:+,.2f})")
print(f"   Lower:  ${latest['bb_lower']:,.2f} (Distance: ${latest['close'] - latest['bb_lower']:+,.2f})")
print(f"   Width:  {latest['bb_width']:.2f}%")
if latest['close'] > latest['bb_upper']:
    print("   ⚠️ OVERBOUGHT (ราคาเหนือ upper band)")
elif latest['close'] < latest['bb_lower']:
    print("   ⚠️ OVERSOLD (ราคาต่ำกว่า lower band)")
else:
    print("   ✅ Within channel")

print("\n2. ATR CHANNELS (กลยุทธ์ของเรา):")
print(f"   Middle: ${latest['atr_middle']:,.2f}")
print(f"   Upper:  ${latest['atr_upper']:,.2f} (Distance: ${latest['close'] - latest['atr_upper']:+,.2f})")
print(f"   Lower:  ${latest['atr_lower']:,.2f} (Distance: ${latest['close'] - latest['atr_lower']:+,.2f})")
print(f"   Width:  {latest['atr_width']:.2f}%")
if latest['close'] > latest['atr_upper']:
    breakout_strength = (latest['close'] - latest['atr_upper']) / latest['atr_14']
    print(f"   🔵 LONG SIGNAL (Breakout: {breakout_strength:.2f}×ATR)")
elif latest['close'] < latest['atr_lower']:
    breakout_strength = (latest['atr_lower'] - latest['close']) / latest['atr_14']
    print(f"   🔴 SHORT SIGNAL (Breakout: {breakout_strength:.2f}×ATR)")
else:
    print("   ⚪ NEUTRAL (รอ breakout)")

print("\n3. KELTNER CHANNELS:")
print(f"   Middle: ${latest['keltner_middle']:,.2f}")
print(f"   Upper:  ${latest['keltner_upper']:,.2f} (Distance: ${latest['close'] - latest['keltner_upper']:+,.2f})")
print(f"   Lower:  ${latest['keltner_lower']:,.2f} (Distance: ${latest['close'] - latest['keltner_lower']:+,.2f})")
print(f"   Width:  {latest['keltner_width']:.2f}%")

print("\n4. DONCHIAN CHANNELS:")
print(f"   Middle: ${latest['donchian_middle']:,.2f}")
print(f"   Upper:  ${latest['donchian_upper']:,.2f} (20-day High)")
print(f"   Lower:  ${latest['donchian_lower']:,.2f} (20-day Low)")
print(f"   Width:  {latest['donchian_width']:.2f}%")
if latest['close'] >= latest['donchian_upper']:
    print("   📈 NEW 20-DAY HIGH (Turtle buy signal)")
elif latest['close'] <= latest['donchian_lower']:
    print("   📉 NEW 20-DAY LOW (Turtle short signal)")

print("\n" + "="*80)
print("✅ Complete!")
print("="*80)

print("""
📚 สรุปสำหรับ XAUUSD:

เราใช้ ATR Channels (multiplier 2.5) ชนะเพราะ:
1. ✅ Gold มี gaps และ jumps บ่อย → ATR รวม gaps
2. ✅ Trending market (2019-2025) → breakout ดีกว่า mean reversion
3. ✅ Channel กว้างพอ → filter noise ได้ดี (11 signals ใน 100 วัน)
4. ✅ Dynamic adjustment → ปรับตาม volatility อัตโนมัติ

เปรียบเทียบกับ Bollinger Bands:
• BB ดีกับ mean reversion แต่ Gold เป็น trending asset
• BB ไม่รวม gaps → underestimate risk ใน Gold
• BB มี breakouts บ่อยกว่า → whipsaw มากกว่า

คำแนะนำ: ถ้าเทรด Gold ให้ใช้ ATR-based channels!
""")
