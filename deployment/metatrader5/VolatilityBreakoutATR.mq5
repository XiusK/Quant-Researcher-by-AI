//+------------------------------------------------------------------+
//|                                     VolatilityBreakoutATR.mq5    |
//|                               Developed by Quant Researcher AI   |
//|                                                                  |
//+------------------------------------------------------------------+
//| Expert Advisor: Volatility Breakout ATR Strategy                |
//| Mathematical Foundation:                                         |
//|   - ATR-based dynamic channel (2.5x multiplier)                  |
//|   - Entry: Price breaks above/below channel                      |
//|   - Exit: Price returns to channel or opposite signal            |
//|   - Risk Management: Inverse volatility position sizing          |
//|                                                                  |
//| Strategy Performance (XAUUSD 2019-2025):                         |
//|   - Sharpe Ratio: 0.37                                           |
//|   - Total Return: +6.34%                                         |
//|   - Max Drawdown: -3.53%                                         |
//|   - Trades: 365                                                  |
//+------------------------------------------------------------------+

#property copyright "Quant Researcher AI"
#property version   "1.00"
#property strict

//+------------------------------------------------------------------+
//| Input Parameters                                                 |
//+------------------------------------------------------------------+

// Channel Parameters
input group "========== Channel Settings =========="
input int      InpChannelLength    = 20;      // Channel Length (MA Period)
input int      InpATRPeriod        = 14;      // ATR Period
input double   InpATRMultiplier    = 2.5;     // ATR Multiplier

// Filter Parameters
input group "========== Filter Settings =========="
input bool     InpUseVolFilter     = true;    // Enable Volatility Filter
input double   InpMinVolatility    = 12.0;    // Minimum Volatility (% annual)
input bool     InpUseTimeExit      = false;   // Enable Time-Based Exit
input int      InpTimeExitBars     = 5;       // Exit After N Bars

// Exit Management
input group "========== Exit Management =========="
input double   InpMinProfitGate    = 1.0;     // Min Profit Before Middle Exit (x ATR)
input double   InpProfitTarget     = 3.0;     // Take Profit Target (x ATR)
input double   InpTrailingStart    = 1.5;     // Start Trailing After (x ATR)
input double   InpTrailingDistance = 1.0;     // Trailing Stop Distance (x ATR)

// Position Sizing
input group "========== Risk Management =========="
input bool     InpUseMoneyManage   = true;    // Position Size Mode: true=% Risk | false=Fixed Lot
input double   InpRiskPercent      = 2.0;     // [% Risk Mode] Risk Per Trade (%)
input double   InpLotSize          = 0.1;     // [Fixed Mode] Lot Size
input double   InpMaxPosition      = 25.0;    // [% Risk Mode] Max Position Size (%)
input double   InpTargetVol        = 15.0;    // [% Risk Mode] Target Portfolio Vol (%)
input bool     InpUseInverseVol    = true;    // [% Risk Mode] Inverse Volatility Sizing

// Trading Settings
input group "========== Trading Settings =========="
input bool     InpDebugMode       = false;    // Enable Debug Messages
input int      InpMagicNumber      = 202602;  // Magic Number
input string   InpTradeComment     = "VB-ATR";// Trade Comment
input int      InpSlippage         = 30;      // Slippage (points)

// Trading Hours
input group "========== Trading Hours =========="
input bool     InpUseTradingHours  = false;   // Enable Trading Hours Filter
input int      InpStartHour        = 0;       // Trading Start Hour
input int      InpEndHour          = 23;      // Trading End Hour

//+------------------------------------------------------------------+
//| Global Variables                                                 |
//+------------------------------------------------------------------+

double    g_atr[];
double    g_channel_middle[];
double    g_channel_upper[];
double    g_channel_lower[];
double    g_realized_vol[];

int       g_atr_handle;
int       g_ma_handle;

datetime  g_last_bar_time = 0;
int       g_bars_in_trade = 0;
datetime  g_entry_time = 0;
double    g_entry_price = 0;
double    g_trailing_stop = 0;
double    g_entry_atr = 0;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    // Create indicator handles
    g_atr_handle = iATR(_Symbol, PERIOD_CURRENT, InpATRPeriod);
    g_ma_handle = iMA(_Symbol, PERIOD_CURRENT, InpChannelLength, 0, MODE_SMA, PRICE_CLOSE);
    
    if(g_atr_handle == INVALID_HANDLE || g_ma_handle == INVALID_HANDLE)
    {
        Print("Error creating indicator handles");
        return(INIT_FAILED);
    }
    
    // Set array as series
    ArraySetAsSeries(g_atr, true);
    ArraySetAsSeries(g_channel_middle, true);
    ArraySetAsSeries(g_channel_upper, true);
    ArraySetAsSeries(g_channel_lower, true);
    ArraySetAsSeries(g_realized_vol, true);
    
    // Initialize last bar time to previous bar (will trigger on first new bar)
    g_last_bar_time = iTime(_Symbol, PERIOD_CURRENT, 1);
    
    // Check for existing positions and recover data
    CheckExistingPosition();
    
    Print("====================================");
    Print("Volatility Breakout ATR EA Initialized");
    Print("Symbol: ", _Symbol);
    Print("Period: ", PeriodToString(PERIOD_CURRENT));
    Print("Channel Length: ", InpChannelLength);
    Print("ATR Period: ", InpATRPeriod);
    Print("ATR Multiplier: ", InpATRMultiplier);
    Print("Position Size Mode: ", (InpUseMoneyManage ? "% Risk" : "Fixed Lot"));
    if(InpUseMoneyManage)
        Print("Risk Per Trade: ", InpRiskPercent, "% | Target Vol: ", InpTargetVol, "%");
    else
        Print("Fixed Lot Size: ", InpLotSize);
    Print("NOTE: Settlement time 23:00-02:00 will be skipped");
    Print("====================================" );
    
    return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    // Release indicator handles
    if(g_atr_handle != INVALID_HANDLE)
        IndicatorRelease(g_atr_handle);
    if(g_ma_handle != INVALID_HANDLE)
        IndicatorRelease(g_ma_handle);
    
    // Remove channel lines
    ObjectDelete(0, "VB_Upper");
    ObjectDelete(0, "VB_Middle");
    ObjectDelete(0, "VB_Lower");
    Comment("");
    
    Print("Volatility Breakout ATR EA Deinitialized");
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
    // Check for new bar
    datetime current_bar_time = iTime(_Symbol, PERIOD_CURRENT, 0);
    if(current_bar_time == g_last_bar_time)
        return;
    
    g_last_bar_time = current_bar_time;
    
    // Update indicators
    if(!UpdateIndicators())
    {
        if(InpDebugMode)
            Print("DEBUG: Failed to update indicators");
        return;
    }
    
    // Check trading hours (includes settlement time check)
    if(!IsWithinTradingHours())
        return;
    
    // Check if market is actually open
    if(!IsMarketOpen())
        return;
    
    // Get current price
    double current_price = iClose(_Symbol, PERIOD_CURRENT, 0);
    
    // Calculate volatility filter
    bool vol_filter_passed = CheckVolatilityFilter();
    
    // Debug print volatility status
    if(InpDebugMode && !vol_filter_passed)
        Print("DEBUG: Vol filter FAILED - Current: ", g_realized_vol[0], "% | Min required: ", InpMinVolatility, "%");
    
    // Check positions
    int position_type = GetPositionType();
    
    // Entry Logic
    if(position_type == 0)  // No position
    {
        // Debug: Print channel values once per bar
        if(InpDebugMode)
        {
            static datetime last_debug = 0;
            if(current_bar_time != last_debug)
            {
                Print("DEBUG: Price=", current_price, " | Middle=", g_channel_middle[0], 
                      " | Upper=", g_channel_upper[0], " | Lower=", g_channel_lower[0]);
                
                // Safe ATR percentage calculation (avoid zero divide)
                string atr_info = "DEBUG: ATR=" + DoubleToString(g_atr[0], _Digits);
                if(current_price > 0 && g_atr[0] > 0)
                {
                    double atr_pct = (g_atr[0] / current_price) * 100;
                    atr_info += " (ATR%=" + DoubleToString(atr_pct, 2) + "%)";
                }
                else
                {
                    atr_info += " (ATR%=N/A)";
                }
                
                string vol_info = " | Vol=";
                if(ArraySize(g_realized_vol) > 0)
                    vol_info += DoubleToString(g_realized_vol[0], 1) + "%";
                else
                    vol_info += "N/A";
                
                Print(atr_info + vol_info);
                last_debug = current_bar_time;
            }
        }
        
        // Safety check: ATR must be valid
        if(g_atr[0] <= 0)
        {
            Print("WARNING: Invalid ATR value: ", g_atr[0]);
            return;
        }
        
        // Long Entry
        if(current_price > g_channel_upper[0] && vol_filter_passed)
        {
            // Double-check no position exists (safety)
            if(GetPositionType() != 0)
            {
                Print("WARNING: Position already exists, skipping new entry");
                return;
            }
            
            double confidence = (current_price - g_channel_upper[0]) / g_atr[0];
            Print("========== LONG SIGNAL ==========");
            Print("Price: ", current_price, " > Upper Band: ", g_channel_upper[0]);
            Print("Confidence: ", confidence, "x ATR");
            OpenPosition(ORDER_TYPE_BUY, confidence);
        }
        // Short Entry
        else if(current_price < g_channel_lower[0] && vol_filter_passed)
        {
            // Double-check no position exists (safety)
            if(GetPositionType() != 0)
            {
                Print("WARNING: Position already exists, skipping new entry");
                return;
            }
            
            double confidence = (g_channel_lower[0] - current_price) / g_atr[0];
            Print("========== SHORT SIGNAL ==========");
            Print("Price: ", current_price, " < Lower Band: ", g_channel_lower[0]);
            Print("Confidence: ", confidence, "x ATR");
            OpenPosition(ORDER_TYPE_SELL, confidence);
        }
    }
    // Exit Logic
    else
    {
        g_bars_in_trade++;
        
        // Safety check: Entry ATR must be valid
        if(g_entry_atr <= 0 || g_entry_price <= 0)
        {
            Print("ERROR: Invalid entry data - ATR: ", g_entry_atr, " Price: ", g_entry_price);
            Print("Closing position due to invalid entry data");
            ClosePosition("Invalid Entry Data");
            return;
        }
        
        // Calculate profit in ATR units
        double profit_atr = 0;
        if(position_type == 1)  // Long
            profit_atr = (current_price - g_entry_price) / g_entry_atr;
        else  // Short
            profit_atr = (g_entry_price - current_price) / g_entry_atr;
        
        bool should_exit = false;
        string exit_reason = "";
        
        // Exit Long
        if(position_type == 1)
        {
            // 1. Profit Target - Take profit at 3x ATR
            if(profit_atr >= InpProfitTarget)
            {
                should_exit = true;
                exit_reason = StringFormat("Profit Target: %.2fx ATR (Target: %.1fx)", profit_atr, InpProfitTarget);
            }
            // 2. Trailing Stop - Activate after 1.5x ATR profit
            else if(profit_atr >= InpTrailingStart)
            {
                double new_trailing = current_price - (InpTrailingDistance * g_entry_atr);
                if(g_trailing_stop == 0 || new_trailing > g_trailing_stop)
                {
                    g_trailing_stop = new_trailing;
                    Print("Trailing stop updated: ", DoubleToString(g_trailing_stop, _Digits), " (Profit: ", DoubleToString(profit_atr, 2), "x ATR)");
                }
                
                if(current_price <= g_trailing_stop)
                {
                    should_exit = true;
                    exit_reason = StringFormat("Trailing Stop: %.2fx ATR profit locked", profit_atr);
                }
            }
            // 3. Middle Line Exit - Only if profit >= 1x ATR
            else if(profit_atr >= InpMinProfitGate && current_price <= g_channel_middle[0])
            {
                should_exit = true;
                exit_reason = StringFormat("Middle Exit: %.2fx ATR profit (Min: %.1fx)", profit_atr, InpMinProfitGate);
            }
            // 4. Emergency Exit - Price broke to opposite side
            else if(current_price < g_channel_lower[0])
            {
                should_exit = true;
                exit_reason = StringFormat("Emergency: Broke lower band (%.2fx ATR)", profit_atr);
            }
            // 5. Time-based Exit
            else if(InpUseTimeExit && g_bars_in_trade >= InpTimeExitBars)
            {
                should_exit = true;
                exit_reason = StringFormat("Time Exit: %d bars (%.2fx ATR)", g_bars_in_trade, profit_atr);
            }
        }
        // Exit Short
        else if(position_type == -1)
        {
            // 1. Profit Target
            if(profit_atr >= InpProfitTarget)
            {
                should_exit = true;
                exit_reason = StringFormat("Profit Target: %.2fx ATR (Target: %.1fx)", profit_atr, InpProfitTarget);
            }
            // 2. Trailing Stop
            else if(profit_atr >= InpTrailingStart)
            {
                double new_trailing = current_price + (InpTrailingDistance * g_entry_atr);
                if(g_trailing_stop == 0 || new_trailing < g_trailing_stop)
                {
                    g_trailing_stop = new_trailing;
                    Print("Trailing stop updated: ", DoubleToString(g_trailing_stop, _Digits), " (Profit: ", DoubleToString(profit_atr, 2), "x ATR)");
                }
                
                if(current_price >= g_trailing_stop)
                {
                    should_exit = true;
                    exit_reason = StringFormat("Trailing Stop: %.2fx ATR profit locked", profit_atr);
                }
            }
            // 3. Middle Line Exit
            else if(profit_atr >= InpMinProfitGate && current_price >= g_channel_middle[0])
            {
                should_exit = true;
                exit_reason = StringFormat("Middle Exit: %.2fx ATR profit (Min: %.1fx)", profit_atr, InpMinProfitGate);
            }
            // 4. Emergency Exit
            else if(current_price > g_channel_upper[0])
            {
                should_exit = true;
                exit_reason = StringFormat("Emergency: Broke upper band (%.2fx ATR)", profit_atr);
            }
            // 5. Time-based Exit
            else if(InpUseTimeExit && g_bars_in_trade >= InpTimeExitBars)
            {
                should_exit = true;
                exit_reason = StringFormat("Time Exit: %d bars (%.2fx ATR)", g_bars_in_trade, profit_atr);
            }
        }
        
        if(should_exit)
        {
            ClosePosition(exit_reason);
        }
    }
    
    // Display info on chart
    DisplayInfo();
    
    // Draw channel lines
    DrawChannelLines();
}

//+------------------------------------------------------------------+
//| Update Indicators                                                |
//+------------------------------------------------------------------+
bool UpdateIndicators()
{
    // Copy ATR values
    if(CopyBuffer(g_atr_handle, 0, 0, 3, g_atr) < 0)
    {
        Print("Error copying ATR buffer");
        return false;
    }
    
    // Copy MA values
    if(CopyBuffer(g_ma_handle, 0, 0, 3, g_channel_middle) < 0)
    {
        Print("Error copying MA buffer");
        return false;
    }
    
    // Calculate channel bands
    ArrayResize(g_channel_upper, 3);
    ArrayResize(g_channel_lower, 3);
    ArraySetAsSeries(g_channel_upper, true);
    ArraySetAsSeries(g_channel_lower, true);
    
    for(int i = 0; i < 3; i++)
    {
        g_channel_upper[i] = g_channel_middle[i] + (InpATRMultiplier * g_atr[i]);
        g_channel_lower[i] = g_channel_middle[i] - (InpATRMultiplier * g_atr[i]);
    }
    
    // Calculate realized volatility
    CalculateRealizedVolatility();
    
    return true;
}

//+------------------------------------------------------------------+
//| Calculate Realized Volatility                                    |
//+------------------------------------------------------------------+
void CalculateRealizedVolatility()
{
    int lookback = 20;
    double log_returns[];
    ArrayResize(log_returns, lookback);
    
    // Get close prices
    double close_prices[];
    ArraySetAsSeries(close_prices, true);
    
    if(CopyClose(_Symbol, PERIOD_CURRENT, 0, lookback + 1, close_prices) <= 0)
        return;
    
    // Calculate log returns
    double sum = 0;
    for(int i = 0; i < lookback; i++)
    {
        log_returns[i] = MathLog(close_prices[i] / close_prices[i + 1]);
        sum += log_returns[i];
    }
    
    // Calculate standard deviation
    double mean = sum / lookback;
    double variance = 0;
    for(int i = 0; i < lookback; i++)
    {
        variance += MathPow(log_returns[i] - mean, 2);
    }
    variance /= (lookback - 1);
    
    double daily_vol = MathSqrt(variance);
    double annual_vol = daily_vol * MathSqrt(252) * 100;  // Annualized %
    
    ArrayResize(g_realized_vol, 1);
    ArraySetAsSeries(g_realized_vol, true);
    g_realized_vol[0] = annual_vol;
}

//+------------------------------------------------------------------+
//| Check Volatility Filter                                          |
//+------------------------------------------------------------------+
bool CheckVolatilityFilter()
{
    if(!InpUseVolFilter)
        return true;
    
    if(ArraySize(g_realized_vol) == 0)
        return false;
    
    return g_realized_vol[0] >= InpMinVolatility;
}

//+------------------------------------------------------------------+
//| Check Trading Hours                                              |
//+------------------------------------------------------------------+
bool IsWithinTradingHours()
{
    // Settlement time check only applies to intraday timeframes
    // Daily/Weekly/Monthly charts use end-of-day data, so skip this check
    if(PERIOD_CURRENT < PERIOD_D1)
    {
        MqlDateTime dt;
        TimeCurrent(dt);
        
        // Avoid daily settlement time (23:00-02:00) for Gold intraday trading
        if(dt.hour >= 23 || dt.hour < 2)
            return false;
    }
    
    // Check custom trading hours if enabled
    if(!InpUseTradingHours)
        return true;
    
    MqlDateTime dt;
    TimeCurrent(dt);
    
    if(InpStartHour <= InpEndHour)
        return (dt.hour >= InpStartHour && dt.hour <= InpEndHour);
    else
        return (dt.hour >= InpStartHour || dt.hour <= InpEndHour);
}

//+------------------------------------------------------------------+
//| Check if Market is Open                                          |
//+------------------------------------------------------------------+
bool IsMarketOpen()
{
    // Check if we have bid/ask prices (most reliable check)
    double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
    
    if(bid == 0 || ask == 0)
    {
        if(InpDebugMode)
            Print("DEBUG: No quotes - Market likely closed (Bid: ", bid, ", Ask: ", ask, ")");
        return false;
    }
    
    // Check spread is reasonable (not too wide indicating thin liquidity)
    double spread = ask - bid;
    double spread_pct = (spread / bid) * 100;
    
    if(spread_pct > 0.5)  // If spread > 0.5%, market may be unstable
    {
        if(InpDebugMode)
            Print("DEBUG: Spread too wide: ", DoubleToString(spread_pct, 3), "% - Market may be closing");
        return false;
    }
    
    return true;
}

//+------------------------------------------------------------------+
//| Check and Recover Existing Position Data                         |
//+------------------------------------------------------------------+
void CheckExistingPosition()
{
    if(!PositionSelect(_Symbol))
    {
        Print("No existing position found");
        return;
    }
    
    if(PositionGetInteger(POSITION_MAGIC) != InpMagicNumber)
    {
        Print("Position found but different magic number: ", PositionGetInteger(POSITION_MAGIC));
        return;
    }
    
    // Position exists with our magic number - recover data
    ENUM_POSITION_TYPE pos_type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
    g_entry_price = PositionGetDouble(POSITION_PRICE_OPEN);
    g_entry_time = (datetime)PositionGetInteger(POSITION_TIME);
    
    // Calculate entry ATR from current ATR (fallback if we can't recover exact value)
    // First update indicators to get current ATR
    if(CopyBuffer(g_atr_handle, 0, 0, 3, g_atr) > 0)
    {
        g_entry_atr = g_atr[0];  // Use current ATR as approximation
    }
    else
    {
        g_entry_atr = 0;
        Print("WARNING: Could not recover ATR value");
    }
    
    // Calculate bars in trade
    datetime current_time = TimeCurrent();
    int total_bars = Bars(_Symbol, PERIOD_CURRENT, g_entry_time, current_time);
    g_bars_in_trade = (total_bars > 0) ? total_bars : 0;
    
    string direction = (pos_type == POSITION_TYPE_BUY) ? "LONG" : "SHORT";
    Print("====================================");
    Print("RECOVERED EXISTING POSITION");
    Print("Type: ", direction);
    Print("Entry Price: ", g_entry_price);
    Print("Entry Time: ", TimeToString(g_entry_time));
    Print("Estimated Entry ATR: ", g_entry_atr);
    Print("Bars in Trade: ", g_bars_in_trade);
    Print("Volume: ", PositionGetDouble(POSITION_VOLUME));
    Print("Current P/L: ", PositionGetDouble(POSITION_PROFIT));
    Print("====================================");
}

//+------------------------------------------------------------------+
//| Get Position Type                                                |
//| Returns: 0 = no position, 1 = long, -1 = short                   |
//+------------------------------------------------------------------+
int GetPositionType()
{
    if(PositionSelect(_Symbol))
    {
        if(PositionGetInteger(POSITION_MAGIC) == InpMagicNumber)
        {
            ENUM_POSITION_TYPE type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
            if(type == POSITION_TYPE_BUY)
                return 1;
            else if(type == POSITION_TYPE_SELL)
                return -1;
        }
    }
    return 0;
}

//+------------------------------------------------------------------+
//| Calculate Position Size                                          |
//+------------------------------------------------------------------+
double CalculatePositionSize(double confidence)
{
    double lot_size = InpLotSize;
    double min_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
    double max_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
    double lot_step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
    
    if(InpUseMoneyManage)
    {
        // Safety check: ATR must be valid
        if(g_atr[0] <= 0)
        {
            Print("ERROR: Invalid ATR for position sizing: ", g_atr[0], " | Using fixed lot: ", InpLotSize);
            return MathMax(min_lot, InpLotSize);
        }
        
        // Get account balance
        double balance = AccountInfoDouble(ACCOUNT_BALANCE);
        
        // Calculate risk amount
        double risk_amount = balance * (InpRiskPercent / 100.0);
        
        // Calculate lot size based on ATR stop loss
        double stop_distance = g_atr[0] * InpATRMultiplier;
        double tick_value = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
        double tick_size = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
        double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
        
        double stop_in_points = stop_distance / point;
        
        // Safety check: prevent division by zero
        if(stop_in_points <= 0 || tick_value <= 0 || tick_size <= 0)
        {
            Print("ERROR: Invalid calculation parameters - stop_points: ", stop_in_points, 
                  " tick_value: ", tick_value, " tick_size: ", tick_size);
            return MathMax(min_lot, InpLotSize);
        }
        
        double calculated_lot = risk_amount / (stop_in_points * tick_value / tick_size);
        
        // Check if calculated lot is too small (below minimum)
        if(calculated_lot < min_lot)
        {
            Print("WARNING: Calculated lot ", DoubleToString(calculated_lot, 4), 
                  " < Minimum lot ", min_lot, " | Using fixed lot: ", InpLotSize);
            Print("SUGGESTION: Increase Risk% from ", InpRiskPercent, "% to at least ", 
                  DoubleToString((min_lot / calculated_lot) * InpRiskPercent, 1), "%");
            return MathMax(min_lot, InpLotSize);  // Use fixed lot as fallback
        }
        
        lot_size = calculated_lot;
        
        // Inverse volatility sizing
        if(InpUseInverseVol && ArraySize(g_realized_vol) > 0 && g_realized_vol[0] > 0)
        {
            double vol_multiplier = MathMin(InpTargetVol / g_realized_vol[0], 2.0);
            lot_size *= vol_multiplier;
        }
        
        // Apply confidence adjustment
        lot_size *= MathMin(confidence, 2.0);
        
        // Normalize lot size
        lot_size = MathFloor(lot_size / lot_step) * lot_step;
        lot_size = MathMax(min_lot, MathMin(lot_size, max_lot));
        
        // Apply max position limit
        double max_lot_from_balance = balance * (InpMaxPosition / 100.0) / 
                                     (SymbolInfoDouble(_Symbol, SYMBOL_ASK) * 
                                      SymbolInfoDouble(_Symbol, SYMBOL_TRADE_CONTRACT_SIZE));
        lot_size = MathMin(lot_size, max_lot_from_balance);
        
        // Final check: if still below minimum after all adjustments, use fixed lot
        if(lot_size < min_lot)
        {
            Print("WARNING: Final lot ", lot_size, " < min lot after adjustments. Using: ", InpLotSize);
            lot_size = MathMax(min_lot, InpLotSize);
        }
    }
    else
    {
        // Fixed lot mode - just ensure it's within bounds
        lot_size = MathMax(min_lot, MathMin(InpLotSize, max_lot));
    }
    
    return lot_size;
}

//+------------------------------------------------------------------+
//| Open Position                                                     |
//+------------------------------------------------------------------+
void OpenPosition(ENUM_ORDER_TYPE order_type, double confidence)
{
    MqlTradeRequest request = {};
    MqlTradeResult result = {};
    
    // Calculate position size
    double lot_size = CalculatePositionSize(confidence);
    
    // Auto-detect filling mode
    ENUM_ORDER_TYPE_FILLING filling = GetFillingMode();
    
    // Setup order request
    request.action = TRADE_ACTION_DEAL;
    request.symbol = _Symbol;
    request.volume = lot_size;
    request.type = order_type;
    request.deviation = InpSlippage;
    request.magic = InpMagicNumber;
    request.comment = StringFormat("%s|C:%.2f|A:%.2f", InpTradeComment, confidence, g_atr[0]);
    request.type_filling = filling;
    
    if(order_type == ORDER_TYPE_BUY)
    {
        request.price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
    }
    else
    {
        request.price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    }
    
    Print("Sending order: ", (order_type == ORDER_TYPE_BUY ? "BUY" : "SELL"), 
          " | Volume: ", lot_size, " | Price: ", request.price, " | Filling: ", filling);
    
    // Send order
    if(!OrderSend(request, result))
    {
        Print("OrderSend error: ", GetLastError());
        Print("Request: ", request.action, " ", request.symbol, " ", request.volume, " @ ", request.price);
    }
    else
    {
        if(result.retcode == TRADE_RETCODE_DONE)
        {
            g_entry_time = TimeCurrent();
            g_bars_in_trade = 0;
            g_entry_price = result.price;
            g_entry_atr = g_atr[0];
            g_trailing_stop = 0;  // Reset trailing stop
            
            string direction = (order_type == ORDER_TYPE_BUY) ? "LONG" : "SHORT";
            Print("====================================");
            Print("Position Opened: ", direction);
            Print("Price: ", result.price);
            Print("Volume: ", result.volume);
            Print("Confidence: ", DoubleToString(confidence, 2), "x ATR");
            Print("Entry ATR: ", DoubleToString(g_entry_atr, _Digits));
            Print("====================================");
        }
        else
        {
            Print("OrderSend failed: ", result.retcode);
        }
    }
}

//+------------------------------------------------------------------+
//| Close Position                                                    |
//+------------------------------------------------------------------+
void ClosePosition(string reason)
{
    if(!PositionSelect(_Symbol))
        return;
    
    if(PositionGetInteger(POSITION_MAGIC) != InpMagicNumber)
        return;
    
    MqlTradeRequest request = {};
    MqlTradeResult result = {};
    
    request.action = TRADE_ACTION_DEAL;
    request.symbol = _Symbol;
    request.volume = PositionGetDouble(POSITION_VOLUME);
    request.deviation = InpSlippage;
    request.magic = InpMagicNumber;
    request.comment = reason;
    request.type_filling = GetFillingMode();
    
    ENUM_POSITION_TYPE pos_type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
    
    if(pos_type == POSITION_TYPE_BUY)
    {
        request.price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
        request.type = ORDER_TYPE_SELL;
    }
    else
    {
        request.price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
        request.type = ORDER_TYPE_BUY;
    }
    
    request.position = PositionGetInteger(POSITION_TICKET);
    
    if(!OrderSend(request, result))
    {
        Print("ClosePosition error: ", GetLastError());
    }
    else
    {
        if(result.retcode == TRADE_RETCODE_DONE)
        {
            double profit = PositionGetDouble(POSITION_PROFIT);
            
            Print("====================================");
            Print("Position Closed: ", reason);
            Print("Profit: ", DoubleToString(profit, 2));
            Print("Bars in Trade: ", g_bars_in_trade);
            Print("====================================");
            
            g_bars_in_trade = 0;
            g_entry_price = 0;
            g_trailing_stop = 0;
            g_entry_atr = 0;
        }
    }
}

//+------------------------------------------------------------------+
//| Draw Channel Lines on Chart                                      |
//+------------------------------------------------------------------+
void DrawChannelLines()
{
    if(ArraySize(g_channel_upper) == 0 || ArraySize(g_channel_middle) == 0 || ArraySize(g_channel_lower) == 0)
        return;
    
    datetime time_current = iTime(_Symbol, PERIOD_CURRENT, 0);
    datetime time_prev = iTime(_Symbol, PERIOD_CURRENT, 1);
    
    // Draw Upper Band
    string upper_name = "VB_Upper";
    if(ObjectFind(0, upper_name) < 0)
    {
        ObjectCreate(0, upper_name, OBJ_TREND, 0, time_prev, g_channel_upper[1], time_current, g_channel_upper[0]);
        ObjectSetInteger(0, upper_name, OBJPROP_COLOR, clrLime);
        ObjectSetInteger(0, upper_name, OBJPROP_WIDTH, 2);
        ObjectSetInteger(0, upper_name, OBJPROP_STYLE, STYLE_SOLID);
        ObjectSetInteger(0, upper_name, OBJPROP_RAY_RIGHT, false);
    }
    else
    {
        ObjectMove(0, upper_name, 0, time_prev, g_channel_upper[1]);
        ObjectMove(0, upper_name, 1, time_current, g_channel_upper[0]);
    }
    
    // Draw Middle Line
    string middle_name = "VB_Middle";
    if(ObjectFind(0, middle_name) < 0)
    {
        ObjectCreate(0, middle_name, OBJ_TREND, 0, time_prev, g_channel_middle[1], time_current, g_channel_middle[0]);
        ObjectSetInteger(0, middle_name, OBJPROP_COLOR, clrYellow);
        ObjectSetInteger(0, middle_name, OBJPROP_WIDTH, 1);
        ObjectSetInteger(0, middle_name, OBJPROP_STYLE, STYLE_DOT);
        ObjectSetInteger(0, middle_name, OBJPROP_RAY_RIGHT, false);
    }
    else
    {
        ObjectMove(0, middle_name, 0, time_prev, g_channel_middle[1]);
        ObjectMove(0, middle_name, 1, time_current, g_channel_middle[0]);
    }
    
    // Draw Lower Band
    string lower_name = "VB_Lower";
    if(ObjectFind(0, lower_name) < 0)
    {
        ObjectCreate(0, lower_name, OBJ_TREND, 0, time_prev, g_channel_lower[1], time_current, g_channel_lower[0]);
        ObjectSetInteger(0, lower_name, OBJPROP_COLOR, clrRed);
        ObjectSetInteger(0, lower_name, OBJPROP_WIDTH, 2);
        ObjectSetInteger(0, lower_name, OBJPROP_STYLE, STYLE_SOLID);
        ObjectSetInteger(0, lower_name, OBJPROP_RAY_RIGHT, false);
    }
    else
    {
        ObjectMove(0, lower_name, 0, time_prev, g_channel_lower[1]);
        ObjectMove(0, lower_name, 1, time_current, g_channel_lower[0]);
    }
    
    ChartRedraw();
}

//+------------------------------------------------------------------+
//| Display Info on Chart                                            |
//+------------------------------------------------------------------+
void DisplayInfo()
{
    string info = "";
    
    info += "========== Volatility Breakout ATR ==========\n";
    info += "Mode: " + (InpUseMoneyManage ? "% Risk" : "Fixed Lot") + "\n";
    info += "ATR: " + DoubleToString(g_atr[0], _Digits) + "\n";
    
    if(ArraySize(g_realized_vol) > 0)
        info += "Realized Vol: " + DoubleToString(g_realized_vol[0], 1) + "%\n";
    
    double channel_width = (g_channel_upper[0] - g_channel_lower[0]) / iClose(_Symbol, PERIOD_CURRENT, 0) * 100;
    info += "Channel Width: " + DoubleToString(channel_width, 1) + "%\n";
    
    info += "Upper Band: " + DoubleToString(g_channel_upper[0], _Digits) + "\n";
    info += "Middle: " + DoubleToString(g_channel_middle[0], _Digits) + "\n";
    info += "Lower Band: " + DoubleToString(g_channel_lower[0], _Digits) + "\n";
    
    int pos_type = GetPositionType();
    if(pos_type == 1)
    {
        double current_price = iClose(_Symbol, PERIOD_CURRENT, 0);
        
        // Safety check before division
        if(g_entry_atr > 0)
        {
            double profit_atr = (current_price - g_entry_price) / g_entry_atr;
            info += "\nPosition: LONG (Bars: " + IntegerToString(g_bars_in_trade) + ")";
            info += "\nProfit: " + DoubleToString(profit_atr, 2) + "x ATR";
        }
        else
        {
            info += "\nPosition: LONG (Bars: " + IntegerToString(g_bars_in_trade) + ")";
            info += "\nProfit: N/A (Invalid ATR)";
        }
        
        if(g_trailing_stop > 0)
            info += "\nTrailing: " + DoubleToString(g_trailing_stop, _Digits);
    }
    else if(pos_type == -1)
    {
        double current_price = iClose(_Symbol, PERIOD_CURRENT, 0);
        
        // Safety check before division
        if(g_entry_atr > 0)
        {
            double profit_atr = (g_entry_price - current_price) / g_entry_atr;
            info += "\nPosition: SHORT (Bars: " + IntegerToString(g_bars_in_trade) + ")";
            info += "\nProfit: " + DoubleToString(profit_atr, 2) + "x ATR";
        }
        else
        {
            info += "\nPosition: SHORT (Bars: " + IntegerToString(g_bars_in_trade) + ")";
            info += "\nProfit: N/A (Invalid ATR)";
        }
        
        if(g_trailing_stop > 0)
            info += "\nTrailing: " + DoubleToString(g_trailing_stop, _Digits);
    }
    else
        info += "\nPosition: FLAT";
    
    bool vol_filter = CheckVolatilityFilter();
    info += "\nVol Filter: " + (vol_filter ? "PASS" : "FAIL");
    
    Comment(info);
}

//+------------------------------------------------------------------+
//| Get Filling Mode (Auto-detect for broker)                       |
//+------------------------------------------------------------------+
ENUM_ORDER_TYPE_FILLING GetFillingMode()
{
    // Get allowed filling modes for symbol
    int filling = (int)SymbolInfoInteger(_Symbol, SYMBOL_FILLING_MODE);
    
    // Prefer IOC > FOK > RETURN
    if((filling & SYMBOL_FILLING_IOC) == SYMBOL_FILLING_IOC)
        return ORDER_FILLING_IOC;
    else if((filling & SYMBOL_FILLING_FOK) == SYMBOL_FILLING_FOK)
        return ORDER_FILLING_FOK;
    else
        return ORDER_FILLING_RETURN;
}

//+------------------------------------------------------------------+
//| Period to String Helper                                          |
//+------------------------------------------------------------------+
string PeriodToString(ENUM_TIMEFRAMES period)
{
    switch(period)
    {
        case PERIOD_M1:  return "M1";
        case PERIOD_M5:  return "M5";
        case PERIOD_M15: return "M15";
        case PERIOD_M30: return "M30";
        case PERIOD_H1:  return "H1";
        case PERIOD_H4:  return "H4";
        case PERIOD_H8:  return "H8";
        case PERIOD_H12: return "H12";
        case PERIOD_D1:  return "D1";
        case PERIOD_W1:  return "W1";
        case PERIOD_MN1: return "MN1";
        default: 
            return StringFormat("Custom(%d)", PeriodSeconds(period) / 60);
    }
}
