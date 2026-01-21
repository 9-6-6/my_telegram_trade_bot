# Trading Bot System - Complete User Guide

## Table of Contents
1. [System Overview](#system-overview)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Starting the Bots](#starting-the-bots)
5. [Stopping the Bots](#stopping-the-bots)
6. [How Signals Work](#how-signals-work)
7. [Telegram Bot Commands](#telegram-bot-commands)
8. [XM360 Auto Trader Settings](#xm360-auto-trader-settings)
9. [Advanced Features (v2.0)](#advanced-features-v20)
10. [BUY vs SELL Trade Logic](#buy-vs-sell-trade-logic)
11. [Troubleshooting](#troubleshooting)

---

## System Overview

This system consists of **two bots** that work together:

```
┌─────────────────────────────────────────────────────────────────┐
│                      TRADING BOT SYSTEM                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    Signal    ┌──────────────────┐            │
│  │              │ ──────────▶  │                  │            │
│  │  TELEGRAM    │              │   XM360 AUTO     │            │
│  │    BOT       │              │     TRADER       │            │
│  │              │              │                  │            │
│  └──────┬───────┘              └────────┬─────────┘            │
│         │                               │                       │
│         ▼                               ▼                       │
│  ┌──────────────┐              ┌──────────────────┐            │
│  │   Telegram   │              │   MT5 Desktop    │            │
│  │   Messages   │              │   (XM360 Broker) │            │
│  └──────────────┘              └──────────────────┘            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Bot 1: Telegram Bot (`trading_bot.py`)
- Analyzes market data (Gold, Bitcoin, etc.)
- Generates trading signals with Entry, Stop Loss, Take Profit
- Sends signals to Telegram subscribers
- Forwards signals to XM360 Auto Trader

### Bot 2: XM360 Auto Trader (`xm360_auto_trader/auto_trader.py`)
- Receives signals from Telegram Bot
- Validates price (checks if signal price is current)
- Manages risk (10% of balance limit)
- Places real trades on XM360 broker via MT5

---

## Prerequisites

### Required Software
| Software | Purpose | Download |
|----------|---------|----------|
| Python 3.12+ | Run the bots | [python.org](https://python.org) |
| MetaTrader 5 | Connect to XM360 | [xm.com/mt5](https://www.xm.com/mt5) |
| Telegram App | Receive signals | [telegram.org](https://telegram.org) |

### Required Accounts
| Account | Purpose | How to Get |
|---------|---------|------------|
| Telegram Bot Token | Send messages | Create via [@BotFather](https://t.me/BotFather) |
| XM360 Account | Place trades | [xm.com](https://www.xm.com) |

### Python Packages
```bash
pip install python-telegram-bot MetaTrader5 yfinance requests
```

---

## Installation

### Step 1: Clone/Download the Project
```bash
git clone <your-repo-url>
cd my_telegram_trade_bot
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
pip install MetaTrader5
```

### Step 3: Configure Telegram Bot
Edit `src/config.py`:
```python
BOT_TOKEN = "your-telegram-bot-token-here"
```

### Step 4: Configure XM360 Auto Trader
Edit `xm360_auto_trader/config.py`:
```python
# Your XM360 Account Details
ACCOUNT = 315982803           # Your account number
PASSWORD = "Gadhiya@098"      # Your password
SERVER = "XMGlobal-MT5 7"     # Your server

# Risk Settings
MAX_BALANCE_USAGE_PERCENT = 10.0  # Use max 10% of balance
MIN_BALANCE_TO_TRADE = 100.0      # Minimum $100 required
```

### Step 5: Install & Login to MT5
1. Download MT5 from [xm.com/mt5](https://www.xm.com/mt5)
2. Install and open MT5
3. Login: File → Login to Trade Account
4. Enter your XM360 credentials
5. Enable Algo Trading: Tools → Options → Expert Advisors → ✅ Allow algorithmic trading

---

## Starting the Bots

### Method 1: Start Both Bots Together (Recommended)

**Windows - Double-click:**
```
start_all.bat
```

**Or from Terminal:**
```bash
# Terminal 1 - Start Telegram Bot
python trading_bot.py

# Terminal 2 - Start XM360 Auto Trader
python xm360_auto_trader/auto_trader.py
```

### Method 2: Start Bots Separately

#### Start Telegram Bot Only
```bash
cd my_telegram_trade_bot
python trading_bot.py
```
**What you'll see:**
```
Trading Bot Started!
Monitoring: XAUUSD, BTCUSD
Press Ctrl+C to stop...
```

#### Start XM360 Auto Trader Only
```bash
cd my_telegram_trade_bot/xm360_auto_trader
python auto_trader.py
```
**What you'll see:**
```
========================================
   XM360 Auto Trader Started
========================================
Account: 315982803
Balance: $10,000.00
Mode: LIVE TRADING
Waiting for signals...
```

### Startup Checklist
- [ ] MT5 Desktop is open and logged in
- [ ] Algo Trading is enabled in MT5
- [ ] Internet connection is stable
- [ ] Both terminals are running

---

## Stopping the Bots

### Method 1: Keyboard Shortcut
In each terminal window, press:
```
Ctrl + C
```

### Method 2: Close Terminal Windows
Simply close the terminal/command prompt windows.

### Method 3: Task Manager (Force Stop)
1. Press `Ctrl + Shift + Esc`
2. Find `python.exe` processes
3. Click "End Task"

### Safe Shutdown Procedure
1. Stop **XM360 Auto Trader** first (stops new trades)
2. Wait for any pending trades to complete
3. Stop **Telegram Bot**
4. Close MT5 Desktop (optional)

---

## How Signals Work

### Signal Flow Diagram
```
┌────────────────────────────────────────────────────────────────────┐
│                        SIGNAL FLOW                                  │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. ANALYSIS                                                        │
│     ┌─────────────┐                                                │
│     │ Trading Bot │ ─── Analyzes XAUUSD, BTCUSD prices             │
│     │ (Python)    │ ─── Calculates entry, SL, TP                   │
│     └──────┬──────┘                                                │
│            │                                                        │
│            ▼                                                        │
│  2. SIGNAL GENERATED                                                │
│     ┌─────────────────────────────────────┐                        │
│     │ 🔔 GOLD SIGNAL                       │                        │
│     │ Direction: BUY                       │                        │
│     │ Entry: $2,650.50                     │                        │
│     │ Stop Loss: $2,645.00                 │                        │
│     │ Take Profit: $2,660.00               │                        │
│     └──────┬──────────────────────────────┘                        │
│            │                                                        │
│            ├──────────────────┬────────────────────┐               │
│            │                  │                    │               │
│            ▼                  ▼                    ▼               │
│  3. DELIVERY                                                        │
│     ┌──────────┐      ┌─────────────┐      ┌─────────────┐        │
│     │ Telegram │      │ Signal      │      │ Log File    │        │
│     │ Message  │      │ Queue JSON  │      │ (Backup)    │        │
│     └──────────┘      └──────┬──────┘      └─────────────┘        │
│                              │                                      │
│                              ▼                                      │
│  4. AUTO TRADING                                                    │
│     ┌─────────────────────────────────────┐                        │
│     │ XM360 Auto Trader                    │                        │
│     │ ├─ Reads signal from queue           │                        │
│     │ ├─ Validates current price           │                        │
│     │ ├─ Checks balance (10% rule)         │                        │
│     │ └─ Places trade on MT5               │                        │
│     └──────┬──────────────────────────────┘                        │
│            │                                                        │
│            ▼                                                        │
│  5. TRADE EXECUTED                                                  │
│     ┌─────────────────────────────────────┐                        │
│     │ MT5 / XM360 Broker                   │                        │
│     │ ├─ Order placed: BUY 0.01 XAUUSD     │                        │
│     │ ├─ SL: $2,645.00                     │                        │
│     │ └─ TP: $2,660.00                     │                        │
│     └─────────────────────────────────────┘                        │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

### Signal Validation Rules

Before placing a trade, the Auto Trader checks:

| Check | Rule | Example |
|-------|------|---------|
| **Price Validation** | Signal price must be within $5 of current price (Gold) | Signal: $2650, Current: $2651 ✅ |
| **Balance Check** | Must have at least $100 in account | Balance: $500 ✅ |
| **Risk Limit** | Only use 10% of balance per trade | $10,000 balance → Max $1,000 at risk |
| **Market Hours** | Market must be open | Weekdays only for Forex |

### Example Signal Message (Telegram)
```
🔔 GOLD TRADING SIGNAL

📊 Asset: XAUUSD
📈 Direction: BUY
💰 Entry Price: $2,650.50
🛑 Stop Loss: $2,645.00
🎯 Take Profit: $2,660.00

⏰ Generated: 2026-01-21 14:30:00
📱 Auto-trade: Enabled
```

---

## Telegram Bot Commands

### User Commands
| Command | Description |
|---------|-------------|
| `/start` | Start the bot, get welcome message |
| `/subscribe` | Subscribe to receive trading signals |
| `/unsubscribe` | Stop receiving signals |
| `/status` | Check bot status and your subscription |
| `/price` | Get current prices (Gold, Bitcoin) |
| `/help` | Show all available commands |

### How to Subscribe
1. Open Telegram
2. Search for your bot (by username)
3. Send `/start`
4. Send `/subscribe`
5. You'll now receive signals automatically!

### Example Conversation
```
You: /start
Bot: Welcome to Trading Bot! Send /subscribe to receive signals.

You: /subscribe
Bot: ✅ You are now subscribed to trading signals!

You: /price
Bot: 📊 Current Prices:
     XAUUSD (Gold): $2,650.50
     BTCUSD (Bitcoin): $101,250.00

[Later, automatically]
Bot: 🔔 GOLD TRADING SIGNAL
     Direction: BUY
     Entry: $2,650.50
     ...
```

---

## XM360 Auto Trader Settings

### Configuration File: `xm360_auto_trader/config.py`

```python
# ===========================================
# XM360 ACCOUNT SETTINGS
# ===========================================
ACCOUNT = 315982803           # Your XM360 account number
PASSWORD = "Gadhiya@098"      # Your account password
SERVER = "XMGlobal-MT5 7"     # Your MT5 server

# ===========================================
# RISK MANAGEMENT SETTINGS
# ===========================================
MAX_BALANCE_USAGE_PERCENT = 10.0  # Max 10% of balance per trade
MIN_BALANCE_TO_TRADE = 100.0      # Minimum balance required
DEFAULT_LOT_SIZE = 0.01           # Default trade size

# ===========================================
# PRICE VALIDATION SETTINGS
# ===========================================
MAX_PRICE_DEVIATION_GOLD = 5.0    # Max $5 difference for Gold
MAX_PRICE_DEVIATION_CRYPTO = 100.0 # Max $100 difference for Crypto
MAX_PRICE_DEVIATION_FOREX = 0.0010 # Max 10 pips for Forex

# ===========================================
# TRADING ASSETS
# ===========================================
ENABLED_ASSETS = ["XAUUSD", "BTCUSD", "EURUSD"]
```

### Risk Strategy Explained

```
┌─────────────────────────────────────────────────────────────┐
│                    10% BALANCE RULE                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Account Balance: $10,000                                   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ ██████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │   │
│  │    10%                    90%                       │   │
│  │  Available              Protected                   │   │
│  │   $1,000                 $9,000                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ✅ Can trade with: $1,000 maximum                         │
│  ✅ Protected: $9,000 (never touched)                      │
│  ✅ If balance < $100: No trades allowed                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Advanced Features (v2.0)

### New in Version 2.0

The enhanced auto trader (`auto_trader_v2.py`) includes powerful automation features:

| Feature | Description |
|---------|-------------|
| **Pending Orders** | Trade activates only when price reaches signal entry |
| **Trailing Stop Loss** | SL automatically moves up as price moves in your favor |
| **Multiple Take Profits** | Close partial positions at different profit targets |
| **Full Automation** | No manual intervention needed |

### How Trailing Stop Works

```
┌────────────────────────────────────────────────────────────────────┐
│                    TRAILING STOP EXAMPLE (BUY XAUUSD)              │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Signal: BUY @ $2655                                               │
│  Initial SL: $2645                                                 │
│  Trailing: Activate after $5 profit, trail $3 behind               │
│                                                                     │
│  Step 1: Price at $2655 (Entry)                                    │
│  ┌──────────────────────────────────────────────────────┐          │
│  │ Price: $2655    SL: $2645    Status: Waiting         │          │
│  └──────────────────────────────────────────────────────┘          │
│                                                                     │
│  Step 2: Price rises to $2660 (+$5 profit - Trailing activates!)   │
│  ┌──────────────────────────────────────────────────────┐          │
│  │ Price: $2660    SL: $2657    Status: TRAILING 📈     │          │
│  └──────────────────────────────────────────────────────┘          │
│       SL moved from $2645 → $2657 (price - $3 trail)               │
│                                                                     │
│  Step 3: Price rises to $2670                                      │
│  ┌──────────────────────────────────────────────────────┐          │
│  │ Price: $2670    SL: $2667    Status: TRAILING 📈     │          │
│  └──────────────────────────────────────────────────────┘          │
│       SL moved from $2657 → $2667 (locks in $12 profit!)           │
│                                                                     │
│  Step 4: Price drops to $2667 (SL hit)                             │
│  ┌──────────────────────────────────────────────────────┐          │
│  │ TRADE CLOSED @ $2667   Profit: +$12 per lot 🎉       │          │
│  └──────────────────────────────────────────────────────┘          │
│                                                                     │
│  WITHOUT Trailing: Would have closed at original SL $2645 = -$10   │
│  WITH Trailing: Closed at $2667 = +$12 profit!                     │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

### How Multiple Take Profits Work

```
┌────────────────────────────────────────────────────────────────────┐
│                 MULTIPLE TAKE PROFIT EXAMPLE                        │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Signal: BUY XAUUSD @ $2655                                        │
│  Lot Size: 0.02                                                    │
│  TP1: $2665 (close 50%)                                            │
│  TP2: $2675 (close remaining 50%)                                  │
│                                                                     │
│  Step 1: Entry at $2655 with 0.02 lots                             │
│  ┌──────────────────────────────────────────────────────┐          │
│  │ Position: 0.02 lots    Status: Active                │          │
│  └──────────────────────────────────────────────────────┘          │
│                                                                     │
│  Step 2: Price reaches $2665 (TP1 hit)                             │
│  ┌──────────────────────────────────────────────────────┐          │
│  │ 🎯 TP1 HIT! Closed 0.01 lots (+$10 profit)           │          │
│  │ Remaining: 0.01 lots (still running)                 │          │
│  └──────────────────────────────────────────────────────┘          │
│                                                                     │
│  Step 3: Price reaches $2675 (TP2 hit)                             │
│  ┌──────────────────────────────────────────────────────┐          │
│  │ 🎯 TP2 HIT! Closed 0.01 lots (+$20 profit)           │          │
│  │ Trade fully closed!                                  │          │
│  └──────────────────────────────────────────────────────┘          │
│                                                                     │
│  RESULT: Total Profit = $10 + $20 = $30 🎉                         │
│                                                                     │
│  BENEFIT: Even if price reverses after TP1, you already            │
│           secured 50% of your profits!                             │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

### How Pending Orders Work

```
┌────────────────────────────────────────────────────────────────────┐
│                    PENDING ORDER EXAMPLE                            │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Current Price: $2650                                              │
│  Signal: BUY @ $2655 (signal wants entry at $2655)                 │
│                                                                     │
│  WITHOUT Pending Orders:                                           │
│  ┌──────────────────────────────────────────────────────┐          │
│  │ ❌ Trade opens immediately at $2650                  │          │
│  │    (5 dollars BEFORE the signal price!)              │          │
│  └──────────────────────────────────────────────────────┘          │
│                                                                     │
│  WITH Pending Orders:                                              │
│  ┌──────────────────────────────────────────────────────┐          │
│  │ ⏳ Pending order created: BUY STOP @ $2655           │          │
│  │    Waiting for price to reach $2655...               │          │
│  └──────────────────────────────────────────────────────┘          │
│                       ↓                                            │
│  ┌──────────────────────────────────────────────────────┐          │
│  │ Price reaches $2655...                               │          │
│  │ ✅ ACTIVATED! Trade opens exactly at signal price!   │          │
│  └──────────────────────────────────────────────────────┘          │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

### Configuration (auto_trader_v2.py)

Edit the configuration at the top of `xm360_auto_trader/auto_trader_v2.py`:

```python
# Trailing Stop Configuration
TRAILING_STOP_CONFIG = {
    "XAUUSD": {
        "enabled": True,
        "activation_profit_dollars": 5.0,   # Start trailing after $5 profit
        "trail_distance_dollars": 3.0,      # Keep SL $3 behind price
        "step_dollars": 1.0                 # Minimum $1 to update SL
    },
    "BTCUSD": {
        "enabled": True,
        "activation_profit_dollars": 100.0,
        "trail_distance_dollars": 50.0,
        "step_dollars": 20.0
    }
}

# Multiple Take Profit Configuration
TAKE_PROFIT_CONFIG = {
    "XAUUSD": {
        "use_multiple_tps": True,
        "tp_levels": [
            {"offset_dollars": 10.0, "percent_close": 50},   # TP1: +$10, close 50%
            {"offset_dollars": 20.0, "percent_close": 100},  # TP2: +$20, close rest
        ]
    }
}

# Use pending orders or immediate market orders
USE_PENDING_ORDERS = True
```

### Running Enhanced Version

```bash
# Run the enhanced v2.0 auto trader
python xm360_auto_trader/auto_trader_v2.py
```

Or update `START_ALL_BOTS.bat` to use v2:
```batch
python xm360_auto_trader\auto_trader_v2.py
```

---

## BUY vs SELL Trade Logic

All advanced features (pending orders, trailing stop, multiple TPs) work for **both BUY and SELL trades**.

### Comparison Table

```
┌─────────────────────┬──────────────────────┬──────────────────────┐
│      FEATURE        │        BUY           │        SELL          │
├─────────────────────┼──────────────────────┼──────────────────────┤
│ Profit Direction    │ Price goes UP ↑      │ Price goes DOWN ↓    │
├─────────────────────┼──────────────────────┼──────────────────────┤
│ Stop Loss Position  │ BELOW entry price    │ ABOVE entry price    │
├─────────────────────┼──────────────────────┼──────────────────────┤
│ Take Profit         │ ABOVE entry price    │ BELOW entry price    │
├─────────────────────┼──────────────────────┼──────────────────────┤
│ Trailing SL Moves   │ UP ↑ (follows        │ DOWN ↓ (follows      │
│                     │ rising price)        │ falling price)       │
├─────────────────────┼──────────────────────┼──────────────────────┤
│ Pending Order Type  │ BUY_STOP: price      │ SELL_STOP: price     │
│                     │ rises to entry       │ drops to entry       │
│                     │ BUY_LIMIT: price     │ SELL_LIMIT: price    │
│                     │ drops to entry       │ rises to entry       │
├─────────────────────┼──────────────────────┼──────────────────────┤
│ SL Triggers When    │ Price goes DOWN      │ Price goes UP        │
│                     │ below SL level       │ above SL level       │
├─────────────────────┼──────────────────────┼──────────────────────┤
│ TP Triggers When    │ Price goes UP        │ Price goes DOWN      │
│                     │ above TP level       │ below TP level       │
└─────────────────────┴──────────────────────┴──────────────────────┘
```

### BUY Trade Example

```
┌────────────────────────────────────────────────────────────────────┐
│                    BUY TRADE - FULL AUTOMATION                     │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Signal: BUY XAUUSD @ $2655                                        │
│  Current Price: $2650                                              │
│  Stop Loss: $2645 (below entry)                                    │
│  TP1: $2665 (close 50%)                                            │
│  TP2: $2675 (close remaining)                                      │
│  Trailing: Start after $5 profit, trail $3 behind                  │
│                                                                     │
│  Timeline:                                                         │
│  ─────────────────────────────────────────────────────────────     │
│                                                                     │
│  1. ⏳ PENDING: Price at $2650, waiting for $2655                  │
│                                                                     │
│  2. 🔔 ACTIVATED: Price reaches $2655, trade opens                 │
│     Entry: $2655 | SL: $2645 | Lots: 0.02                          │
│                                                                     │
│  3. 📈 PROFIT: Price rises to $2660 (+$5 profit)                   │
│     → Trailing ACTIVATES!                                          │
│     → SL moves UP: $2645 → $2657                                   │
│                                                                     │
│  4. 🎯 TP1 HIT: Price reaches $2665                                │
│     → Close 50% (0.01 lots)                                        │
│     → Profit locked: +$10                                          │
│     → Remaining: 0.01 lots                                         │
│                                                                     │
│  5. 📈 CONTINUE: Price rises to $2672                              │
│     → SL trails UP: $2657 → $2669                                  │
│     → Profit protected: $14 locked                                 │
│                                                                     │
│  6. 🎯 TP2 HIT: Price reaches $2675                                │
│     → Close remaining 50%                                          │
│     → Total profit: ~$30                                           │
│                                                                     │
│  ✅ RESULT: Trade fully closed with maximum profit!                │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

### SELL Trade Example

```
┌────────────────────────────────────────────────────────────────────┐
│                    SELL TRADE - FULL AUTOMATION                    │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Signal: SELL XAUUSD @ $2655                                       │
│  Current Price: $2660                                              │
│  Stop Loss: $2665 (above entry)                                    │
│  TP1: $2645 (close 50%)                                            │
│  TP2: $2635 (close remaining)                                      │
│  Trailing: Start after $5 profit, trail $3 behind                  │
│                                                                     │
│  Timeline:                                                         │
│  ─────────────────────────────────────────────────────────────     │
│                                                                     │
│  1. ⏳ PENDING: Price at $2660, waiting for $2655 (price to DROP)  │
│                                                                     │
│  2. 🔔 ACTIVATED: Price drops to $2655, trade opens                │
│     Entry: $2655 | SL: $2665 | Lots: 0.02                          │
│                                                                     │
│  3. 📉 PROFIT: Price drops to $2650 (+$5 profit)                   │
│     → Trailing ACTIVATES!                                          │
│     → SL moves DOWN: $2665 → $2653                                 │
│                                                                     │
│  4. 🎯 TP1 HIT: Price reaches $2645                                │
│     → Close 50% (0.01 lots)                                        │
│     → Profit locked: +$10                                          │
│     → Remaining: 0.01 lots                                         │
│                                                                     │
│  5. 📉 CONTINUE: Price drops to $2640                              │
│     → SL trails DOWN: $2653 → $2643                                │
│     → Profit protected: $12 locked                                 │
│                                                                     │
│  6. 🎯 TP2 HIT: Price reaches $2635                                │
│     → Close remaining 50%                                          │
│     → Total profit: ~$30                                           │
│                                                                     │
│  ✅ RESULT: Trade fully closed with maximum profit!                │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

### Trailing Stop Visual Comparison

```
BUY TRADE - SL moves UP as price rises:
─────────────────────────────────────────────────────────────────────
Price:    $2655 ─→ $2660 ─→ $2665 ─→ $2670 ─→ $2675
                    ↑
               Trailing starts
                    
SL:       $2645 ─→ $2657 ─→ $2662 ─→ $2667 ─→ $2672
                    ↑         ↑         ↑         ↑
              SL moves UP  UP again  UP again  UP again


SELL TRADE - SL moves DOWN as price falls:
─────────────────────────────────────────────────────────────────────
Price:    $2655 ─→ $2650 ─→ $2645 ─→ $2640 ─→ $2635
                    ↑
               Trailing starts
                    
SL:       $2665 ─→ $2653 ─→ $2648 ─→ $2643 ─→ $2638
                    ↑         ↑         ↑         ↑
            SL moves DOWN DOWN again DOWN again DOWN again
```

### What Happens If Price Reverses?

**BUY Trade:**
```
Entry: $2655, Price rose to $2670, SL trailed to $2667
→ Price suddenly drops to $2667
→ SL HIT! Trade closes at $2667
→ Profit: $12 (instead of loss if no trailing!)
```

**SELL Trade:**
```
Entry: $2655, Price dropped to $2640, SL trailed to $2643
→ Price suddenly rises to $2643
→ SL HIT! Trade closes at $2643
→ Profit: $12 (instead of loss if no trailing!)
```

**Without Trailing Stop:**
```
BUY Trade reversal → Would hit original SL $2645 → LOSS: -$10
SELL Trade reversal → Would hit original SL $2665 → LOSS: -$10

With Trailing Stop:
Both cases → Exit with PROFIT: +$12 ✅
```

---

## Troubleshooting

### Common Issues

#### 1. "MetaTrader 5 not found"
**Solution:**
- Install MT5 from [xm.com/mt5](https://www.xm.com/mt5)
- Make sure MT5 is **OPEN** before starting Auto Trader
- Login to your XM360 account in MT5

#### 2. "Could not connect to MT5"
**Solution:**
- Open MT5 Desktop application
- Login: File → Login to Trade Account
- Enable Algo Trading: Tools → Options → Expert Advisors

#### 3. "Trade rejected - Price too far from market"
**Solution:**
- This is **normal** - the signal price was stale
- The Auto Trader protects you from bad entries
- Wait for a fresh signal

#### 4. "Insufficient balance"
**Solution:**
- Your balance is below $100 minimum
- Deposit funds to your XM360 account
- Or adjust `MIN_BALANCE_TO_TRADE` in config

#### 5. Telegram bot not responding
**Solution:**
- Check if `trading_bot.py` is running
- Verify BOT_TOKEN in `src/config.py`
- Check internet connection

#### 6. Not receiving signals in Telegram
**Solution:**
- Send `/subscribe` to the bot
- Make sure you haven't muted the bot
- Check if bot is running

### Check System Status
Run this command to verify everything:
```bash
python xm360_auto_trader/setup_check.py
```

### View Logs
```bash
# Telegram Bot logs
cat logs/trading_bot.log

# Auto Trader logs
cat xm360_auto_trader/logs/auto_trader.log
```

---

## Quick Reference Card

### Start/Stop Commands
| Action | Command |
|--------|---------|
| Start Both Bots | Double-click `START_ALL_BOTS.bat` |
| Stop Both Bots | Double-click `STOP_ALL_BOTS.bat` or `Ctrl+C` |
| Start Telegram Only | `python trading_bot.py` |
| Start Auto Trader v2 | `python xm360_auto_trader/auto_trader_v2.py` |

### Stop Everything
```
Ctrl + C (in each terminal)
```

### Check Status
```bash
# In Telegram
/status

# Check MT5 connection
python xm360_auto_trader/setup_check.py
```

### Important Files
| File | Purpose |
|------|---------|
| `trading_bot.py` | Main Telegram bot |
| `xm360_auto_trader/auto_trader_v2.py` | Enhanced auto trader (v2.0) |
| `xm360_auto_trader/advanced_trade_manager.py` | Trailing stop, TPs logic |
| `xm360_auto_trader/config.py` | XM360 account settings |
| `src/config.py` | Telegram bot settings |

### Feature Summary (v2.0)
| Feature | BUY Trade | SELL Trade |
|---------|-----------|------------|
| Pending Orders | ✅ Waits for price UP | ✅ Waits for price DOWN |
| Trailing Stop | Moves UP ↑ | Moves DOWN ↓ |
| Multiple TPs | Above entry | Below entry |
| 10% Balance Rule | ✅ | ✅ |
| Price Validation | ✅ | ✅ |

### Trailing Stop Cheat Sheet
```
BUY:  Profit when Price UP   → SL moves UP   → Locks profit
SELL: Profit when Price DOWN → SL moves DOWN → Locks profit
```

---

## Support

If you encounter issues:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review log files in `logs/` and `xm360_auto_trader/logs/` folders
3. Run `python xm360_auto_trader/setup_check.py` to diagnose
4. Restart both bots

---

*Last Updated: January 21, 2026*
*Version: 2.0.0 - Enhanced with Trailing Stop, Multiple TPs, Pending Orders for BUY & SELL*
