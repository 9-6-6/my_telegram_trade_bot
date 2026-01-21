# 🤖 AI-Powered Trading Bot with Telegram Integration

A sophisticated, AI-powered trading bot that provides **autonomous trading signals** for forex, commodities, metals, indices, and cryptocurrencies using machine learning, technical analysis, and sentiment analysis.

![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)
![Telegram](https://img.shields.io/badge/Telegram-Bot%20API-blue.svg)
![ML](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)
![Status](https://img.shields.io/badge/Status-Active-green.svg)

---

## ✨ Features

### ⚡ NEW: Scalp AI Engine
- **AI-Powered Scalping**: Dedicated engine for short-term trades (5min-1hour)
- **Pattern Recognition**: Momentum bursts, RSI reversals, Bollinger bounces, MACD crossovers
- **38 ML Features**: Extended feature set for scalp-specific predictions
- **85%+ Confidence**: Only high-probability scalp signals
- **92%+ Premium Signals**: Ultra-high confidence for best opportunities
- **Quick Targets**: Optimized SL/TP for fast profit capture

### 🤖 Autonomous AI Signal Generator
- **Machine Learning Model**: GradientBoostingClassifier with 200 estimators
- **24 ML Features** for prediction including RSI, MACD, Bollinger Bands, EMA crossovers
- **Self-Learning**: Continuous model improvement with each market scan
- **High-Confidence Scalp Signals**: Only sends signals with 90%+ confidence
- **Multi-Timeframe Analysis**: Analyzes 5m, 15m, 1h, and 4h charts simultaneously

### 📰 News-Based Signal Engine
- **Real-time News Scraping** from financial sources
- **Sentiment Analysis** using NLP techniques
- **News Impact Assessment** on market movements
- **Category-based Analysis**: Forex, Metals, Indices, Crypto news

### 📊 Technical Analysis
- **TradingView Integration**: Real-time technical indicators
- **Support & Resistance Levels**: Auto-calculated key levels
- **Multi-Indicator Analysis**: RSI, MACD, Bollinger Bands, EMA, SMA, ATR
- **Trend Strength Assessment**: Measure directional momentum

### 🎯 Smart Signal Features
- **Precise Entry Points**: Calculated based on market conditions
- **Risk Management**: Automatic Stop Loss & Take Profit levels
- **Risk/Reward Ratio**: Calculated for each signal
- **Session Awareness**: London, New York, Asian session timing

---

## 📈 Supported Markets (116+ Symbols)

### 💱 Forex Pairs (50+)
| Major | Minor | Exotic |
|-------|-------|--------|
| EURUSD | EURGBP | USDMXN |
| GBPUSD | EURJPY | USDZAR |
| USDJPY | GBPJPY | USDTRY |
| AUDUSD | AUDNZD | EURTRY |
| USDCAD | EURCHF | USDSEK |
| NZDUSD | GBPCHF | USDNOK |
| USDCHF | CADJPY | USDPLN |

### 🥇 Precious Metals
| Symbol | Description |
|--------|-------------|
| XAUUSD | Gold/USD |
| XAGUSD | Silver/USD |
| XPTUSD | Platinum/USD |
| XPDUSD | Palladium/USD |

### 🛢️ Energy & Commodities
| Symbol | Description |
|--------|-------------|
| USOIL / WTIUSD | WTI Crude Oil |
| UKOIL / BRENT | Brent Crude Oil |
| NATGAS | Natural Gas |
| COPPER | Copper |

### 📊 Global Indices (25+)
| Americas | Europe | Asia-Pacific |
|----------|--------|--------------|
| US30 (Dow Jones) | UK100 (FTSE) | JPN225 (Nikkei) |
| NAS100 (Nasdaq) | GER40 (DAX) | AUS200 (ASX) |
| SPX500 (S&P 500) | FRA40 (CAC) | HK50 (Hang Seng) |
| US2000 (Russell) | EU50 (Stoxx) | CHINA50 |

### ₿ Cryptocurrencies
| Symbol | Description |
|--------|-------------|
| BTCUSD | Bitcoin |
| ETHUSD | Ethereum |
| BNBUSD | Binance Coin |
| XRPUSD | Ripple |
| SOLUSD | Solana |
| ADAUSD | Cardano |
| DOGEUSD | Dogecoin |

---

## 🚀 Installation

### Prerequisites
- Python 3.12+
- UV Package Manager (recommended) or pip

### Step 1: Clone Repository
```bash
git clone <repository-url>
cd my_telegram_trade_bot
```

### Step 2: Create Virtual Environment
```bash
# Using UV (recommended)
uv venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Or using pip
python -m venv .venv
.venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
# Using UV
uv pip install -r requirements.txt

# Or using pip
pip install -r requirements.txt
```

### Step 4: Configure Environment
Create a `.env` file in the root directory:
```env
# Telegram Bot Token (Get from @BotFather)
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here

# News API Key (Get from https://newsapi.org)
NEWS_API_KEY=your_news_api_key_here

# Optional: Chat ID for notifications
TELEGRAM_CHAT_ID=your_chat_id_here
```

### Step 5: Run the Bot
```bash
python copy_trading_bot.py
```

---

## 💬 Telegram Commands

### Basic Commands
| Command | Description |
|---------|-------------|
| `/start` | Welcome message & quick start guide |
| `/start SYMBOL` | Get instant signal (e.g., `/start EURUSD`) |
| `/help` | Show all available commands |

### Signal Commands
| Command | Description |
|---------|-------------|
| `/newsignal SYMBOL` | Get AI-powered signal with news analysis |
| `/signal SYMBOL` | Get technical analysis signal |
| `/symbols` | List all 116+ supported symbols |

### ⚡ Scalp AI Commands (NEW!)
| Command | Description |
|---------|-------------|
| `/scalp SYMBOL` | Get AI scalp signal for quick trades |
| `/scalpscan` | **Start autonomous SCALP scanning** - 45s intervals |
| `/stopscalp` | Stop scalp scanning |

### 🔥 Autonomous Scanner
| Command | Description |
|---------|-------------|
| `/autoscan` | **Start autonomous AI scanning** - Bot scans markets every 60s and sends high-confidence signals automatically |
| `/stopscan` | Stop autonomous scanning |

### Analysis Commands
| Command | Description |
|---------|-------------|
| `/analyze SYMBOL` | Deep technical analysis |
| `/sentiment SYMBOL` | News sentiment analysis |
| `/levels SYMBOL` | Support & resistance levels |

### Settings
| Command | Description |
|---------|-------------|
| `/settings` | View/modify bot settings |
| `/risk` | Risk management settings |

---

## ⚡ Scalp AI Engine Architecture

### Pattern Detection System
```
┌─────────────────────────────────────────────────────────┐
│              SCALP PATTERN DETECTOR                     │
├─────────────────────────────────────────────────────────┤
│  Detected Patterns:                                     │
│  ├── ⚡ Momentum Burst                                  │
│  ├── 🔄 RSI Reversal (Oversold/Overbought)            │
│  ├── 📊 Bollinger Bounce (Upper/Lower Band)           │
│  ├── 📈 MACD Crossover (Bullish/Bearish)              │
│  ├── 📉 EMA Pullback (Trend Continuation)             │
│  ├── 💚 Support Bounce                                 │
│  └── ❤️ Resistance Rejection                           │
└─────────────────────────────────────────────────────────┘
```

### Scalp AI ML Model (38 Features)
```
┌─────────────────────────────────────────────────────────┐
│              GradientBoostingClassifier                 │
│                   (300 estimators)                      │
├─────────────────────────────────────────────────────────┤
│  Momentum Features:                                     │
│  ├── RSI (5m, 15m, 1h) + Slope + Acceleration          │
│  MACD Features:                                         │
│  ├── Line, Signal, Histogram + Slope + Crossover       │
│  Bollinger Features:                                    │
│  ├── Position, Width, Squeeze Detection                │
│  EMA Features:                                          │
│  ├── Distance from EMA20/50, Crossovers, Alignment     │
│  Pattern Features:                                      │
│  ├── Count, Avg Confidence, Best Score, Confluence     │
├─────────────────────────────────────────────────────────┤
│  Output: SCALP Signal + AI Score (0-100)               │
│  ├── Premium Scalp (92%+ confidence) 🔥                │
│  └── Standard Scalp (85%+ confidence) ⚡               │
└─────────────────────────────────────────────────────────┘
```

---

## 🤖 AI/ML Architecture

### Machine Learning Model
```
┌─────────────────────────────────────────────────────────┐
│              GradientBoostingClassifier                 │
│                   (200 estimators)                      │
├─────────────────────────────────────────────────────────┤
│  24 Input Features:                                     │
│  ├── RSI (5m, 15m, 1h, 4h)                             │
│  ├── MACD (line, signal, histogram)                    │
│  ├── Bollinger Bands (position, width)                 │
│  ├── EMA/SMA Crossovers                                │
│  ├── Volume Analysis                                   │
│  ├── ATR Volatility                                    │
│  ├── Support/Resistance Proximity                      │
│  ├── News Sentiment Score                              │
│  └── Trading Session Score                             │
├─────────────────────────────────────────────────────────┤
│  Output: Signal Type + Confidence Score                 │
│  ├── SCALP_BUY/SELL (90%+ confidence)                  │
│  ├── STRONG_BUY/SELL (85%+ confidence)                 │
│  └── BUY/SELL (75%+ confidence)                        │
└─────────────────────────────────────────────────────────┘
```

### Signal Confidence Thresholds
| Signal Type | Required Confidence | Description |
|-------------|---------------------|-------------|
| PREMIUM SCALP | **92%+** | Ultra-high probability quick trades |
| SCALP | **85%+** | High-probability quick trades |
| STRONG | 85%+ | Strong directional moves |
| REGULAR | 75%+ | Standard trading signals |

---

## 📁 Project Structure

```
my_telegram_trade_bot/
├── copy_trading_bot.py          # Main bot with all integrations
├── scalp_ai_engine.py           # ⚡ NEW: Scalp AI Engine
├── autonomous_signal_generator.py # AI autonomous scanner
├── news_signal_engine.py        # News-based signal engine
├── news_signal_integration.py   # Integration layer
├── trading_bot.py               # Original trading bot
├── trading_agent.py             # Trading agent logic
├── requirements.txt             # Python dependencies
├── .env                         # Environment variables
├── README.md                    # This file
├── logs/                        # Log files
├── models/                      # Saved ML models
│   ├── scalp_ai_model.joblib    # Scalp AI ML model
│   ├── scalp_ai_scaler.joblib   # Scalp feature scaler
│   ├── signal_predictor.joblib  # Main signal model
│   └── signal_scaler.joblib     # Main feature scaler
└── src/
    ├── bot.py                   # Bot utilities
    ├── config.py                # Configuration
    └── setup_bot.py             # Bot setup
```

---

## ⚡ Sample Scalp Signal Output

```
⚡ PREMIUM SCALP SIGNAL ⚡
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🟢 BUY EURUSD

📈 Entry: 1.08500
🛑 Stop Loss: 1.08420
🎯 Take Profit:
   • TP1: 1.08580 (Quick)
   • TP2: 1.08660 (Main)
   • TP3: 1.08740 (Extended)

📊 AI Analysis:
   • AI Score: 94.5/100
   • Confidence: 93.2%
   • Momentum: 85%
   • Pattern Score: 91%
   • R:R Ratio: 1.85

⏱️ Expected Duration: 5-30 minutes
📍 Pip Target: ~8.0 pips

🔍 Analysis:
   🔍 Found 3 pattern(s)
   • ⚡ Momentum Burst: 88%
   • 🔄 RSI Reversal: 85%
   • 📈 MACD Crossover: 82%
   ⏰ Optimal trading session (EU-US overlap)
   ✅ R:R Ratio: 1.85
   🔥 PREMIUM SCALP SIGNAL!
```

---

## 📊 Sample Regular Signal Output

```
🎯 SCALP SIGNAL - EURUSD

📊 Signal: SCALP_BUY
💪 Confidence: 94.5%
⏰ Time: 2026-01-20 14:30:00 UTC

💰 Entry: 1.0845
🛑 Stop Loss: 1.0825 (-20 pips)
🎯 Take Profit: 1.0885 (+40 pips)
📈 Risk/Reward: 1:2.0

📰 News Sentiment: Bullish (0.72)
📊 Technical: Strong Buy
🕐 Session: London/NY Overlap

⚠️ Risk Warning: Trade responsibly!
```

---

## ⚙️ Configuration Options

### Environment Variables
| Variable | Description | Required |
|----------|-------------|----------|
| `TELEGRAM_BOT_TOKEN` | Your Telegram bot token | ✅ Yes |
| `NEWS_API_KEY` | NewsAPI.org API key | Optional |
| `TELEGRAM_CHAT_ID` | Default chat for signals | Optional |

### Customizable Settings
- Minimum confidence threshold (default: 75%)
- Scalp confidence threshold (default: 90%)
- Scan interval (default: 60 seconds)
- Risk/Reward ratio requirements
- Symbols to monitor

---

## 🔧 Dependencies

```
python-telegram-bot>=22.5
tradingview-ta>=3.3.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
tensorflow>=2.15.0
beautifulsoup4>=4.12.0
requests>=2.31.0
python-dotenv>=1.0.0
aiohttp>=3.9.0
apscheduler>=3.10.0
pytz>=2023.3
joblib>=1.3.0
```

---

## ⚠️ Risk Warning

**IMPORTANT: Trading involves significant financial risk.**

- This bot provides signals for **informational purposes only**
- Past performance does not guarantee future results
- Always do your own research (DYOR)
- Never trade with money you cannot afford to lose
- Consider using a demo account first
- The developers are not responsible for any financial losses

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit:
- Bug reports
- Feature requests
- Pull requests
- Documentation improvements

---

## 📜 License

This project is for educational purposes. Use at your own risk.

---

## 📞 Support

- Create an issue on GitHub
- Contact via Telegram bot

---

**Made with ❤️ for traders** 