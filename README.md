# Trading Bot with Telegram Integration

A comprehensive trading bot that provides signals for forex, commodities, metals, and oils using TradingView technical analysis and market news.

## Features

- Real-time trading signals for multiple markets
- Technical analysis using TradingView
- Market news analysis and impact assessment
- Support and resistance level calculations
- Telegram bot interface for easy access
- Automated signal generation
- Risk management with stop-loss and take-profit levels

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd trading-bot
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your API keys:
```bash
cp .env.example .env
```
Then edit the `.env` file and add your:
- Telegram Bot Token (Get from @BotFather)
- News API Key (Get from https://newsapi.org)

## Usage

1. Start the bot:
```bash
python trading_bot.py
```

2. Open Telegram and start a chat with your bot

3. Available commands:
- `/start` - Start the bot and get welcome message
- `/signals` - Get current trading signals

## Signal Generation

The bot generates signals based on:
1. Technical analysis from TradingView
2. Support and resistance levels
3. Market news sentiment
4. Risk management calculations

Each signal includes:
- Buy/Sell recommendation
- Entry price
- Stop loss level
- Take profit level
- Confidence percentage

## Supported Markets

- Forex pairs (EURUSD, GBPUSD, USDJPY, etc.)
- Commodities (Gold, Silver, Copper)
- Indices (US30, US500, USTEC)
- Cryptocurrencies (BTCUSD, ETHUSD)

## Risk Warning

Trading involves significant risk. This bot provides signals for informational purposes only. Always do your own research and never trade with money you cannot afford to lose.

## Contributing

Feel free to submit issues and enhancement requests! 