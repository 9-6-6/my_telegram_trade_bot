import os
import sys
from pathlib import Path

def setup_bot():
    """Set up the trading bot with the provided token."""
    try:
        # Create .env file
        env_path = Path('.env')
        env_content = """# Telegram Bot Configuration
TELEGRAM_BOT_TOKEN=7849517577:AAGx8PhFyAf-cEFt06pfL_CPT8x9REVB1_U

# Trading Configuration
DEFAULT_TIMEFRAME=1h
MIN_CONFIDENCE=0.85
MAX_SPREAD=0.0002
MIN_VOLUME=1500

# Risk Management
MAX_DAILY_LOSS=2.0
MAX_POSITION_SIZE=1.0
MAX_OPEN_TRADES=3

# Trading Hours (UTC)
TRADING_START=08:00
TRADING_END=20:00

# Market Pairs
FOREX_PAIRS=EURUSD,GBPUSD,USDJPY,AUDUSD,USDCAD
COMMODITIES=XAUUSD,XAGUSD,USOIL,UKOIL

# Technical Analysis
RSI_PERIOD=14
MACD_FAST=12
MACD_SLOW=26
MACD_SIGNAL=9
BOLLINGER_PERIOD=20
BOLLINGER_STD=2

# Signal Filters
MIN_INDICATORS=7
MIN_PATTERNS=3
MIN_RISK_REWARD=2.5
"""
        
        # Write .env file
        with open(env_path, 'w') as f:
            f.write(env_content)
        
        print("\nBot configuration set successfully!")
        print("Created .env file with Telegram Bot Token and default settings")
        print("\nTo start the bot, run: python trading_bot.py")
        
    except Exception as e:
        print(f"Error setting up bot: {e}")
        sys.exit(1)

if __name__ == "__main__":
    setup_bot() 