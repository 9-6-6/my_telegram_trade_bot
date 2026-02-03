import os
import sys
from getpass import getpass

def setup_environment():
    """Set up environment variables for the trading bot."""
    print("Setting up environment variables for the trading bot...")
    
    # Get Telegram Bot Token
    telegram_token = getpass("Enter your Telegram Bot Token (or press Enter to skip): ")
    if telegram_token:
        os.environ['TELEGRAM_BOT_TOKEN'] = telegram_token
        print("Telegram Bot Token set successfully")
    
    # Get News API Key
    news_api_key = getpass("Enter your News API Key (or press Enter to skip): ")
    if news_api_key:
        os.environ['NEWS_API_KEY'] = news_api_key
        print("News API Key set successfully")
    
    # Set default trading configuration
    os.environ['DEFAULT_TIMEFRAME'] = '1h'
    os.environ['MIN_CONFIDENCE'] = '0.85'
    os.environ['MAX_SPREAD'] = '0.0002'
    os.environ['MIN_VOLUME'] = '1500'
    
    # Set risk management
    os.environ['MAX_DAILY_LOSS'] = '2.0'
    os.environ['MAX_POSITION_SIZE'] = '1.0'
    os.environ['MAX_OPEN_TRADES'] = '3'
    
    # Set trading hours
    os.environ['TRADING_START'] = '08:00'
    os.environ['TRADING_END'] = '20:00'
    
    # Set trading pairs
    os.environ['FOREX_PAIRS'] = 'EURUSD,GBPUSD,USDJPY,AUDUSD,USDCAD'
    os.environ['COMMODITIES'] = 'XAUUSD,XAGUSD,USOIL,UKOIL'
    
    # Set technical analysis parameters
    os.environ['RSI_PERIOD'] = '14'
    os.environ['MACD_FAST'] = '12'
    os.environ['MACD_SLOW'] = '26'
    os.environ['MACD_SIGNAL'] = '9'
    os.environ['BOLLINGER_PERIOD'] = '20'
    os.environ['BOLLINGER_STD'] = '2'
    
    # Set signal filters
    os.environ['MIN_INDICATORS'] = '7'
    os.environ['MIN_PATTERNS'] = '3'
    os.environ['MIN_RISK_REWARD'] = '2.5'
    
    print("\nEnvironment variables set successfully!")
    print("\nTo test the configuration, run: python test_config.py")

if __name__ == "__main__":
    setup_environment() 