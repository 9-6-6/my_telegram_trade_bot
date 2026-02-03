import os
from config import config, logger

def test_configuration():
    """Test the configuration loading and validation."""
    try:
        # Test Telegram Bot Token
        if not config.TELEGRAM_BOT_TOKEN:
            logger.warning("Please set TELEGRAM_BOT_TOKEN in your environment variables")
        else:
            logger.info("Telegram Bot Token loaded successfully")
        
        # Test Trading Configuration
        logger.info(f"Default Timeframe: {config.DEFAULT_TIMEFRAME}")
        logger.info(f"Minimum Confidence: {config.MIN_CONFIDENCE}")
        logger.info(f"Maximum Spread: {config.MAX_SPREAD}")
        logger.info(f"Minimum Volume: {config.MIN_VOLUME}")
        
        # Test News API
        if not config.NEWS_API_KEY:
            logger.warning("Please set NEWS_API_KEY in your environment variables")
        else:
            logger.info("News API Key loaded successfully")
        
        # Test Risk Management
        logger.info(f"Maximum Daily Loss: {config.MAX_DAILY_LOSS}%")
        logger.info(f"Maximum Position Size: {config.MAX_POSITION_SIZE}%")
        logger.info(f"Maximum Open Trades: {config.MAX_OPEN_TRADES}")
        
        # Test Trading Hours
        logger.info(f"Trading Hours: {config.TRADING_START} - {config.TRADING_END} UTC")
        
        # Test Trading Pairs
        logger.info("Forex Pairs:")
        for pair in config.FOREX_PAIRS:
            logger.info(f"- {pair}")
        
        logger.info("Commodities:")
        for commodity in config.COMMODITIES:
            logger.info(f"- {commodity}")
        
        # Test Technical Analysis Parameters
        tech_params = config.get_technical_params()
        logger.info("Technical Analysis Parameters:")
        for param, value in tech_params.items():
            logger.info(f"- {param}: {value}")
        
        # Test Signal Filters
        signal_filters = config.get_signal_filters()
        logger.info("Signal Filters:")
        for filter_name, value in signal_filters.items():
            logger.info(f"- {filter_name}: {value}")
        
        logger.info("Configuration test completed successfully")
        
    except Exception as e:
        logger.error(f"Configuration test failed: {e}")
        raise

if __name__ == "__main__":
    test_configuration() 