import os
os.environ['TZ'] = 'UTC'
from dotenv import load_dotenv
from typing import Dict, List
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/trading_bot.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class Config:
    def __init__(self):
        """Initialize configuration with default values."""
        # Load environment variables
        load_dotenv()
        
        # Telegram Bot Configuration
        self.TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
        if not self.TELEGRAM_BOT_TOKEN:
            logger.warning("TELEGRAM_BOT_TOKEN not found in environment variables")
        
        # Trading Configuration
        self.DEFAULT_TIMEFRAME = os.getenv('DEFAULT_TIMEFRAME', '1h')
        self.MIN_CONFIDENCE = float(os.getenv('MIN_CONFIDENCE', '0.85'))
        self.MAX_SPREAD = float(os.getenv('MAX_SPREAD', '0.0002'))
        self.MIN_VOLUME = int(os.getenv('MIN_VOLUME', '1500'))
        
        # News API Configuration
        self.NEWS_API_KEY = os.getenv('NEWS_API_KEY', '')
        if not self.NEWS_API_KEY:
            logger.warning("NEWS_API_KEY not found in environment variables")
        
        # Risk Management
        self.MAX_DAILY_LOSS = float(os.getenv('MAX_DAILY_LOSS', '2.0'))
        self.MAX_POSITION_SIZE = float(os.getenv('MAX_POSITION_SIZE', '1.0'))
        self.MAX_OPEN_TRADES = int(os.getenv('MAX_OPEN_TRADES', '3'))
        
        # Trading Hours (UTC)
        self.TRADING_START = os.getenv('TRADING_START', '08:00')
        self.TRADING_END = os.getenv('TRADING_END', '20:00')
        
        # Market Pairs
        self.FOREX_PAIRS = os.getenv('FOREX_PAIRS', 'EURUSD,GBPUSD,USDJPY,AUDUSD,USDCAD').split(',')
        self.COMMODITIES = os.getenv('COMMODITIES', 'XAUUSD,XAGUSD,USOIL,UKOIL').split(',')
        
        # Technical Analysis
        self.RSI_PERIOD = int(os.getenv('RSI_PERIOD', '14'))
        self.MACD_FAST = int(os.getenv('MACD_FAST', '12'))
        self.MACD_SLOW = int(os.getenv('MACD_SLOW', '26'))
        self.MACD_SIGNAL = int(os.getenv('MACD_SIGNAL', '9'))
        self.BOLLINGER_PERIOD = int(os.getenv('BOLLINGER_PERIOD', '20'))
        self.BOLLINGER_STD = float(os.getenv('BOLLINGER_STD', '2'))
        
        # Signal Filters
        self.MIN_INDICATORS = int(os.getenv('MIN_INDICATORS', '7'))
        self.MIN_PATTERNS = int(os.getenv('MIN_PATTERNS', '3'))
        self.MIN_RISK_REWARD = float(os.getenv('MIN_RISK_REWARD', '2.5'))
        
        # News Sources
        self.NEWS_SOURCES = {
            'fxstreet': 'https://www.fxstreet.com/news',
            'forexfactory': 'https://www.forexfactory.com/',
            'forex_com': 'https://www.forex.com/en/news-and-analysis/',
            'forexlive': 'https://www.forexlive.com/',
            'ig': 'https://www.ig.com/en/news-and-trade-ideas',
            'babypips': 'https://www.babypips.com/learn/forex/news-and-market-data'
        }
        
        # Validate configuration
        self.validate_config()
    
    def validate_config(self):
        """Validate the configuration values."""
        try:
            # Check required tokens
            if not self.TELEGRAM_BOT_TOKEN:
                raise ValueError("TELEGRAM_BOT_TOKEN is required")
            
            # Validate numeric values
            if not 0 < self.MIN_CONFIDENCE <= 1:
                raise ValueError("MIN_CONFIDENCE must be between 0 and 1")
            
            if self.MAX_SPREAD <= 0:
                raise ValueError("MAX_SPREAD must be positive")
            
            if self.MIN_VOLUME <= 0:
                raise ValueError("MIN_VOLUME must be positive")
            
            # Validate trading hours
            if not self.validate_time_format(self.TRADING_START):
                raise ValueError("Invalid TRADING_START time format")
            
            if not self.validate_time_format(self.TRADING_END):
                raise ValueError("Invalid TRADING_END time format")
            
            # Validate technical analysis parameters
            if self.RSI_PERIOD <= 0:
                raise ValueError("RSI_PERIOD must be positive")
            
            if self.MACD_FAST >= self.MACD_SLOW:
                raise ValueError("MACD_FAST must be less than MACD_SLOW")
            
            if self.BOLLINGER_PERIOD <= 0:
                raise ValueError("BOLLINGER_PERIOD must be positive")
            
            if self.BOLLINGER_STD <= 0:
                raise ValueError("BOLLINGER_STD must be positive")
            
            logger.info("Configuration validated successfully")
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            raise
    
    @staticmethod
    def validate_time_format(time_str: str) -> bool:
        """Validate time format (HH:MM)."""
        try:
            hours, minutes = map(int, time_str.split(':'))
            return 0 <= hours <= 23 and 0 <= minutes <= 59
        except:
            return False
    
    def get_trading_pairs(self) -> Dict[str, List[str]]:
        """Get all trading pairs."""
        return {
            'forex': self.FOREX_PAIRS,
            'commodities': self.COMMODITIES
        }
    
    def get_technical_params(self) -> Dict:
        """Get technical analysis parameters."""
        return {
            'rsi_period': self.RSI_PERIOD,
            'macd_fast': self.MACD_FAST,
            'macd_slow': self.MACD_SLOW,
            'macd_signal': self.MACD_SIGNAL,
            'bollinger_period': self.BOLLINGER_PERIOD,
            'bollinger_std': self.BOLLINGER_STD
        }
    
    def get_signal_filters(self) -> Dict:
        """Get signal filtering parameters."""
        return {
            'min_confidence': self.MIN_CONFIDENCE,
            'min_risk_reward': self.MIN_RISK_REWARD,
            'max_spread': self.MAX_SPREAD,
            'min_volume': self.MIN_VOLUME,
            'min_indicators': self.MIN_INDICATORS,
            'min_patterns': self.MIN_PATTERNS
        }

# Create a global config instance
config = Config() 