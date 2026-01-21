"""
XM360 Auto Trader Configuration

IMPORTANT: Update these settings with your XM360 account details.
Start with DEMO account to test the system before going live.
"""

# =============================================================================
# MT5 ACCOUNT SETTINGS
# =============================================================================

# Your XM360 account number (e.g., 12345678)
ACCOUNT = 315982803  # XM360 Demo Account - Standard

# Your MT5 password (NOT your XM portal password)
PASSWORD = "Gadhiya@098"  # MT5 Trading Password

# MT5 Server name
# Demo: "XMGlobal-MT5" or "XMGlobal-MT5 2" 
# Live: Check your XM account for the correct server
SERVER = "XMGlobal-MT5 7"

# Path to MetaTrader 5 terminal (usually auto-detected)
# Example: "C:/Program Files/XM Global MT5/terminal64.exe"
MT5_PATH = None  # Leave None for auto-detection

# =============================================================================
# TRADING SETTINGS
# =============================================================================

# Use DEMO account (set to False for LIVE trading - USE WITH CAUTION!)
USE_DEMO = True

# Default lot size for trades
DEFAULT_LOT_SIZE = 0.01

# Maximum lot size allowed
MAX_LOT_SIZE = 0.1

# Minimum lot size
MIN_LOT_SIZE = 0.01

# =============================================================================
# RISK MANAGEMENT
# =============================================================================

# Maximum risk per trade as percentage of account balance
MAX_RISK_PERCENT = 1.0  # 1% max risk per trade

# Maximum percentage of balance to use for trading (margin usage)
MAX_BALANCE_USAGE_PERCENT = 10.0  # Only use 10% of total balance for trading

# Minimum account balance required to trade (stop trading if below this)
MIN_BALANCE_TO_TRADE = 100.0  # Stop trading if balance falls below $100

# Minimum free margin required to open new positions
MIN_FREE_MARGIN = 50.0  # Need at least $50 free margin to trade

# Maximum number of open positions
MAX_OPEN_POSITIONS = 5

# Maximum daily loss as percentage of account balance
MAX_DAILY_LOSS_PERCENT = 5.0  # Stop trading if daily loss exceeds 5%

# Maximum daily trades
MAX_DAILY_TRADES = 20

# =============================================================================
# SIGNAL SETTINGS
# =============================================================================

# Slippage tolerance in points
MAX_SLIPPAGE = 20

# Signal expiry time in seconds (ignore signals older than this)
SIGNAL_EXPIRY_SECONDS = 60

# Minimum time between signals for same symbol (seconds)
MIN_SIGNAL_INTERVAL = 300  # 5 minutes

# =============================================================================
# PRICE VALIDATION SETTINGS
# =============================================================================

# Maximum allowed price deviation from signal entry price (as percentage)
# If current price differs more than this %, trade will be rejected
MAX_PRICE_DEVIATION_PERCENT = 0.5  # 0.5% maximum deviation

# Maximum price deviation in absolute points for different asset types
# These override percentage if specified (set to 0 to use percentage only)
MAX_PRICE_DEVIATION_FOREX = 0.0  # Use percentage for forex
MAX_PRICE_DEVIATION_GOLD = 5.0   # $5 max deviation for gold (XAUUSD)
MAX_PRICE_DEVIATION_SILVER = 0.5  # $0.50 max deviation for silver
MAX_PRICE_DEVIATION_CRYPTO = 500.0  # $500 max deviation for BTC
MAX_PRICE_DEVIATION_INDICES = 50.0  # 50 points max for indices

# Enable/disable price validation
ENABLE_PRICE_VALIDATION = True

# Log price comparison details
LOG_PRICE_VALIDATION = True

# =============================================================================
# SYMBOL MAPPING (Telegram Symbol -> MT5 Symbol)
# =============================================================================
# XM360 may use different symbol names than what we use in signals
SYMBOL_MAPPING = {
    'XAUUSD': 'GOLD',      # Gold
    'XAGUSD': 'SILVER',    # Silver
    'BTCUSD': 'BTCUSD',    # Bitcoin (if available)
    'EURUSD': 'EURUSD',
    'GBPUSD': 'GBPUSD',
    'USDJPY': 'USDJPY',
    'AUDUSD': 'AUDUSD',
    'USDCAD': 'USDCAD',
    'USDCHF': 'USDCHF',
    'NZDUSD': 'NZDUSD',
    'EURJPY': 'EURJPY',
    'GBPJPY': 'GBPJPY',
    'EURGBP': 'EURGBP',
    'EURAUD': 'EURAUD',
    'USOIL': 'OIL',        # US Oil / WTI Crude
    'US30': 'US30',        # Dow Jones
    'US500': 'US500',      # S&P 500
    'US100': 'US100',      # NASDAQ 100
}

# Symbols to auto-trade (leave empty to trade all mapped symbols)
ALLOWED_SYMBOLS = []  # e.g., ['XAUUSD', 'EURUSD', 'GBPUSD']

# Symbols to NEVER auto-trade
BLOCKED_SYMBOLS = ['BTCUSD']  # Crypto may not be available on XM

# =============================================================================
# LOGGING
# =============================================================================

# Enable detailed logging
VERBOSE_LOGGING = True

# Log file path
LOG_FILE = "xm360_auto_trader/logs/auto_trader.log"

# Trade history file
TRADE_HISTORY_FILE = "xm360_auto_trader/logs/trade_history.json"

# =============================================================================
# SIGNAL QUEUE (Communication with Trading Bot)
# =============================================================================

# File-based signal queue path
SIGNAL_QUEUE_FILE = "xm360_auto_trader/signal_queue.json"

# Check for new signals every N seconds
SIGNAL_CHECK_INTERVAL = 1

# =============================================================================
# NOTIFICATIONS
# =============================================================================

# Send trade notifications back to Telegram
SEND_TELEGRAM_NOTIFICATIONS = True

# Telegram bot token (same as main bot)
TELEGRAM_BOT_TOKEN = "7849517577:AAGx8PhFyAf-cEFt06pfL_CPT8x9REVB1_U"

# Chat ID to send trade notifications (your user ID)
NOTIFICATION_CHAT_ID = None  # TODO: Enter your Telegram chat ID
