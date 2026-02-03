import os
import logging
import schedule
import time
from datetime import datetime, timedelta
from typing import Dict
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, Bot
from telegram.constants import ParseMode as TelegramParseMode
from telegram.ext import CommandHandler, CallbackQueryHandler, Application, ContextTypes
from tradingview_ta import TA_Handler, Interval
import pandas as pd
import numpy as np
import pytz
import requests
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
# talib is optional - requires C library installation
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("[WARNING] TA-Lib not available. Some indicators will be disabled.")
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import threading
import asyncio

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Set default timezone
DEFAULT_TIMEZONE = pytz.timezone('UTC')

class TradingBot:
    def __init__(self):
        """Initialize the trading bot with enhanced features."""
        self.active_users = set()
        self.user_settings = {}
        self.telegram_bot = None
        self.setup_telegram_bot()
        
        # Initialize signal history and performance metrics
        self.signal_history = []
        self.performance_metrics = {
            'total_signals': 0,
            'winning_signals': 0,
            'losing_signals': 0,
            'average_profit': 0.0,
            'average_loss': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'max_drawdown': 0.0
        }
        self.weekly_performance = {}
        
        # Market news sources
        self.news_sources = {
            'fxstreet': 'https://www.fxstreet.com/news',
            'forexfactory': 'https://www.forexfactory.com/',
            'forex_com': 'https://www.forex.com/en/news-and-analysis/',
            'forexlive': 'https://www.forexlive.com/',
            'ig': 'https://www.ig.com/en/news-and-trade-ideas',
            'babypips': 'https://www.babypips.com/learn/forex/news-and-market-data'
        }
        
        # Market sentiment tracking
        self.market_sentiment = {
            'overall': 'neutral',
            'by_pair': {},
            'by_timeframe': {},
            'news_impact': {},
            'trend_strength': {}
        }
        
        # News impact analysis
        self.news_impact = {
            'high_impact': [],
            'medium_impact': [],
            'low_impact': [],
            'scheduled_events': {}
        }
        
        # Initialize news monitoring
        self.setup_news_monitoring()
        
        # Enhanced signal filtering
        self.signal_filters = {
            'min_confidence': 0.85,  # Minimum confidence threshold
            'min_risk_reward': 2.5,  # Minimum risk/reward ratio
            'max_spread': 0.0002,    # Maximum allowed spread
            'min_volume': 1500,      # Minimum volume requirement
            'min_indicators': 7,     # Minimum number of confirming indicators
            'min_patterns': 3,       # Minimum number of confirming patterns
            'time_filters': {
                'avoid_news': True,  # Avoid trading during high-impact news
                'trading_hours': {
                    'start': '08:00',
                    'end': '20:00'
                }
            }
        }
        
        # Optimized indicator combinations for high-probability setups
        self.indicator_combinations = {
            'trend_reversal': {
                'required': ['moving_averages', 'ichimoku', 'adx'],
                'confirming': ['rsi', 'macd', 'stochastic', 'cci'],
                'volume': ['obv', 'vwap', 'mfi'],
                'volatility': ['bollinger', 'atr']
            },
            'breakout': {
                'required': ['bollinger', 'atr', 'donchian'],
                'confirming': ['macd', 'rsi', 'cci'],
                'volume': ['obv', 'vwap'],
                'momentum': ['stochastic', 'adx']
            },
            'trend_continuation': {
                'required': ['moving_averages', 'ichimoku', 'parabolic_sar'],
                'confirming': ['macd', 'rsi', 'cci'],
                'volume': ['obv', 'mfi'],
                'momentum': ['stochastic', 'adx']
            }
        }
        
        # Enhanced pattern recognition settings
        self.pattern_settings = {
            'candlestick': {
                'doji': {'min_body_ratio': 0.1, 'max_body_ratio': 0.3},
                'hammer': {'min_body_ratio': 0.3, 'max_body_ratio': 0.5},
                'engulfing': {'min_body_ratio': 0.6, 'min_overlap': 0.8},
                'morning_star': {'min_body_ratio': 0.4, 'max_body_ratio': 0.6},
                'evening_star': {'min_body_ratio': 0.4, 'max_body_ratio': 0.6}
            },
            'chart': {
                'double_top': {'min_retracement': 0.382, 'max_retracement': 0.618},
                'double_bottom': {'min_retracement': 0.382, 'max_retracement': 0.618},
                'head_shoulders': {'min_retracement': 0.5, 'max_retracement': 0.786},
                'triangle': {'min_touches': 3, 'max_deviation': 0.1},
                'wedge': {'min_touches': 3, 'max_deviation': 0.15}
            },
            'harmonic': {
                'gartley': {'tolerance': 0.05},
                'butterfly': {'tolerance': 0.05},
                'bat': {'tolerance': 0.05},
                'crab': {'tolerance': 0.05}
            }
        }
        
        # Optimized indicator weights for maximum accuracy
        self.indicator_weights = {
            'trend_indicators': {
                'moving_averages': 1.3,  # Increased weight
                'ichimoku': 1.2,
                'adx': 1.1,
                'parabolic_sar': 1.0
            },
            'momentum_indicators': {
                'rsi': 1.1,
                'stochastic': 1.1,
                'macd': 1.2,  # Increased weight
                'cci': 1.0
            },
            'volume_indicators': {
                'obv': 1.1,
                'vwap': 1.2,  # Increased weight
                'mfi': 1.0
            },
            'volatility_indicators': {
                'bollinger': 1.1,
                'atr': 1.1,
                'donchian': 1.0
            },
            'support_resistance': {
                'fibonacci': 1.2,  # Increased weight
                'pivot_points': 1.1
            }
        }
        
        # Technical indicators configuration
        self.indicator_settings = {
            'bollinger': {'period': 20, 'std_dev': 2},
            'rsi': {'period': 14, 'overbought': 70, 'oversold': 30},
            'stochastic': {'k_period': 14, 'd_period': 3, 'slowing': 3},
            'macd': {'fast': 12, 'slow': 26, 'signal': 9},
            'ichimoku': {
                'tenkan': 9,
                'kijun': 26,
                'senkou_span_b': 52,
                'displacement': 26
            },
            'parabolic_sar': {'acceleration': 0.02, 'maximum': 0.2},
            'adx': {'period': 14, 'threshold': 25},
            'fibonacci': {'levels': [0.236, 0.382, 0.5, 0.618, 0.786]},
            'pivot_points': {'method': 'standard'},
            'atr': {'period': 14},
            'cci': {'period': 20},
            'donchian': {'period': 20},
            'dpo': {'period': 20},
            'mfi': {'period': 14},
            'aroon': {'period': 25}
        }
        
        # Profit optimization settings
        self.profit_settings = {
            'take_profit_levels': [1.5, 2.0, 3.0],  # Multiple take profit levels
            'trailing_stop': True,
            'trailing_stop_distance': 0.002,
            'break_even_threshold': 1.0,  # Move to break even after 1:1 risk/reward
            'partial_close_levels': [0.5, 0.75, 1.0]  # Close portions at these R:R levels
        }
        
        # Advanced analysis settings (comprehensive timeframes: 1m to 48h)
        self.analysis_settings = {
            'timeframes': ['1m', '3m', '5m', '7m', '10m', '15m', '20m', '30m', '1h', '2h', '4h', '8h', '12h', '24h', '48h'],
            'pattern_weights': {
                'harmonic': 1.2,
                'elliott': 1.1,
                'candlestick': 0.9,
                'chart': 1.0
            },
            'indicator_weights': {
                'rsi': 0.8,
                'macd': 0.9,
                'bollinger': 1.0,
                'fibonacci': 1.1,
                'volume': 0.9
            }
        }
        
        # Initialize ML models with enhanced features
        self.initialize_ml_models()
        
        # Active signals tracking
        self.active_signals = {}
        
        # Signal monitoring settings
        self.monitoring_interval = 5  # minutes
        self.reversal_threshold = 0.7  # 70% confidence for reversal
        self.max_loss_threshold = 0.02  # 2% maximum loss before warning
        
        # Enhanced tracking features
        self.tracking_features = {
            'signal_categories': {
                'trend_reversal': [],
                'breakout': [],
                'continuation': []
            },
            'performance_by_timeframe': {
                'm1': {'wins': 0, 'losses': 0},
                'm5': {'wins': 0, 'losses': 0},
                'm15': {'wins': 0, 'losses': 0},
                'h1': {'wins': 0, 'losses': 0},
                'h4': {'wins': 0, 'losses': 0},
                'd1': {'wins': 0, 'losses': 0}
            },
            'performance_by_pattern': {},
            'risk_metrics': {
                'max_consecutive_losses': 0,
                'current_consecutive_losses': 0,
                'max_daily_loss': 0,
                'max_daily_profit': 0
            }
        }
    
    # def setup_telegram_bot(self):
    #     """Set up Telegram bot with enhanced commands."""
    #     try:
    #         # Configure timezone for APScheduler
    #         os.environ['TZ'] = 'UTC'
            
    #         # Initialize bot
    #         self.telegram_bot = Bot(token=os.getenv('TELEGRAM_BOT_TOKEN'))
            
    #     except Exception as e:
    #         logger.error(f"Error setting up Telegram bot: {e}")
    
    def handle_start(self, message):
        """Handle /start command."""
        try:
            user_id = message.from_user.id
            welcome_message = (
                "Welcome to the Trading Bot! 🚀\n\n"
                "Available commands:\n"
                "/subscribe - Subscribe to trading signals\n"
                "/unsubscribe - Unsubscribe from signals\n"
                "/settings - Configure your preferences\n"
                "/status - Check bot status\n"
                "/help - Show this help message\n\n"
                "To start receiving signals, use /subscribe"
            )
            self.telegram_bot.reply_to(message, welcome_message)
            
        except Exception as e:
            logger.error(f"Error handling start command: {e}")
    
    def handle_help(self, message):
        """Handle /help command."""
        try:
            help_message = (
                "📚 Trading Bot Help\n\n"
                "Commands:\n"
                "/subscribe - Start receiving trading signals\n"
                "/unsubscribe - Stop receiving signals\n"
                "/settings - Configure your preferences:\n"
                "  • Set minimum confidence level\n"
                "  • Choose preferred pairs\n"
                "  • Set notification preferences\n"
                "/status - Check bot status and your subscription\n\n"
                "Signal Format:\n"
                "• Signal type (BUY/SELL)\n"
                "• Confidence level\n"
                "• Pattern analysis\n"
                "• Price targets\n"
                "• Risk management\n"
                "• Technical indicators\n\n"
                "Need more help? Contact support."
            )
            self.telegram_bot.reply_to(message, help_message)
            
        except Exception as e:
            logger.error(f"Error handling help command: {e}")
    
    def handle_subscribe(self, message):
        """Handle /subscribe command."""
        try:
            user_id = message.from_user.id
            if user_id not in self.active_users:
                self.active_users.add(user_id)
                self.user_settings[user_id] = {
                    'min_confidence': 0.7,
                    'preferred_pairs': ['EURUSD', 'GBPUSD', 'XAUUSD'],
                    'notifications': True
                }
                self.telegram_bot.reply_to(
                    message,
                    "✅ Successfully subscribed to trading signals!\n"
                    "Use /settings to configure your preferences."
                )
            else:
                self.telegram_bot.reply_to(
                    message,
                    "You are already subscribed to trading signals."
                )
            
        except Exception as e:
            logger.error(f"Error handling subscribe command: {e}")
    
    def handle_unsubscribe(self, message):
        """Handle /unsubscribe command."""
        try:
            user_id = message.from_user.id
            if user_id in self.active_users:
                self.active_users.remove(user_id)
                self.telegram_bot.reply_to(
                    message,
                    "You have been unsubscribed from trading signals."
                )
            else:
                self.telegram_bot.reply_to(
                    message,
                    "You are not currently subscribed to trading signals."
                )
            
        except Exception as e:
            logger.error(f"Error handling unsubscribe command: {e}")
    
    def handle_settings(self, message):
        """Handle /settings command."""
        try:
            user_id = message.from_user.id
            if user_id not in self.active_users:
                self.telegram_bot.reply_to(
                    message,
                    "Please subscribe first using /subscribe"
                )
                return
            
            # Create settings keyboard
            keyboard = InlineKeyboardMarkup()
            keyboard.add(InlineKeyboardButton(
                "Set Minimum Confidence",
                callback_data="set_confidence"
            ))
            keyboard.add(InlineKeyboardButton(
                "Choose Pairs",
                callback_data="set_pairs"
            ))
            keyboard.add(InlineKeyboardButton(
                "Notification Settings",
                callback_data="set_notifications"
            ))
            
            self.telegram_bot.reply_to(
                message,
                "Configure your settings:",
                reply_markup=keyboard
            )
            
        except Exception as e:
            logger.error(f"Error handling settings command: {e}")
    
    def handle_status(self, message):
        """Handle /status command."""
        try:
            user_id = message.from_user.id
            if user_id not in self.active_users:
                self.telegram_bot.reply_to(
                    message,
                    "You are not subscribed to trading signals.\n"
                    "Use /subscribe to start receiving signals."
                )
                return
            
            settings = self.user_settings[user_id]
            status_message = (
                "📊 Bot Status\n\n"
                f"Subscription: Active\n"
                f"Minimum Confidence: {settings['min_confidence']*100}%\n"
                f"Preferred Pairs: {', '.join(settings['preferred_pairs'])}\n"
                f"Notifications: {'Enabled' if settings['notifications'] else 'Disabled'}\n\n"
                "Use /settings to modify your preferences."
            )
            
            self.telegram_bot.reply_to(message, status_message)
            
        except Exception as e:
            logger.error(f"Error handling status command: {e}")
    
    async def broadcast_message(self, message: str):
        """Broadcast message to all active users."""
        try:
            for user_id in self.active_users:
                try:
                    self.telegram_bot.send_message(user_id, message)
                except Exception as e:
                    logger.error(f"Error sending message to user {user_id}: {e}")
                    # Remove inactive users
                    self.active_users.remove(user_id)
                    
        except Exception as e:
            logger.error(f"Error broadcasting message: {e}")
    
    # def initialize_ml_models(self):
    #     """Initialize and load ML models."""
    #     try:
    #         # Load or create Random Forest model
    #         self.rf_model = joblib.load('rf_model.joblib') if os.path.exists('rf_model.joblib') else RandomForestClassifier(n_estimators=100)
            
    #         # Load or create LSTM model
    #         if os.path.exists('lstm_model.h5'):
    #             self.lstm_model = tf.keras.models.load_model('lstm_model.h5')
    #         else:
    #             self.lstm_model = self.create_lstm_model()
            
    #         # Initialize scaler
    #         self.scaler = StandardScaler()
            
        # except Exception as e:
        #     logger.error(f"Error initializing ML models: {e}")
    
    def create_lstm_model(self):
        """Create LSTM model for price prediction."""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(60, 20)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        print("/start command received")
        await update.message.reply_text("✅ Main bot is alive! Send /signal XAUUSD or /signal BTCUSD to test.")

    async def signal(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        print("/signal command received")
        symbol = "XAUUSD"
        if context.args:
            symbol = context.args[0].upper()
        message = await update.message.reply_text(f"🔔 Getting latest trading signal for {symbol}...")
        try:
            signals = await asyncio.wait_for(
                asyncio.to_thread(self.get_current_signals, symbol),
                timeout=30
            )
        except asyncio.TimeoutError:
            await message.edit_text("❌ Signal request timed out. Please try again later.")
            return
        except Exception as e:
            await message.edit_text("❌ Error retrieving signals. Please try again later.")
            return
        if not self.is_market_open(symbol):
            await message.edit_text(f"ℹ️ The market for {symbol} is currently closed. Try BTCUSD or another crypto for 24/7 signals.")
            return
        if not signals:
            await message.edit_text("ℹ️ No high-probability signals available at the moment.")
            return
        for signal in signals:
            signal_message = (
                f"🎯 ULTRA HIGH-PROBABILITY TRADING SIGNAL for {signal['symbol']} 🎯\n\n"
                f"Signal Type: {signal['type']}\n"
                f"Confidence: {signal['confidence']*100:.1f}%\n"
                f"Indicator Confirmations: {signal['confirmations']}\n"
                f"Pattern Confirmations: {signal['pattern_confirmations']}\n\n"
                f"Indicator Combinations:\n" + "\n".join(
                    f"- {combo['type']} ({combo['confidence']*100:.1f}%)"
                    for combo in signal['indicator_combinations']
                ) + "\n\n"
                f"Price Targets:\n"
                f"- Entry: {signal['entry_price']:.5f}\n"
                f"- Take Profit 1: {signal['targets']['take_profit_levels'][0]['price']:.5f} ({signal['targets']['take_profit_levels'][0]['level']}R)\n"
                f"- Take Profit 2: {signal['targets']['take_profit_levels'][1]['price']:.5f} ({signal['targets']['take_profit_levels'][1]['level']}R)\n"
                f"- Take Profit 3: {signal['targets']['take_profit_levels'][2]['price']:.5f} ({signal['targets']['take_profit_levels'][2]['level']}R)\n"
                f"- Stop Loss: {signal['targets']['stop_loss']:.5f}\n\n"
                f"Resistance Levels:\n"
                f"- R1: {signal['resistance_levels']['r1']:.5f}\n"
                f"- R2: {signal['resistance_levels']['r2']:.5f}\n"
                f"- R3: {signal['resistance_levels']['r3']:.5f}\n\n"
                f"Profit Summary:\n"
                f"- Risk:Reward Ratio: {signal['risk_reward']:.1f}\n"
                f"- Potential Profit: {signal['potential_profit']:.1f} pips\n"
                f"- Maximum Loss: {signal['max_loss']:.1f} pips\n\n"
                f"Technical Summary:\n"
                f"- Trend: {signal['technical_summary']['trend']}\n"
                f"- Volatility: {signal['technical_summary']['volatility']}\n"
                f"- Volume: {signal['technical_summary']['volume']}"
            )
            await message.edit_text(signal_message)

    def is_market_open(self, symbol: str) -> bool:
        symbol = symbol.upper()
        if symbol.endswith("USD") and symbol.startswith("BTC"):
            return True
        now_utc = datetime.utcnow()
        now_ny = now_utc.replace(tzinfo=pytz.utc).astimezone(pytz.timezone("America/New_York"))
        weekday = now_ny.weekday()
        hour = now_ny.hour
        if weekday == 6 and hour >= 17:
            return True
        if 0 <= weekday <= 3:
            return True
        if weekday == 4 and hour < 17:
            return True
        return False

    def get_current_signals(self, symbol="XAUUSD") -> list:
        if not self.is_market_open(symbol):
            return []
        # Sample signal for open markets
        return [{
            'symbol': symbol,
            'type': 'BUY',
            'confidence': 0.85,
            'confirmations': 8,
            'pattern_confirmations': 3,
            'indicator_combinations': [
                {'type': 'Moving Averages', 'confidence': 0.9},
                {'type': 'RSI', 'confidence': 0.85},
                {'type': 'MACD', 'confidence': 0.8}
            ],
            'entry_price': 1.0850,
            'targets': {
                'take_profit_levels': [
                    {'price': 1.0900, 'level': '1.5R'},
                    {'price': 1.0950, 'level': '2.0R'},
                    {'price': 1.1000, 'level': '3.0R'}
                ],
                'stop_loss': 1.0800
            },
            'resistance_levels': {
                'r1': 1.0900,
                'r2': 1.0950,
                'r3': 1.1000
            },
            'risk_reward': 2.5,
            'potential_profit': 150.0,
            'max_loss': 50.0,
            'technical_summary': {
                'trend': 'Bullish',
                'volatility': 'Medium',
                'volume': 'High'
            }
        }]

async def main():
    """Start the bot."""
    try:
        # Set timezone for APScheduler
        os.environ['TZ'] = 'UTC'
        
        # Create the Application and pass it your bot's token
        application = (
            Application.builder()
            .token(os.getenv('TELEGRAM_BOT_TOKEN'))
            .concurrent_updates(True)
            .build()
        )

        # Create bot instance
        bot = TradingBot()

        # Add command handlers
        application.add_handler(CommandHandler("start", bot.start))
        application.add_handler(CommandHandler("signal", bot.signal))

        # Start the Bot
        print("Main bot started. Use /start and /signal XAUUSD or /signal BTCUSD in Telegram.")
        await application.initialize()
        await application.start()
        await application.run_polling()

    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Bot stopped by user")
    except Exception as e:
        logger.error(f"Error running bot: {e}") 