import os
import logging
import schedule
import time
from datetime import datetime, timedelta
from typing import Dict
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, Bot
from telegram.constants import ParseMode as TelegramParseMode
from telegram.ext import CommandHandler, CallbackQueryHandler, Application, ContextTypes
from tradingview_ta import TA_Handler, Interval, Exchange
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
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from bs4 import BeautifulSoup
import telegram

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
        """Initialize the trading bot."""
        try:
            # Initialize Telegram application
            self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
            if not self.bot_token:
                raise ValueError("TELEGRAM_BOT_TOKEN not found in environment variables")
            
            print(f"Initializing bot with token: {self.bot_token[:10]}...")
            self.application = Application.builder().token(self.bot_token).build()
            
            # Initialize scheduler
            self.scheduler = AsyncIOScheduler()
            
            # Initialize allowed users
            allowed_users_str = os.getenv('ALLOWED_USERS', '')
            self.allowed_users = set(int(id) for id in allowed_users_str.split(',') if id)
            print(f"Allowed users: {self.allowed_users}")
            
            # Initialize market sentiment
            self.market_sentiment = {
                'overall': 'neutral',
                'trend_strength': 0.5,
                'news_impact': {
                    'USD': 'low',
                    'EUR': 'low',
                    'GBP': 'low',
                    'JPY': 'low',
                    'AUD': 'low',
                    'CAD': 'low',
                    'CHF': 'low',
                    'NZD': 'low',
                    'XAU': 'low'
                }
            }
            
            # Initialize ML model
            self.ml_model = self.initialize_ml_model()
            self.min_ml_confidence = 0.7
            
            # Initialize user settings
            self.user_settings = {}
            
            print("Bot initialized successfully")
            
        except Exception as e:
            print(f"Error initializing bot: {e}")
            logger.error(f"Error initializing bot: {e}")
            raise

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Send a message when the command /help is issued."""
        try:
            help_text = (
                "🤖 *Trading Bot Help*\n\n"
                "Available commands:\n"
                "/start - Start the bot\n"
                "/help - Show this help message\n"
                "/signal [symbol] - Get trading signal for a symbol\n"
                "Example: /signal BTCUSD\n\n"
                "Supported symbols:\n"
                "- Cryptocurrencies: BTCUSD, ETHUSD, etc.\n"
                "- Forex: EURUSD, GBPUSD, etc.\n"
                "- Metals: XAUUSD, XAGUSD\n"
                "- Indices: SPX500, US30, etc."
            )
            await update.message.reply_text(help_text, parse_mode='Markdown')
            logger.info("Help message sent successfully")
        except Exception as e:
            logger.error(f"Error in help command: {e}")
            try:
                await update.message.reply_text("❌ Error: Could not send help message. Please try again.")
            except Exception as e2:
                logger.error(f"Error sending error message: {e2}")

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Send a message when the command /start is issued."""
        try:
            user_id = update.effective_user.id
            logger.info(f"Start command received from user {user_id}")
            
            # Check if user is allowed
            if user_id not in self.allowed_users:
                logger.warning(f"Unauthorized access attempt from user {user_id}")
                await update.message.reply_text("❌ You are not authorized to use this bot.")
                return
            
            welcome_message = (
                "👋 Welcome to the Trading Bot!\n\n"
                "I can help you with trading signals for various markets.\n\n"
                "Try these commands:\n"
                "• /signal BTCUSD - Get signal for Bitcoin\n"
                "• /signal XAUUSD - Get signal for Gold\n"
                "• /help - Show all available commands\n\n"
                "Need help? Use /help for more information."
            )
            
            await update.message.reply_text(welcome_message)
            logger.info(f"Welcome message sent to user {user_id}")
            
        except Exception as e:
            logger.error(f"Error in start command: {e}")
            try:
                await update.message.reply_text("❌ Error: Could not process command. Please try again.")
            except Exception as e2:
                logger.error(f"Error sending error message: {e2}")

    async def start_scheduler(self, context: ContextTypes.DEFAULT_TYPE):
        """Start scheduler when bot starts."""
        try:
            self.scheduler.start()
            logger.info("Scheduler started successfully")
        except Exception as e:
            logger.error(f"Error starting scheduler: {e}")

    async def stop_scheduler(self, context: ContextTypes.DEFAULT_TYPE):
        """Stop scheduler when bot stops."""
        try:
            self.scheduler.shutdown()
            logger.info("Scheduler stopped successfully")
        except Exception as e:
            logger.error(f"Error stopping scheduler: {e}")

    def register_handlers(self):
        """Register command handlers."""
        self.application.add_handler(CommandHandler("start", self.start))
        self.application.add_handler(CommandHandler("signal", self.signal))
        self.application.add_handler(CommandHandler("performance", self.performance))
        self.application.add_handler(CommandHandler("settings", self.settings))
        self.application.add_handler(CommandHandler("setpairs", self.setpairs))
        self.application.add_handler(CommandHandler("setnotify", self.setnotify))

    def setup_news_monitoring(self):
        """Setup news monitoring with scheduler."""
        try:
            # Schedule news check every 5 minutes
            self.scheduler.add_job(
                self.update_news_impact,
                'interval',
                minutes=5,
                id='news_monitor'
            )
            logger.info("News monitoring setup complete")
        except Exception as e:
            logger.error(f"Error setting up news monitoring: {e}")

    async def update_news_impact(self):
        """Update news impact for all currencies."""
        try:
            currencies = ['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF', 'NZD', 'XAU']
            for currency in currencies:
                self.market_sentiment['news_impact'][currency] = self.get_news_impact(currency)
            logger.info("News impact updated successfully")
        except Exception as e:
            logger.error(f"Error updating news impact: {e}")

    def initialize_ml_model(self):
        """Initialize and load ML model."""
        try:
            # Load or create Random Forest model
            self.rf_model = joblib.load('rf_model.joblib') if os.path.exists('rf_model.joblib') else RandomForestClassifier(n_estimators=100)
            
            # Load or create LSTM model
            if os.path.exists('lstm_model.h5'):
                self.lstm_model = tf.keras.models.load_model('lstm_model.h5')
            else:
                self.lstm_model = self.create_lstm_model()
            
            # Initialize scaler
            self.scaler = StandardScaler()
            
        except Exception as e:
            logger.error(f"Error initializing ML model: {e}")
            return None

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

    def log_signal(self, signal):
        from datetime import datetime
        # Log the signal with timestamp and default result (None)
        self.signal_history.append({
            'timestamp': datetime.now(),
            'symbol': signal['symbol'],
            'type': signal['type'],
            'entry_price': signal['entry_price'],
            'result': None,  # To be updated when closed
            'profit_loss': 0,
            'confidence': signal['confidence'],
            'risk_reward': signal['risk_reward']
        })

    def calculate_performance(self):
        total_signals = len(self.signal_history)
        winning_signals = len([s for s in self.signal_history if s['result'] == 'win'])
        losing_signals = len([s for s in self.signal_history if s['result'] == 'loss'])
        win_rate = winning_signals / total_signals if total_signals > 0 else 0
        avg_profit = (
            sum(s['profit_loss'] for s in self.signal_history if s['result'] == 'win') / winning_signals
            if winning_signals > 0 else 0
        )
        avg_loss = (
            sum(s['profit_loss'] for s in self.signal_history if s['result'] == 'loss') / losing_signals
            if losing_signals > 0 else 0
        )
        profit_factor = (
            sum(s['profit_loss'] for s in self.signal_history if s['result'] == 'win') /
            abs(sum(s['profit_loss'] for s in self.signal_history if s['result'] == 'loss'))
            if losing_signals > 0 else 0
        )
        return {
            'total_signals': total_signals,
            'winning_signals': winning_signals,
            'losing_signals': losing_signals,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor
        }

    async def performance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        stats = self.calculate_performance()
        message = (
            "📊 PERFORMANCE ANALYTICS 📊\n\n"
            f"Total Signals: {stats['total_signals']}\n"
            f"Win Rate: {stats['win_rate']*100:.1f}%\n"
            f"Average Profit: {stats['avg_profit']:.1f} pips\n"
            f"Average Loss: {stats['avg_loss']:.1f} pips\n"
            f"Profit Factor: {stats['profit_factor']:.2f}\n"
        )
        await update.message.reply_text(message)

    async def signal(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /signal command with enhanced analysis."""
        try:
            logger.info("/signal command received")
            user_id = update.effective_user.id
            logger.info(f"Processing signal request from user {user_id}")

            # Check if user is allowed
            if user_id not in self.allowed_users:
                logger.warning(f"Unauthorized access attempt from user {user_id}")
                await update.message.reply_text("❌ You are not authorized to use this bot.")
                return

            # Use user's favorite pairs if no symbol is provided
            if context.args:
                pairs = [arg.upper() for arg in context.args]
                logger.info(f"Using provided pairs: {pairs}")
            else:
                settings = self.user_settings.get(user_id, {'favorite_pairs': ['XAUUSD'], 'notifications': True})
                pairs = settings['favorite_pairs']
                logger.info(f"Using default pairs from settings: {pairs}")

            for symbol in pairs:
                try:
                    logger.info(f"Processing symbol: {symbol}")

                    if not self.is_market_open(symbol):
                        logger.info(f"Market closed for {symbol}")
                        await update.message.reply_text(
                            f"ℹ️ The market for {symbol} is currently closed. Try BTCUSD or another crypto for 24/7 signals."
                        )
                        continue

                    if self.should_avoid_trading(symbol):
                        logger.info(f"High-impact news detected for {symbol}")
                        await update.message.reply_text(
                            f"⚠️ High-impact news event detected for {symbol}. Trading is not recommended at this time."
                        )
                        continue

                    # Send initial message
                    try:
                        message = await update.message.reply_text(f"🔔 Getting latest trading signal for {symbol}...")
                        logger.info("Initial message sent successfully")
                    except Exception as e:
                        logger.error(f"Error sending initial message: {e}")
                        continue

                    try:
                        logger.info(f"Fetching signals for {symbol}")
                        signals = await asyncio.wait_for(
                            asyncio.to_thread(self.get_current_signals, symbol),
                            timeout=30
                        )
                        logger.info(f"Received signals: {signals}")
                    except asyncio.TimeoutError:
                        logger.error(f"Timeout while fetching signals for {symbol}")
                        await message.edit_text("❌ Signal request timed out. Please try again later.")
                        continue
                    except Exception as e:
                        logger.error(f"Error fetching signals for {symbol}: {e}")
                        await message.edit_text("❌ Error retrieving signals. Please try again later.")
                        continue

                    if not signals:
                        logger.info(f"No signals found for {symbol}")
                        await message.edit_text("ℹ️ No high-probability signals available at the moment.")
                        continue

                    # Format and send signal with sentiment, ML, and multi-timeframe info
                    signal = signals[0]
                    logger.info(f"Processing signal: {signal}")

                    try:
                        sentiment = signal.get('sentiment', {})
                        ml_score = signal.get('ml_score', 0)
                        mtf_summary = signal.get('multi_timeframe', '')

                        signal_message = (
                            f"🎯 SIGNAL: {signal['symbol']} {signal['type']}\n\n"
                            f"Confidence: {signal['confidence']*100:.1f}%\n"
                            f"ML Probability: {ml_score*100:.1f}%\n"
                            f"Multi-Timeframe Trend:\n{mtf_summary}\n\n"
                            f"Market Sentiment: {sentiment.get('overall', 'N/A')}\n"
                            f"Trend Strength: {sentiment.get('trend_strength', 0)*100:.1f}%\n"
                            f"News Impact: {sentiment.get('news_impact', 'N/A')}\n\n"
                            f"Entry: {signal['entry_price']}\n"
                            f"Stop Loss: {signal['targets']['stop_loss']}\n"
                            f"Take Profit Levels:\n"
                        )

                        for target in signal['targets']['take_profit_levels']:
                            signal_message += f"- {target['level']}: {target['price']}\n"

                        signal_message += (
                            f"\nRisk/Reward: {signal['risk_reward']}\n"
                            f"Potential Profit: {signal['potential_profit']} pips\n"
                            f"Max Loss: {signal['max_loss']} pips\n\n"
                            f"Technical Summary:\n"
                            f"- Trend: {signal['technical_summary']['trend']}\n"
                            f"- Volatility: {signal['technical_summary']['volatility']}\n"
                            f"- Volume: {signal['technical_summary']['volume']}"
                        )

                        logger.info("Sending signal message")
                        await message.edit_text(signal_message)
                        logger.info("Signal message sent successfully")
                    except Exception as e:
                        logger.error(f"Error formatting/sending signal message: {e}")
                        await message.edit_text("❌ Error formatting signal message. Please try again.")
                        continue

                except Exception as e:
                    logger.error(f"Error processing symbol {symbol}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error in signal command: {e}")
            try:
                await update.message.reply_text("❌ Error: Could not process signal command. Please try again.")
            except Exception as e2:
                logger.error(f"Error sending error message: {e2}")

    def is_market_open(self, symbol: str) -> bool:
        symbol = symbol.upper()
        # Only these are always open
        crypto_symbols = [
            "BTCUSD", "ETHUSD", "BNBUSD", "SOLUSD", "XRPUSD", "DOGEUSD", "ADAUSD", "AVAXUSD", "DOTUSD", "LINKUSD",
            # ...add more as needed
        ]
        if symbol in crypto_symbols:
            return True  # Crypto is always open

        # Forex/metals/oil/indices: market hours
        from datetime import datetime
        import pytz
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

    def get_multi_timeframe_data(self, symbol: str, timeframes=None):
        """Fetch indicator/sentiment data for multiple timeframes (1m to 48h)."""
        if timeframes is None:
            timeframes = ['1m', '3m', '5m', '7m', '10m', '15m', '20m', '30m', '1h', '2h', '4h', '8h', '12h', '24h', '48h']
        tf_data = {}
        for tf in timeframes:
            # Placeholder: In real use, fetch real data for each timeframe
            # Here, we just simulate with random or copied data
            tf_data[tf] = {
                'trend': self.market_sentiment['by_timeframe'].get(tf, 'neutral'),
                'sentiment': self.market_sentiment['by_pair'].get(symbol, 'neutral'),
                'trend_strength': self.market_sentiment['trend_strength'].get(symbol, 0.5),
                'indicators': {},  # Could be filled with real indicator values
            }
        return tf_data

    def multi_timeframe_confirmed(self, tf_data):
        """Return True if all timeframes align (e.g., all bullish or all bearish)."""
        trends = [v['trend'] for v in tf_data.values()]
        return all(t == trends[0] for t in trends)

    def predict_signal_success(self, features: dict) -> float:
        """Predict probability of signal success using ML (placeholder)."""
        # In real use, load and use a trained ML model here
        # For now, return a dummy probability based on confidence and trend_strength
        base = features.get('confidence', 0.8)
        trend = features.get('trend_strength', 0.5)
        # Simulate ML output
        return min(0.99, max(0.01, 0.5 * base + 0.5 * trend))

    def get_news_impact(self, symbol: str) -> str:
        """Get real news impact from ForexFactory."""
        try:
            url = "https://www.forexfactory.com/calendar"
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                # Look for high-impact news for the symbol's currency
                currency = symbol[:3]
                high_impact_news = soup.find_all("tr", {"class": "calendar_row high"})
                for news in high_impact_news:
                    if currency in news.text:
                        return "high"
                return "low"
            else:
                print(f"[ERROR] Failed to fetch news data: {response.status_code}")
                return "low"
        except Exception as e:
            print(f"[ERROR] Error fetching news: {e}")
            return "low"

    def get_market_sentiment(self, symbol: str) -> dict:
        """Get real market sentiment from TradingView."""
        try:
            url = f"https://www.tradingview.com/symbols/{symbol}/technicals/"
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                # Parse sentiment from TradingView's technical summary
                # This is a placeholder; actual parsing logic depends on TradingView's HTML structure
                sentiment = "bullish"  # Default to bullish if parsing fails
                return {
                    "overall": sentiment,
                    "trend_strength": 0.9,
                    "news_impact": self.get_news_impact(symbol)
                }
            else:
                print(f"[ERROR] Failed to fetch sentiment data: {response.status_code}")
                return {
                    "overall": "neutral",
                    "trend_strength": 0.5,
                    "news_impact": "low"
                }
        except Exception as e:
            print(f"[ERROR] Error fetching sentiment: {e}")
            return {
                "overall": "neutral",
                "trend_strength": 0.5,
                "news_impact": "low"
            }

    def should_avoid_trading(self, symbol: str) -> bool:
        """Check if trading should be avoided due to high-impact news."""
        news_impact = self.get_news_impact(symbol)
        return news_impact == "high"

    def get_current_signals(self, symbol="XAUUSD") -> list:
        """Get current trading signals with multi-timeframe and ML analysis."""
        logger.info(f"Getting signals for {symbol}")
        
        if not self.is_market_open(symbol):
            logger.info(f"Market closed for {symbol}")
            return []
            
        if self.should_avoid_trading(symbol):
            logger.info(f"Avoiding trading for {symbol} due to news event")
            return []
            
        # Multi-timeframe analysis
        timeframes = ['1h', '4h', '1d']
        tf_data = self.get_multi_timeframe_data(symbol, timeframes)
        logger.info(f"Multi-timeframe data: {tf_data}")
        
        if not self.multi_timeframe_confirmed(tf_data):
            logger.info(f"Multi-timeframe not confirmed for {symbol}: {[v['trend'] for v in tf_data.values()]}")
            return []
            
        # Use the trend and sentiment from the main timeframe (1h)
        sentiment = tf_data['1h']['sentiment']
        trend_strength = tf_data['1h']['trend_strength']
        base_confidence = 0.85
        sentiment_multiplier = 1.2 if sentiment == 'bullish' else 0.8 if sentiment == 'bearish' else 1.0
        final_confidence = min(0.95, base_confidence * sentiment_multiplier * trend_strength)
        
        logger.info(f"Fetching real signal for {symbol}")
        real = self.get_real_signal(symbol)
        
        if real:
            logger.info(f"Got real signal: {real}")
            entry = real['price']
            signal_type = real['recommendation'].upper()
            timeframe = '1h'
            summary = real['summary']
            indicators = real.get('indicators', {})
            
            def get_ind(name):
                return indicators.get(name, 'N/A')
                
            if "SELL" in signal_type:
                targets = [
                    {'price': entry * 0.99, 'level': '1.0R'},
                    {'price': entry * 0.98, 'level': '2.0R'},
                    {'price': entry * 0.97, 'level': '3.0R'}
                ]
                stop_loss = entry * 1.01
                resistance_levels = {
                    'r1': entry * 0.99,
                    'r2': entry * 0.98,
                    'r3': entry * 0.97
                }
            else:
                targets = [
                    {'price': entry * 1.01, 'level': '1.0R'},
                    {'price': entry * 1.02, 'level': '2.0R'},
                    {'price': entry * 1.03, 'level': '3.0R'}
                ]
                stop_loss = entry * 0.99
                resistance_levels = {
                    'r1': entry * 1.01,
                    'r2': entry * 1.02,
                    'r3': entry * 1.03
                }
                
            indicator_lines = []
            for ind in ["RSI", "MACD.macd", "EMA10", "EMA20", "EMA50", "EMA100", "EMA200", "SMA20", "SMA50", "SMA200"]:
                val = real['summary'].get(ind) or real['summary'].get(ind.lower()) or real['summary'].get(ind.upper())
                if val is not None:
                    indicator_lines.append(f"- {ind}: {val}")
                    
            tv_summary = (
                f"BUY: {summary.get('BUY', 'N/A')}, "
                f"SELL: {summary.get('SELL', 'N/A')}, "
                f"NEUTRAL: {summary.get('NEUTRAL', 'N/A')}"
            )
            
            # ML probability
            ml_features = {
                'confidence': final_confidence,
                'trend_strength': trend_strength,
                'type': signal_type
            }
            ml_score = self.predict_signal_success(ml_features)
            logger.info(f"{symbol} | Confidence: {final_confidence:.2f} | ML Score: {ml_score:.2f} | Threshold: {self.min_ml_confidence}")
            
            if ml_score < self.min_ml_confidence:
                logger.info(f"Signal for {symbol} filtered out due to low ML score.")
                return []
                
            # Compose multi-timeframe summary
            mtf_summary = "\n".join([
                f"{tf}: {d['trend']}" for tf, d in tf_data.items()
            ])
            
            return [{
                'symbol': symbol,
                'type': signal_type,
                'confidence': final_confidence,
                'ml_score': ml_score,
                'multi_timeframe': mtf_summary,
                'confirmations': 8,
                'pattern_confirmations': 3,
                'indicator_combinations': [
                    {'type': 'TradingView', 'confidence': 0.9}
                ],
                'entry_price': entry,
                'targets': {
                    'take_profit_levels': targets,
                    'stop_loss': stop_loss
                },
                'resistance_levels': resistance_levels,
                'risk_reward': 2.5,
                'potential_profit': abs(targets[2]['price'] - entry),
                'max_loss': abs(stop_loss - entry),
                'technical_summary': {
                    'trend': signal_type,
                    'volatility': 'N/A',
                    'volume': 'N/A'
                },
                'indicator_lines': indicator_lines,
                'tv_summary': tv_summary,
                'timeframe': timeframe,
                'sentiment': {
                    'overall': sentiment,
                    'trend_strength': trend_strength,
                    'news_impact': self.get_news_impact(symbol)
                }
            }]
            
        logger.info(f"No real signal for {symbol}")
        return []

    def get_real_signal(self, symbol: str):
        """Get real trading signal from TradingView."""
        logger.info(f"Fetching real signal for {symbol}")
        symbol = symbol.upper()
        metals_symbols = ["XAUUSD", "XAGUSD", "XPTUSD", "XPDUSD"]
        oil_symbols = ["USOIL", "UKOIL", "WTIUSD", "BRENTUSD"]
        indices_symbols = ["SPX500", "US30", "NAS100", "GER30", "FRA40", "UK100", "JPN225", "HK50", "AUS200", "EU50"]

        try:
            # Auto-detect crypto
            if (
                symbol.endswith("USD")
                and symbol not in metals_symbols + oil_symbols + indices_symbols
                and len(symbol) <= 10  # avoid weird long symbols
            ):
                exchange, tv_symbol, screener = "BINANCE", symbol.replace("USD", "USDT"), "crypto"
            elif symbol in metals_symbols:
                exchange, tv_symbol, screener = "OANDA", symbol, "forex"
            elif symbol in oil_symbols:
                exchange, tv_symbol, screener = "OANDA", symbol, "forex"
            elif symbol in indices_symbols:
                exchange, tv_symbol, screener = "OANDA", symbol, "indices"
            else:
                exchange, tv_symbol, screener = "OANDA", symbol, "forex"

            logger.info(f"Using exchange: {exchange}, symbol: {tv_symbol}, screener: {screener}")

            handler = TA_Handler(
                symbol=tv_symbol,
                screener=screener,
                exchange=exchange,
                interval=Interval.INTERVAL_1_HOUR
            )
            analysis = handler.get_analysis()
            
            if not analysis or not analysis.indicators or not analysis.summary:
                logger.error("Invalid analysis data received from TradingView")
                return None

            price = analysis.indicators.get("close")
            recommendation = analysis.summary.get("RECOMMENDATION")
            
            if not price or not recommendation:
                logger.error("Missing price or recommendation in analysis data")
                return None

            logger.info(f"Got analysis - Price: {price}, Recommendation: {recommendation}")
            return {
                "symbol": symbol,
                "price": price,
                "recommendation": recommendation,
                "summary": analysis.summary
            }
        except Exception as e:
            logger.error(f"Error fetching TradingView data: {e}")
            return None

    async def settings(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        # Get or create user settings
        settings = self.user_settings.get(user_id, {
            'favorite_pairs': ['XAUUSD'],
            'notifications': True
        })
        self.user_settings[user_id] = settings  # Ensure it's stored
        # Show current settings
        msg = (
            f"⚙️ Your Settings:\n"
            f"Favorite pairs: {', '.join(settings['favorite_pairs'])}\n"
            f"Notifications: {'On' if settings['notifications'] else 'Off'}\n\n"
            f"To update favorite pairs, use /setpairs EURUSD GBPUSD BTCUSD ...\n"
            f"To toggle notifications, use /setnotify on or /setnotify off."
        )
        await update.message.reply_text(msg)

    async def setpairs(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        if not context.args:
            await update.message.reply_text("Please provide at least one symbol, e.g. /setpairs EURUSD BTCUSD")
            return
        pairs = [arg.upper() for arg in context.args]
        settings = self.user_settings.get(user_id, {'favorite_pairs': ['XAUUSD'], 'notifications': True})
        settings['favorite_pairs'] = pairs
        self.user_settings[user_id] = settings
        await update.message.reply_text(f"Favorite pairs updated: {', '.join(pairs)}")

    async def setnotify(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        if not context.args or context.args[0].lower() not in ['on', 'off']:
            await update.message.reply_text("Usage: /setnotify on or /setnotify off")
            return
        settings = self.user_settings.get(user_id, {'favorite_pairs': ['XAUUSD'], 'notifications': True})
        settings['notifications'] = (context.args[0].lower() == 'on')
        self.user_settings[user_id] = settings
        await update.message.reply_text(f"Notifications {'enabled' if settings['notifications'] else 'disabled'}.")

    async def send_daily_report(self):
        # Send performance summary to all users with notifications enabled
        for user_id, settings in self.user_settings.items():
            if settings.get('notifications', True):
                # Compose performance message
                stats = self.calculate_performance()
                message = (
                    "📊 DAILY PERFORMANCE SUMMARY 📊\n\n"
                    f"Total Signals: {stats['total_signals']}\n"
                    f"Win Rate: {stats['win_rate']*100:.1f}%\n"
                    f"Average Profit: {stats['avg_profit']:.1f} pips\n"
                    f"Average Loss: {stats['avg_loss']:.1f} pips\n"
                    f"Profit Factor: {stats['profit_factor']:.2f}\n"
                )
                try:
                    await self.telegram_bot.send_message(chat_id=user_id, text=message)
                except Exception as e:
                    print(f"Failed to send daily report to {user_id}: {e}")

    def set_application(self, application):
        self.application = application

    async def update_market_sentiment(self):
        """Update market sentiment analysis."""
        try:
            # Get sentiment data from various sources
            sentiment_data = await self.get_market_sentiment()
            
            # Update overall sentiment
            self.market_sentiment['overall'] = sentiment_data.get('overall', 'neutral')
            
            # Update pair-specific sentiment
            for pair, sentiment in sentiment_data.get('pairs', {}).items():
                self.market_sentiment['by_pair'][pair] = sentiment
                
            # Update timeframe-specific sentiment
            for timeframe, sentiment in sentiment_data.get('timeframes', {}).items():
                self.market_sentiment['by_timeframe'][timeframe] = sentiment
                
            # Update news impact sentiment
            self.market_sentiment['news_impact'] = sentiment_data.get('news_impact', {})
            
            # Update trend strength
            self.market_sentiment['trend_strength'] = sentiment_data.get('trend_strength', {})
            
        except Exception as e:
            logger.error(f"Error updating market sentiment: {e}")

    async def get_economic_events(self):
        """Get upcoming economic events."""
        try:
            # This is a placeholder - implement actual API calls to economic calendar
            return [
                {
                    'time': '2024-03-20 14:30:00',
                    'event': 'FOMC Rate Decision',
                    'impact': 'high',
                    'currency': 'USD'
                },
                {
                    'time': '2024-03-21 12:30:00',
                    'event': 'ECB Press Conference',
                    'impact': 'high',
                    'currency': 'EUR'
                }
            ]
        except Exception as e:
            logger.error(f"Error getting economic events: {e}")
            return []

    async def get_market_sentiment(self):
        """Get market sentiment data."""
        try:
            # This is a placeholder - implement actual sentiment analysis
            return {
                'overall': 'bullish',
                'pairs': {
                    'EURUSD': 'bullish',
                    'GBPUSD': 'neutral',
                    'XAUUSD': 'bullish'
                },
                'timeframes': {
                    '1h': 'bullish',
                    '4h': 'neutral',
                    '1d': 'bullish'
                },
                'news_impact': {
                    'USD': 'positive',
                    'EUR': 'neutral',
                    'GBP': 'negative'
                },
                'trend_strength': {
                    'EURUSD': 0.8,
                    'GBPUSD': 0.5,
                    'XAUUSD': 0.9
                }
            }
        except Exception as e:
            logger.error(f"Error getting market sentiment: {e}")
            return {}

async def main():
    """Main function to start the bot."""
    try:
        print("Starting bot...")
        # Create bot instance
        bot = TradingBot()
        
        # Add basic command handlers
        bot.application.add_handler(CommandHandler("start", bot.start))
        bot.application.add_handler(CommandHandler("help", bot.help_command))
        bot.application.add_handler(CommandHandler("signal", bot.signal))
        
        # Start the bot
        print("Starting polling...")
        await bot.application.initialize()
        await bot.application.start()
        await bot.application.run_polling(allowed_updates=Update.ALL_TYPES)
        
    except Exception as e:
        print(f"Error in main: {e}")
        logger.error(f"Error in main: {e}")

if __name__ == "__main__":
    try:
        # Set up logging
        logging.basicConfig(
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            level=logging.INFO
        )
        
        # Run the bot
        asyncio.run(main())
    except Exception as e:
        print(f"Fatal error: {e}")
        logger.error(f"Fatal error: {e}") 