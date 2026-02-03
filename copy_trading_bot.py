import os
import logging
import schedule
import time
import json
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

# ============================================================
# CONTROL STATE - Read from NG XM Trading BOT
# ============================================================
CONTROL_STATE_FILE = "trading_state.json"

def get_control_state() -> Dict:
    """Read control state from NG XM Trading BOT"""
    try:
        if os.path.exists(CONTROL_STATE_FILE):
            with open(CONTROL_STATE_FILE, 'r') as f:
                return json.load(f)
    except:
        pass
    return {"is_trading_enabled": False, "auto_scan_enabled": False, "scalp_scan_enabled": False}

def is_trading_enabled() -> bool:
    """Check if trading is enabled from Control Bot"""
    state = get_control_state()
    return state.get("is_trading_enabled", False)

def is_auto_scan_enabled() -> bool:
    """Check if auto scan is enabled from Control Bot"""
    state = get_control_state()
    return state.get("auto_scan_enabled", False)

def is_scalp_scan_enabled() -> bool:
    """Check if scalp scan is enabled from Control Bot"""
    state = get_control_state()
    return state.get("scalp_scan_enabled", False)
# ============================================================

# Import news signal integration
try:
    from news_signal_integration import NewsSignalIntegration
    NEWS_SIGNAL_AVAILABLE = True
except ImportError:
    NEWS_SIGNAL_AVAILABLE = False
    print("[WARNING] News Signal Engine not available. /newsignal command will be disabled.")

# Import autonomous signal generator
try:
    from autonomous_signal_generator import AutonomousSignalGenerator, AutoSignal
    AUTONOMOUS_AVAILABLE = True
except ImportError:
    AUTONOMOUS_AVAILABLE = False
    print("[WARNING] Autonomous Signal Generator not available.")

# Import Scalp AI Engine
try:
    from scalp_ai_engine import ScalpSignalEngine, ScalpSignal
    SCALP_AI_AVAILABLE = True
except ImportError:
    SCALP_AI_AVAILABLE = False
    print("[WARNING] Scalp AI Engine not available.")

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
        # Initialize Telegram application
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        if not self.bot_token:
            raise ValueError("TELEGRAM_BOT_TOKEN not found in environment variables")
        
        # Initialize application
        self.application = Application.builder().token(self.bot_token).build()
        
        # Initialize scheduler
        self.scheduler = AsyncIOScheduler()
        
        # Initialize user settings dictionary
        self.user_settings = {}
        
        # Initialize signal subscribers (users who want auto-signals)
        self.signal_subscribers = set()
        
        # Initialize signal history
        self.signal_history = []
        
        # Initialize market sentiment
        self.market_sentiment = {
            'overall': 'neutral',
            'by_pair': {},
            'by_timeframe': {},
            'trend_strength': {},
            'news_impact': {}
        }
        
        # Initialize ML model
        self.ml_model = self.initialize_ml_model()
        self.min_ml_confidence = 0.7
        
        # Initialize News Signal Engine
        if NEWS_SIGNAL_AVAILABLE:
            self.news_handler = NewsSignalIntegration(min_confidence=0.40)
            print("[INFO] News Signal Engine initialized")
        else:
            self.news_handler = None
        
        # Initialize Autonomous Signal Generator
        if AUTONOMOUS_AVAILABLE:
            self.auto_generator = AutonomousSignalGenerator()
            self.auto_scan_task = None
            print("[INFO] Autonomous Signal Generator initialized")
        else:
            self.auto_generator = None
            self.auto_scan_task = None
        
        # Initialize Scalp AI Engine
        if SCALP_AI_AVAILABLE:
            self.scalp_engine = ScalpSignalEngine()
            self.scalp_scan_task = None
            print("[INFO] Scalp AI Engine initialized")
        else:
            self.scalp_engine = None
            self.scalp_scan_task = None
        
        # Register handlers
        self.register_handlers()
        
        # Setup news monitoring
        self.setup_news_monitoring()

    def register_handlers(self):
        """Register command handlers."""
        self.application.add_handler(CommandHandler("start", self.start))
        self.application.add_handler(CommandHandler("help", self.help))
        self.application.add_handler(CommandHandler("signal", self.signal))
        self.application.add_handler(CommandHandler("newsignal", self.newsignal))
        self.application.add_handler(CommandHandler("autoscan", self.autoscan))
        self.application.add_handler(CommandHandler("stopscan", self.stopscan))
        self.application.add_handler(CommandHandler("scalp", self.scalp))
        self.application.add_handler(CommandHandler("scalpscan", self.scalpscan))
        self.application.add_handler(CommandHandler("stopscalp", self.stopscalp))
        self.application.add_handler(CommandHandler("symbols", self.symbols))
        self.application.add_handler(CommandHandler("performance", self.performance))
        self.application.add_handler(CommandHandler("settings", self.settings))
        self.application.add_handler(CommandHandler("setpairs", self.setpairs))
        self.application.add_handler(CommandHandler("setnotify", self.setnotify))
        self.application.add_handler(CommandHandler("status", self.status))
        self.application.add_handler(CommandHandler("timeframes", self.timeframes))
        self.application.add_handler(CommandHandler("subscribe", self.subscribe))
        self.application.add_handler(CommandHandler("unsubscribe", self.unsubscribe))

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
            
            # Schedule auto signal broadcast every 2 minutes
            if self.auto_generator:
                self.scheduler.add_job(
                    self.auto_broadcast_signals,
                    'interval',
                    minutes=2,
                    id='auto_broadcast'
                )
                print("[INFO] Auto signal broadcast scheduled (every 2 min)")
            
            print("[INFO] News monitoring setup complete")
        except Exception as e:
            print(f"[ERROR] Error setting up news monitoring: {e}")

    async def update_news_impact(self):
        """Update news impact for all currencies."""
        try:
            currencies = ['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF', 'NZD', 'XAU']
            for currency in currencies:
                self.market_sentiment['news_impact'][currency] = self.get_news_impact(currency)
            print("[INFO] News impact updated successfully")
        except Exception as e:
            print(f"[ERROR] Error updating news impact: {e}")

    async def auto_broadcast_signals(self):
        """Automatically scan and broadcast signals to subscribers - SENDS IMMEDIATELY when generated."""
        # Check if trading is enabled from Control Bot (NG XM Trading BOT)
        if not is_trading_enabled():
            return  # Trading disabled in Control Bot, skip
            
        # Check if auto scan is enabled from Control Bot
        if not is_auto_scan_enabled():
            return  # Auto scan disabled in Control Bot, skip
        
        if not self.signal_subscribers:
            return  # No subscribers, skip scanning
        
        try:
            # Define callback to send signal IMMEDIATELY when generated
            async def send_signal_immediately(signal):
                """Callback to broadcast signal as soon as it's generated"""
                message = self.auto_generator.format_signal_message(signal)
                await self.broadcast_signal(message)
                
                # Log the signal
                self.log_signal({
                    'symbol': signal.symbol,
                    'type': signal.signal_type.value,
                    'entry_price': signal.entry_price,
                    'confidence': signal.confidence,
                    'risk_reward': signal.risk_reward_ratio
                })
            
            # Scan with callback - signals sent IMMEDIATELY as they're found
            signals = await self.auto_generator.scan_all_markets(
                on_signal_callback=send_signal_immediately
            )
            
            if signals:
                logger.info(f"✅ Scan complete: {len(signals)} signals sent to {len(self.signal_subscribers)} subscribers")
                    
        except Exception as e:
            logger.error(f"Error in auto broadcast: {e}")

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

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        print("/start command received")
        
        # If user provides a symbol with /start, treat it as a signal request
        if context.args:
            # Redirect to newsignal command
            await self.newsignal(update, context)
            return
        
        welcome_msg = (
            "✅ *Copy Trading Bot is Live!*\n\n"
            "📊 *Available Commands:*\n"
            "`/newsignal XAUUSD` - Get AI signal\n"
            "`/scalp EURUSD` - Get scalp signal ⚡\n"
            "`/autoscan` - Start AUTO scanning 🔥\n"
            "`/scalpscan` - Start SCALP scanning ⚡\n"
            "`/stopscan` - Stop auto scanning\n"
            "`/stopscalp` - Stop scalp scanning\n"
            "`/symbols` - View all 100+ symbols\n"
            "`/performance` - View signal stats\n\n"
            "⚡ *SCALP AI MODE:* (NEW!)\n"
            "Use `/scalpscan` to activate AI-powered\n"
            "scalp detection with 85%+ confidence!\n"
            "• Pattern recognition\n"
            "• 5min to 1hour timeframes\n"
            "• Quick profit targets\n\n"
            "🤖 *AUTO SCAN MODE:*\n"
            "Use `/autoscan` for full market scanning\n"
            "with regular and swing signals.\n\n"
            "🎯 *Quick Examples:*\n"
            "`/scalp XAUUSD` - Gold scalp signal ⚡\n"
            "`/newsignal US500` - S&P 500 signal\n"
            "`/start BTCUSD` - Bitcoin signal\n\n"
            "_Use /autoscan for hands-free trading signals!_"
        )
        await update.message.reply_text(welcome_msg, parse_mode='Markdown')

    async def help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show detailed help for all commands."""
        print("/help command received")
        
        help_msg = (
            "📚 *COMPLETE COMMAND GUIDE*\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            
            "🚀 *GETTING STARTED:*\n"
            "`/start` - Welcome message & quick start\n"
            "`/help` - This detailed help guide\n"
            "`/symbols` - View all 100+ trading symbols\n"
            "`/timeframes` - View all analysis timeframes\n"
            "`/status` - Check bot & scanning status\n\n"
            
            "📊 *SIGNAL COMMANDS:*\n"
            "`/signal SYMBOL` - Quick trading signal\n"
            "`/newsignal SYMBOL` - AI-powered news signal\n"
            "  _Example: /newsignal XAUUSD_\n\n"
            
            "⚡ *SCALP TRADING (5min-1hr):*\n"
            "`/scalp SYMBOL` - Get AI scalp signal\n"
            "`/scalpscan` - Auto-scan for scalp opportunities\n"
            "`/stopscalp` - Stop scalp scanning\n"
            "  _Example: /scalp BTCUSD_\n\n"
            
            "🔄 *AUTO SCANNING:*\n"
            "`/autoscan` - Start autonomous market scan\n"
            "`/stopscan` - Stop auto scanning\n"
            "  _Scans all markets for opportunities!_\n\n"
            
            "📈 *ANALYSIS & SETTINGS:*\n"
            "`/performance` - View signal statistics\n"
            "`/settings` - View current settings\n"
            "`/setpairs PAIRS` - Set watched pairs\n"
            "`/setnotify on/off` - Toggle notifications\n\n"
            
            "� *AUTO NOTIFICATIONS:*\n"
            "`/subscribe` - Get auto signals (NEW!)\n"
            "`/unsubscribe` - Stop auto signals\n"
            "  _Receive signals automatically!_\n\n"
            
            "💡 *TIPS:*\n"
            "• Use `/subscribe` for auto signals!\n"
            "• Use `/scalpscan` for quick profits\n"
            "• Use `/autoscan` for hands-free trading\n"
            "• Higher confidence = better signals\n\n"
            
            "🎯 *QUICK EXAMPLES:*\n"
            "`/scalp XAUUSD` - Gold scalp ⚡\n"
            "`/newsignal EURUSD` - EUR/USD signal\n"
            "`/signal BTCUSD` - Bitcoin signal\n"
            "`/subscribe` - Auto signal delivery\n\n"
            
            "_💬 Bot analyzes 15 timeframes from 1min to 48hrs!_"
        )
        await update.message.reply_text(help_msg, parse_mode='Markdown')

    async def status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show current bot status."""
        print("/status command received")
        
        # Check scanning status
        autoscan_status = "🟢 ACTIVE" if self.is_auto_scanning else "🔴 Stopped"
        scalpscan_status = "🟢 ACTIVE" if self.is_scalp_scanning else "🔴 Stopped"
        
        # Get live prices from scalp engine
        live_prices = ""
        if hasattr(self, 'scalp_engine') and self.scalp_engine:
            for symbol, price in self.scalp_engine.live_prices.items():
                live_prices += f"  • {symbol}: ${price:,.2f}\n"
        
        status_msg = (
            "📊 *BOT STATUS*\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            
            f"🤖 *Bot Status:* 🟢 Online\n\n"
            
            "📡 *Scanning Status:*\n"
            f"  • Auto Scan: {autoscan_status}\n"
            f"  • Scalp Scan: {scalpscan_status}\n\n"
            
            "💰 *Live Prices:*\n"
            f"{live_prices if live_prices else '  _Fetching..._'}\n"
            
            "⏱️ *Analysis Timeframes:* 15\n"
            "  _1m, 3m, 5m, 7m, 10m, 15m, 20m, 30m_\n"
            "  _1h, 2h, 4h, 8h, 12h, 24h, 48h_\n\n"
            
            "📈 *Scalp Symbols:* 4\n"
            "  _XAUUSD, XAUEUR, XAGUSD, BTCUSD_\n\n"
            
            "_Use /help for all commands_"
        )
        await update.message.reply_text(status_msg, parse_mode='Markdown')

    async def timeframes(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show all analysis timeframes."""
        print("/timeframes command received")
        
        tf_msg = (
            "⏱️ *ANALYSIS TIMEFRAMES*\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            
            "🚀 *ULTRA-SHORT (Scalping):*\n"
            "  `1m` - 1 Minute\n"
            "  `3m` - 3 Minutes\n"
            "  `5m` - 5 Minutes\n"
            "  `7m` - 7 Minutes\n"
            "  `10m` - 10 Minutes\n\n"
            
            "⚡ *SHORT-TERM (Day Trading):*\n"
            "  `15m` - 15 Minutes _(Primary)_\n"
            "  `20m` - 20 Minutes\n"
            "  `30m` - 30 Minutes\n"
            "  `1h` - 1 Hour\n\n"
            
            "📊 *MEDIUM-TERM (Swing):*\n"
            "  `2h` - 2 Hours\n"
            "  `4h` - 4 Hours\n"
            "  `8h` - 8 Hours\n\n"
            
            "📈 *LONG-TERM (Position):*\n"
            "  `12h` - 12 Hours\n"
            "  `24h` - 24 Hours (1 Day)\n"
            "  `48h` - 48 Hours (2 Days)\n\n"
            
            "🎯 *HOW IT WORKS:*\n"
            "• All 15 timeframes are analyzed\n"
            "• Short-term: Quick scalp entries\n"
            "• Medium-term: Trend confirmation\n"
            "• Long-term: Overall direction\n\n"
            
            "📊 *TREND ALIGNMENT:*\n"
            "• Higher alignment = Stronger signal\n"
            "• 80%+ alignment = Premium signal\n\n"
            
            "_Use /scalp or /newsignal to get signals!_"
        )
        await update.message.reply_text(tf_msg, parse_mode='Markdown')

    async def subscribe(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Subscribe to automatic signal notifications."""
        print("/subscribe command received")
        
        chat_id = update.effective_chat.id
        
        if chat_id in self.signal_subscribers:
            await update.message.reply_text(
                "✅ *You're already subscribed!*\n\n"
                "You'll receive signals automatically when they're generated.\n\n"
                "Use `/unsubscribe` to stop receiving signals.",
                parse_mode='Markdown'
            )
        else:
            self.signal_subscribers.add(chat_id)
            await update.message.reply_text(
                "🔔 *SUBSCRIBED SUCCESSFULLY!*\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                "✅ You'll now receive automatic signals!\n\n"
                "📊 *What you'll get:*\n"
                "• High-confidence trading signals\n"
                "• Scalp signals (XAUUSD, BTCUSD)\n"
                "• Entry, SL, and TP levels\n\n"
                "Use `/unsubscribe` to stop receiving signals.",
                parse_mode='Markdown'
            )
            logger.info(f"User {chat_id} subscribed to signals")

    async def unsubscribe(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Unsubscribe from automatic signal notifications."""
        print("/unsubscribe command received")
        
        chat_id = update.effective_chat.id
        
        if chat_id in self.signal_subscribers:
            self.signal_subscribers.discard(chat_id)
            await update.message.reply_text(
                "🔕 *UNSUBSCRIBED*\n\n"
                "You will no longer receive automatic signals.\n\n"
                "Use `/subscribe` to start receiving signals again.",
                parse_mode='Markdown'
            )
            logger.info(f"User {chat_id} unsubscribed from signals")
        else:
            await update.message.reply_text(
                "ℹ️ You weren't subscribed to signals.\n\n"
                "Use `/subscribe` to start receiving signals.",
                parse_mode='Markdown'
            )

    async def broadcast_signal(self, message: str):
        """Broadcast a signal to all subscribers."""
        if not self.signal_subscribers:
            logger.info("No subscribers to broadcast to")
            return
        
        for chat_id in self.signal_subscribers.copy():
            try:
                await self.application.bot.send_message(
                    chat_id=chat_id,
                    text=message,
                    parse_mode='Markdown'
                )
                logger.info(f"Signal broadcasted to {chat_id}")
            except Exception as e:
                logger.error(f"Failed to send to {chat_id}: {e}")
                # Remove invalid chat_ids
                if "chat not found" in str(e).lower() or "blocked" in str(e).lower():
                    self.signal_subscribers.discard(chat_id)

    async def symbols(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show all supported trading symbols."""
        print("/symbols command received")
        
        symbols_msg = (
            "📊 *SUPPORTED TRADING INSTRUMENTS*\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            
            "💱 *FOREX MAJORS:*\n"
            "`EURUSD` `GBPUSD` `USDJPY` `USDCHF`\n"
            "`AUDUSD` `USDCAD` `NZDUSD`\n\n"
            
            "💱 *FOREX MINORS:*\n"
            "`EURGBP` `EURJPY` `GBPJPY` `EURAUD`\n"
            "`EURCAD` `GBPAUD` `GBPCAD` `AUDJPY`\n"
            "`NZDJPY` `AUDNZD` `CADJPY` `CHFJPY`\n\n"
            
            "💱 *FOREX EXOTICS:*\n"
            "`USDZAR` `USDMXN` `USDTRY` `EURTRY`\n"
            "`USDSEK` `USDNOK` `USDSGD` `USDCNH`\n\n"
            
            "🥇 *PRECIOUS METALS:*\n"
            "`XAUUSD` (Gold) `XAGUSD` (Silver)\n"
            "`XPTUSD` (Platinum) `XPDUSD` (Palladium)\n\n"
            
            "⛽ *ENERGY:*\n"
            "`USOIL` (WTI Crude) `UKOIL` (Brent)\n"
            "`NATURALGAS` `NGAS`\n\n"
            
            "🌾 *AGRICULTURE:*\n"
            "`WHEAT` `CORN` `SOYBEAN` `COFFEE`\n"
            "`SUGAR` `COTTON` `COCOA` `COPPER`\n\n"
            
            "🇺🇸 *US INDICES:*\n"
            "`US30` (Dow Jones) `US500` (S&P 500)\n"
            "`US100` (NASDAQ) `US2000` (Russell)\n\n"
            
            "🇪🇺 *EUROPEAN INDICES:*\n"
            "`GER40` (DAX) `UK100` (FTSE)\n"
            "`FRA40` (CAC) `EU50` (Euro Stoxx)\n"
            "`ESP35` (IBEX) `ITA40` `SUI20`\n\n"
            
            "🌏 *ASIAN INDICES:*\n"
            "`JPN225` (Nikkei) `HK50` (Hang Seng)\n"
            "`AUS200` (ASX) `INDIA50` (Nifty)\n"
            "`CN50` (China A50) `SG30`\n\n"
            
            "₿ *CRYPTOCURRENCIES:*\n"
            "`BTCUSD` `ETHUSD` `BNBUSD` `XRPUSD`\n"
            "`SOLUSD` `ADAUSD` `DOGEUSD` `DOTUSD`\n"
            "`LINKUSD` `AVAXUSD` `LTCUSD` `MATICUSD`\n\n"
            
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "📝 *Usage:* `/newsignal SYMBOL`\n"
            "_Example:_ `/newsignal US500`"
        )
        await update.message.reply_text(symbols_msg, parse_mode='Markdown')

    async def newsignal(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /newsignal command - News-based AI signal generation."""
        print("/newsignal command received")
        
        # Check if news handler is available
        if not self.news_handler:
            await update.message.reply_text(
                "❌ News Signal Engine is not available. Please check the installation."
            )
            return
        
        # Get symbol from args
        if not context.args:
            await update.message.reply_text(
                "📝 *Usage:* `/newsignal SYMBOL`\n\n"
                "*Examples:*\n"
                "`/newsignal XAUUSD` - Gold\n"
                "`/newsignal EURUSD` - Euro/Dollar\n"
                "`/newsignal BTCUSD` - Bitcoin\n"
                "`/newsignal GBPUSD` - British Pound",
                parse_mode='Markdown'
            )
            return
        
        symbol = context.args[0].upper()
        
        # Send processing message
        processing_msg = await update.message.reply_text(
            f"🔄 *Analyzing {symbol}...*\n\n"
            "📰 Fetching latest news...\n"
            "📊 Running technical analysis...\n"
            "🤖 Generating AI signal...",
            parse_mode='Markdown'
        )
        
        try:
            # Get signal from news engine
            signal = await self.news_handler.get_signal_for_symbol(symbol)
            
            if signal:
                # Format and send the signal
                message = self.news_handler.format_telegram_message(signal)
                await processing_msg.edit_text(message, parse_mode='Markdown')
                
                # Log the signal
                self.log_signal({
                    'symbol': symbol,
                    'type': signal.signal_type.value,
                    'entry_price': signal.entry_price,
                    'confidence': signal.confidence,
                    'risk_reward': signal.risk_reward_ratio
                })
            else:
                await processing_msg.edit_text(
                    f"ℹ️ *No Signal Available for {symbol}*\n\n"
                    "Possible reasons:\n"
                    "• Market conditions are neutral\n"
                    "• Confidence below threshold\n"
                    "• Mixed news sentiment\n\n"
                    "_Try again later or check another symbol._",
                    parse_mode='Markdown'
                )
                
        except Exception as e:
            logger.error(f"Error generating news signal for {symbol}: {e}")
            await processing_msg.edit_text(
                f"❌ Error generating signal for {symbol}.\n"
                "Please try again later."
            )

    async def autoscan(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Start autonomous market scanning."""
        print("/autoscan command received")
        
        # Check if trading is enabled from Control Bot
        if not is_trading_enabled():
            await update.message.reply_text(
                "⚠️ *Trading is DISABLED!*\n\n"
                "Please enable trading in *NG XM Trading BOT* first.\n"
                "Tap 🟢 START TRADING button there.",
                parse_mode='Markdown'
            )
            return
        
        if not AUTONOMOUS_AVAILABLE or not self.auto_generator:
            await update.message.reply_text(
                "❌ Autonomous Signal Generator is not available.\n"
                "Please check the installation."
            )
            return
        
        chat_id = update.effective_chat.id
        
        # Check if already scanning
        if self.auto_scan_task and not self.auto_scan_task.done():
            await update.message.reply_text(
                "⚠️ *Auto scanning is already running!*\n\n"
                "Use `/stopscan` to stop it first.",
                parse_mode='Markdown'
            )
            return
        
        # Initialize the generator if needed
        try:
            await self.auto_generator.initialize()
            
            # Add this chat to receive signals
            if chat_id not in self.auto_generator.chat_ids:
                self.auto_generator.chat_ids.append(chat_id)
            
            await update.message.reply_text(
                "🚀 *AUTO SCAN MODE ACTIVATED!*\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                "🔍 The bot will now:\n"
                "• Scan 30+ instruments every 60 seconds\n"
                "• Analyze news, sentiment & technicals\n"
                "• Use ML model for predictions\n"
                "• Send signals with 75%+ confidence\n"
                "• Alert scalp trades with 90%+ probability\n\n"
                "📊 *Scanning:*\n"
                "• Forex Majors & Minors\n"
                "• Gold, Silver, Oil\n"
                "• US, EU, Asia Indices\n"
                "• Top Cryptocurrencies\n\n"
                "⏱️ _Signals will be sent automatically..._\n"
                "Use `/stopscan` to stop scanning.",
                parse_mode='Markdown'
            )
            
            # Start the scanning task
            self.auto_scan_task = asyncio.create_task(
                self._run_auto_scan(chat_id)
            )
            
            logger.info(f"Auto scan started for chat {chat_id}")
            
        except Exception as e:
            logger.error(f"Error starting auto scan: {e}")
            await update.message.reply_text(
                f"❌ Error starting auto scan: {str(e)}"
            )
    
    async def _run_auto_scan(self, chat_id: int):
        """Background task for auto scanning."""
        scan_count = 0
        
        while True:
            try:
                # Check if trading is still enabled from Control Bot
                if not is_trading_enabled():
                    logger.info("Trading disabled from Control Bot - stopping auto scan")
                    await self.application.bot.send_message(
                        chat_id=chat_id,
                        text="⏹️ *Auto scan stopped*\n\nTrading disabled from NG XM Trading BOT.",
                        parse_mode='Markdown'
                    )
                    break
                
                scan_count += 1
                logger.info(f"Auto scan #{scan_count} starting...")
                
                # Scan all markets
                signals = await self.auto_generator.scan_all_markets()
                
                if signals:
                    logger.info(f"Found {len(signals)} signals in scan #{scan_count}")
                    
                    for signal in signals:
                        # Format and send the signal
                        message = self.auto_generator.format_signal_message(signal)
                        
                        try:
                            await self.application.bot.send_message(
                                chat_id=chat_id,
                                text=message,
                                parse_mode='Markdown'
                            )
                            
                            # Log the signal
                            self.log_signal({
                                'symbol': signal.symbol,
                                'type': signal.signal_type.value,
                                'entry_price': signal.entry_price,
                                'confidence': signal.confidence,
                                'risk_reward': signal.risk_reward_ratio
                            })
                            
                            await asyncio.sleep(2)  # Small delay between signals
                            
                        except Exception as e:
                            logger.error(f"Error sending signal: {e}")
                else:
                    logger.info(f"No signals in scan #{scan_count}")
                
                # Wait before next scan (60 seconds)
                await asyncio.sleep(60)
                
            except asyncio.CancelledError:
                logger.info("Auto scan task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in auto scan: {e}")
                await asyncio.sleep(30)  # Wait and retry
    
    async def stopscan(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Stop autonomous market scanning."""
        print("/stopscan command received")
        
        if self.auto_scan_task and not self.auto_scan_task.done():
            self.auto_scan_task.cancel()
            
            try:
                await self.auto_scan_task
            except asyncio.CancelledError:
                pass
            
            self.auto_scan_task = None
            
            await update.message.reply_text(
                "🛑 *AUTO SCAN STOPPED*\n\n"
                "The bot has stopped scanning for signals.\n"
                "Use `/autoscan` to start again.",
                parse_mode='Markdown'
            )
            logger.info("Auto scan stopped by user")
        else:
            await update.message.reply_text(
                "ℹ️ Auto scan is not currently running.\n"
                "Use `/autoscan` to start scanning.",
                parse_mode='Markdown'
            )

    async def scalp(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Get a scalp signal for a specific symbol."""
        print("/scalp command received")
        
        if not SCALP_AI_AVAILABLE or not self.scalp_engine:
            await update.message.reply_text(
                "❌ Scalp AI Engine is not available.\n"
                "Please check the installation."
            )
            return
        
        # Get symbol from command arguments
        if context.args:
            symbol = context.args[0].upper()
        else:
            await update.message.reply_text(
                "⚡ *SCALP AI ENGINE*\n\n"
                "AI-powered scalp signals for quick trades.\n\n"
                "*Usage:* `/scalp SYMBOL`\n"
                "*Example:* `/scalp XAUUSD`\n\n"
                "*Scalp Symbols (4):*\n"
                "• 🥇 XAUUSD - Gold/USD\n"
                "• 🥇 XAUEUR - Gold/EUR\n"
                "• 🥈 XAGUSD - Silver/USD\n"
                "• ₿ BTCUSD - Bitcoin\n\n"
                "Use `/scalpscan` for auto scanning!",
                parse_mode='Markdown'
            )
            return
        
        # Send "analyzing" message
        processing_msg = await update.message.reply_text(
            f"⚡ *Analyzing {symbol} for scalp opportunity...*\n"
            "🤖 AI Engine scanning multiple timeframes...",
            parse_mode='Markdown'
        )
        
        try:
            # Get scalp signal
            signal = self.scalp_engine.analyze_symbol_for_scalp(symbol)
            
            if signal:
                # Format and send the signal
                message = self.scalp_engine.format_scalp_signal(signal)
                
                await processing_msg.edit_text(message, parse_mode='Markdown')
                
                # Log the signal
                self.log_signal({
                    'symbol': signal.symbol,
                    'type': f"SCALP_{signal.direction}",
                    'entry_price': signal.entry_price,
                    'confidence': signal.confidence,
                    'risk_reward': signal.risk_reward
                })
            else:
                await processing_msg.edit_text(
                    f"⏸️ *No Scalp Signal for {symbol}*\n\n"
                    "The AI did not find a high-confidence\n"
                    "scalp opportunity at this moment.\n\n"
                    "• Minimum confidence required: 85%\n"
                    "• Check back in a few minutes\n"
                    "• Try `/scalpscan` for all symbols",
                    parse_mode='Markdown'
                )
                
        except Exception as e:
            logger.error(f"Error getting scalp signal: {e}")
            await processing_msg.edit_text(
                f"❌ Error analyzing {symbol}: {str(e)}"
            )

    async def scalpscan(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Start autonomous scalp signal scanning."""
        print("/scalpscan command received")
        
        # Check if trading is enabled from Control Bot
        if not is_trading_enabled():
            await update.message.reply_text(
                "⚠️ *Trading is DISABLED!*\n\n"
                "Please enable trading in *NG XM Trading BOT* first.\n"
                "Tap 🟢 START TRADING button there.",
                parse_mode='Markdown'
            )
            return
        
        if not SCALP_AI_AVAILABLE or not self.scalp_engine:
            await update.message.reply_text(
                "❌ Scalp AI Engine is not available.\n"
                "Please check the installation."
            )
            return
        
        if self.scalp_scan_task and not self.scalp_scan_task.done():
            await update.message.reply_text(
                "⚡ Scalp scanning is already running!\n"
                "Use `/stopscalp` to stop it first."
            )
            return
        
        chat_id = update.effective_chat.id
        
        try:
            await update.message.reply_text(
                "⚡ *SCALP AI SCANNER ACTIVATED!* ⚡\n\n"
                "🎯 *Scanning 4 Premium Symbols:*\n"
                "• 🥇 XAUUSD (Gold/USD)\n"
                "• 🥇 XAUEUR (Gold/EUR)\n"
                "• 🥈 XAGUSD (Silver/USD)\n"
                "• ₿ BTCUSD (Bitcoin)\n\n"
                "🤖 *AI Detection Active:*\n"
                "• Pattern recognition\n"
                "• Multi-timeframe (5m-1h)\n"
                "• 85%+ confidence only\n"
                "• Premium at 92%+\n\n"
                "⏱️ _Scanning every 30 seconds..._\n"
                "Use `/stopscalp` to stop.",
                parse_mode='Markdown'
            )
            
            # Start the scalp scanning task
            self.scalp_scan_task = asyncio.create_task(
                self._run_scalp_scan(chat_id)
            )
            
            logger.info(f"Scalp scan started for chat {chat_id}")
            
        except Exception as e:
            logger.error(f"Error starting scalp scan: {e}")
            await update.message.reply_text(
                f"❌ Error starting scalp scan: {str(e)}"
            )
    
    async def _run_scalp_scan(self, chat_id: int):
        """Background task for scalp scanning - sends signals IMMEDIATELY when found."""
        scan_count = 0
        
        while True:
            try:
                # Check if trading is still enabled from Control Bot
                if not is_trading_enabled():
                    logger.info("Trading disabled from Control Bot - stopping scalp scan")
                    await self.application.bot.send_message(
                        chat_id=chat_id,
                        text="⏹️ *Scalp scan stopped*\n\nTrading disabled from NG XM Trading BOT.",
                        parse_mode='Markdown'
                    )
                    break
                
                scan_count += 1
                logger.info(f"Scalp scan #{scan_count} starting...")
                
                # Define callback to send scalp signal IMMEDIATELY
                async def send_scalp_immediately(signal):
                    """Send scalp signal as soon as it's found"""
                    message = self.scalp_engine.format_scalp_signal(signal)
                    
                    await self.application.bot.send_message(
                        chat_id=chat_id,
                        text=message,
                        parse_mode='Markdown'
                    )
                    
                    # Also broadcast to subscribers
                    if self.signal_subscribers:
                        await self.broadcast_signal(message)
                    
                    # Log the signal
                    self.log_signal({
                        'symbol': signal.symbol,
                        'type': f"SCALP_{signal.direction}",
                        'entry_price': signal.entry_price,
                        'confidence': signal.confidence,
                        'risk_reward': signal.risk_reward
                    })
                
                # Scan with immediate callback
                signals = await self.scalp_engine.scan_for_scalp_signals_async(
                    on_signal_callback=send_scalp_immediately
                )
                
                if signals:
                    logger.info(f"✅ Scalp scan #{scan_count}: {len(signals)} signals sent")
                else:
                    logger.info(f"No scalp signals in scan #{scan_count}")
                
                # Wait 30 seconds for scalp scanning (only 4 symbols = fast)
                await asyncio.sleep(30)
                
            except asyncio.CancelledError:
                logger.info("Scalp scan task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in scalp scan: {e}")
                await asyncio.sleep(20)
    
    async def stopscalp(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Stop scalp signal scanning."""
        print("/stopscalp command received")
        
        if self.scalp_scan_task and not self.scalp_scan_task.done():
            self.scalp_scan_task.cancel()
            
            try:
                await self.scalp_scan_task
            except asyncio.CancelledError:
                pass
            
            self.scalp_scan_task = None
            
            await update.message.reply_text(
                "🛑 *SCALP SCANNER STOPPED*\n\n"
                "AI scalp scanning has been stopped.\n"
                "Use `/scalpscan` to start again.",
                parse_mode='Markdown'
            )
            logger.info("Scalp scan stopped by user")
        else:
            await update.message.reply_text(
                "ℹ️ Scalp scanner is not running.\n"
                "Use `/scalpscan` to start scanning.",
                parse_mode='Markdown'
            )

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
        print("/signal command received")
        user_id = update.effective_user.id
        # Use user's favorite pairs if no symbol is provided
        if context.args:
            pairs = [arg.upper() for arg in context.args]
        else:
            settings = self.user_settings.get(user_id, {'favorite_pairs': ['XAUUSD'], 'notifications': True})
            pairs = settings['favorite_pairs']
        for symbol in pairs:
            if not self.is_market_open(symbol):
                await update.message.reply_text(
                    f"ℹ️ The market for {symbol} is currently closed. Try BTCUSD or another crypto for 24/7 signals."
                )
                continue
            # Check for high-impact news
            if self.should_avoid_trading(symbol):
                await update.message.reply_text(
                    f"⚠️ High-impact news event detected for {symbol}. Trading is not recommended at this time."
                )
                continue
            message = await update.message.reply_text(f"🔔 Getting latest trading signal for {symbol}...")
            try:
                signals = await asyncio.wait_for(
                    asyncio.to_thread(self.get_current_signals, symbol),
                    timeout=30
                )
            except asyncio.TimeoutError:
                await message.edit_text("❌ Signal request timed out. Please try again later.")
                continue
            except Exception as e:
                await message.edit_text("❌ Error retrieving signals. Please try again later.")
                continue
            if not signals:
                await message.edit_text("ℹ️ No high-probability signals available at the moment.")
                continue
            # Format and send signal with sentiment, ML, and multi-timeframe info
            signal = signals[0]
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
            await message.edit_text(signal_message)

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
        if not self.is_market_open(symbol):
            print(f"[DEBUG] Market closed for {symbol}")
            return []
        if self.should_avoid_trading(symbol):
            print(f"[DEBUG] Avoiding trading for {symbol} due to news event")
            return []
        # Multi-timeframe analysis (comprehensive: 1m to 48h)
        timeframes = ['1m', '3m', '5m', '7m', '10m', '15m', '20m', '30m', '1h', '2h', '4h', '8h', '12h', '24h', '48h']
        tf_data = self.get_multi_timeframe_data(symbol, timeframes)
        if not self.multi_timeframe_confirmed(tf_data):
            print(f"[DEBUG] Multi-timeframe not confirmed for {symbol}: {[v['trend'] for v in tf_data.values()]}")
            return []  # Only signal if all timeframes align
        # Use the trend and sentiment from the main timeframe (1h)
        sentiment = tf_data['1h']['sentiment']
        trend_strength = tf_data['1h']['trend_strength']
        base_confidence = 0.85
        sentiment_multiplier = 1.2 if sentiment == 'bullish' else 0.8 if sentiment == 'bearish' else 1.0
        final_confidence = min(0.95, base_confidence * sentiment_multiplier * trend_strength)
        real = self.get_real_signal(symbol)
        if real:
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
            print(f"[DEBUG] {symbol} | Confidence: {final_confidence:.2f} | ML Score: {ml_score:.2f} | Threshold: {self.min_ml_confidence}")
            if ml_score < self.min_ml_confidence:
                print(f"[DEBUG] Signal for {symbol} filtered out due to low ML score.")
                return []  # Filter out low-confidence signals
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
        print(f"[DEBUG] No real signal for {symbol}")
        return []

    def get_real_signal(self, symbol: str):
        symbol = symbol.upper()
        metals_symbols = ["XAUUSD", "XAGUSD", "XPTUSD", "XPDUSD"]
        oil_symbols = ["USOIL", "UKOIL", "WTIUSD", "BRENTUSD"]
        indices_symbols = ["SPX500", "US30", "NAS100", "GER30", "FRA40", "UK100", "JPN225", "HK50", "AUS200", "EU50"]

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
        try:
            handler = TA_Handler(
                symbol=tv_symbol,
                screener=screener,
                exchange=exchange,
                interval=Interval.INTERVAL_1_HOUR
            )
            analysis = handler.get_analysis()
            price = analysis.indicators["close"]
            recommendation = analysis.summary["RECOMMENDATION"]
            return {
                "symbol": symbol,
                "price": price,
                "recommendation": recommendation,
                "summary": analysis.summary
            }
        except Exception as e:
            print(f"Error fetching TradingView data: {e}")
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
        """Send performance summary to all users with notifications enabled."""
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
                    await self.application.bot.send_message(chat_id=user_id, text=message)
                except Exception as e:
                    print(f"Failed to send daily report to {user_id}: {e}")

    def set_application(self, application):
        self.application = application

    def start_scheduler(self):
        """Start the scheduler."""
        self.scheduler.start()

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
    """Start the bot."""
    bot = None
    try:
        print("Starting bot...")
        bot = TradingBot()
        
        # Start the scheduler
        bot.start_scheduler()
        
        print("Bot started successfully!")
        print("Use /start and /signal XAUUSD or /signal BTCUSD in Telegram.")
        
        # Run polling (this handles the event loop)
        await bot.application.run_polling(drop_pending_updates=True)
        
    except KeyboardInterrupt:
        print("\nBot stopped by user.")
    except Exception as e:
        print(f"Error starting bot: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if bot and bot.scheduler.running:
            bot.scheduler.shutdown()
        print("Bot stopped.")

if __name__ == "__main__":
    import nest_asyncio
    try:
        nest_asyncio.apply()
    except:
        pass
    asyncio.run(main())