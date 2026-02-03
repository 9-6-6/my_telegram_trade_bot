"""
🚀 COMBINED TRADING SYSTEM
===========================
Runs BOTH bots in ONE terminal:
- NG XM Trading BOT (Control) - Token: 7849517577
- NGBOT (Signals) - Token: 8548359658

When you press START in Control Bot → Signals flow to NGBOT automatically!
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime
from typing import Dict

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, Bot
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

# ============================================================
# CONFIGURATION
# ============================================================

# NG XM Trading BOT - Control (Start/Stop buttons)
CONTROL_BOT_TOKEN = "7849517577:AAGx8PhFyAf-cEFt06pfL_CPT8x9REVB1_U"

# NGBOT - Signals
SIGNAL_BOT_TOKEN = "8548359658:AAE420kDIrgpyExD8gJwi9b4kZfNsJ1nJYA"

# Your Chat ID
ADMIN_CHAT_ID = 603932135

# XM360 Account Info
XM_ACCOUNT = "315982803"
XM_SERVER = "XMGlobal-MT5 7"

# State file
STATE_FILE = "trading_state.json"

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================
# STATE MANAGEMENT
# ============================================================

def load_state() -> Dict:
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
    except:
        pass
    return {
        "is_trading_enabled": False,
        "auto_scan_enabled": False,
        "scalp_scan_enabled": False,
        "balance": 10000.0,
        "start_time": None,
        "trailing_enabled": True,
        "max_risk_percent": 10.0
    }

def save_state(state: Dict):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2, default=str)

# ============================================================
# SIGNAL BOT (NGBOT) - Import from copy_trading_bot
# ============================================================

# We'll import the TradingBot class
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from copy_trading_bot import TradingBot
    TRADING_BOT_AVAILABLE = True
    print("[INFO] ✅ copy_trading_bot.py loaded successfully")
except ImportError as e:
    TRADING_BOT_AVAILABLE = False
    print(f"[WARNING] ❌ Could not import TradingBot: {e}")

# Global reference to signal bot
signal_bot = None
signal_bot_task = None

# ============================================================
# CONTROL BOT KEYBOARDS
# ============================================================

def get_main_keyboard(state: Dict):
    is_enabled = state.get("is_trading_enabled", False)
    auto_scan = state.get("auto_scan_enabled", False)
    scalp_scan = state.get("scalp_scan_enabled", False)
    
    if is_enabled:
        trading_btn = InlineKeyboardButton("🔴 STOP TRADING", callback_data="stop_trading")
    else:
        trading_btn = InlineKeyboardButton("🟢 START TRADING", callback_data="start_trading")
    
    if auto_scan:
        auto_btn = InlineKeyboardButton("⏹️ Stop Auto", callback_data="stop_autoscan")
    else:
        auto_btn = InlineKeyboardButton("🔄 Auto Scan", callback_data="start_autoscan")
    
    if scalp_scan:
        scalp_btn = InlineKeyboardButton("⏹️ Stop Scalp", callback_data="stop_scalpscan")
    else:
        scalp_btn = InlineKeyboardButton("⚡ Scalp Scan", callback_data="start_scalpscan")
    
    keyboard = [
        [trading_btn],
        [auto_btn, scalp_btn],
        [
            InlineKeyboardButton("📊 Status", callback_data="status"),
            InlineKeyboardButton("💰 Balance", callback_data="balance")
        ],
        [
            InlineKeyboardButton("⚙️ Settings", callback_data="settings"),
            InlineKeyboardButton("❓ Help", callback_data="help")
        ],
        [InlineKeyboardButton("🔄 Refresh", callback_data="refresh")]
    ]
    return InlineKeyboardMarkup(keyboard)

def get_settings_keyboard(state: Dict):
    trailing = "✅" if state.get("trailing_enabled", True) else "❌"
    risk = state.get("max_risk_percent", 10)
    
    keyboard = [
        [InlineKeyboardButton(f"📈 Trailing: {trailing}", callback_data="toggle_trailing")],
        [InlineKeyboardButton(f"⚠️ Risk: {risk}%", callback_data="show_risk")],
        [InlineKeyboardButton("◀️ Back", callback_data="menu")]
    ]
    return InlineKeyboardMarkup(keyboard)

# ============================================================
# CONTROL BOT HANDLERS
# ============================================================

async def control_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start in Control Bot"""
    state = load_state()
    
    status = "🟢 ENABLED" if state.get("is_trading_enabled") else "🔴 DISABLED"
    auto = "🔄 ON" if state.get("auto_scan_enabled") else "OFF"
    scalp = "⚡ ON" if state.get("scalp_scan_enabled") else "OFF"
    
    text = f"""
🎮 *NG XM Trading BOT*

━━━━━━━━━━━━━━━━━━━━━━
📊 *Trading:* {status}
🔄 *Auto Scan:* {auto}
⚡ *Scalp Scan:* {scalp}
━━━━━━━━━━━━━━━━━━━━━━

*Tap START to begin trading!*
Signals will go to NGBOT automatically.

🏦 {XM_ACCOUNT} | {XM_SERVER}
    """
    
    await update.message.reply_text(
        text, parse_mode='Markdown',
        reply_markup=get_main_keyboard(state)
    )

async def control_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle button presses in Control Bot"""
    global signal_bot
    
    query = update.callback_query
    await query.answer()
    
    action = query.data
    state = load_state()
    
    if action == "start_trading":
        state["is_trading_enabled"] = True
        state["start_time"] = datetime.now().isoformat()
        save_state(state)
        
        # Send notification to NGBOT
        if signal_bot:
            try:
                await signal_bot.application.bot.send_message(
                    chat_id=ADMIN_CHAT_ID,
                    text="🟢 *TRADING ENABLED!*\n\n"
                         "Use `/autoscan` or `/scalpscan` to start receiving signals.\n"
                         "Or use `/newsignal XAUUSD` for a single signal.",
                    parse_mode='Markdown'
                )
            except Exception as e:
                logger.error(f"Error sending to NGBOT: {e}")
        
        text = """
🟢 *TRADING ENABLED!*

✅ Trading is now ACTIVE
📡 NGBOT ready for signals

*Next:* Tap Auto Scan or Scalp Scan
Or go to NGBOT and use commands.
        """
        await query.edit_message_text(text, parse_mode='Markdown', reply_markup=get_main_keyboard(state))
        logger.info("✅ Trading ENABLED")
        
    elif action == "stop_trading":
        state["is_trading_enabled"] = False
        state["auto_scan_enabled"] = False
        state["scalp_scan_enabled"] = False
        state["start_time"] = None
        save_state(state)
        
        # Notify NGBOT
        if signal_bot:
            try:
                # Stop any running scans
                if signal_bot.auto_scan_task and not signal_bot.auto_scan_task.done():
                    signal_bot.auto_scan_task.cancel()
                if signal_bot.scalp_scan_task and not signal_bot.scalp_scan_task.done():
                    signal_bot.scalp_scan_task.cancel()
                    
                await signal_bot.application.bot.send_message(
                    chat_id=ADMIN_CHAT_ID,
                    text="🔴 *TRADING STOPPED!*\n\nAll scanning stopped.",
                    parse_mode='Markdown'
                )
            except Exception as e:
                logger.error(f"Error: {e}")
        
        text = "🔴 *TRADING STOPPED!*\n\nNo new signals. Tap START to resume."
        await query.edit_message_text(text, parse_mode='Markdown', reply_markup=get_main_keyboard(state))
        logger.info("🔴 Trading STOPPED")
        
    elif action == "start_autoscan":
        if not state.get("is_trading_enabled"):
            await query.answer("⚠️ Enable trading first!", show_alert=True)
            return
            
        state["auto_scan_enabled"] = True
        save_state(state)
        
        # Start auto scan in NGBOT
        if signal_bot and hasattr(signal_bot, 'auto_generator') and signal_bot.auto_generator:
            try:
                # Initialize if needed
                await signal_bot.auto_generator.initialize()
                
                # Start scanning task
                if signal_bot.auto_scan_task is None or signal_bot.auto_scan_task.done():
                    signal_bot.auto_scan_task = asyncio.create_task(
                        run_auto_scan(signal_bot, ADMIN_CHAT_ID)
                    )
                
                await signal_bot.application.bot.send_message(
                    chat_id=ADMIN_CHAT_ID,
                    text="🔄 *AUTO SCAN STARTED!*\n\n"
                         "🔍 Scanning 30+ markets every 60s\n"
                         "📊 Signals will appear here automatically!\n\n"
                         "_Use Control Bot to stop_",
                    parse_mode='Markdown'
                )
            except Exception as e:
                logger.error(f"Error starting auto scan: {e}")
        
        text = "🔄 *AUTO SCAN ON!*\n\nSignals flowing to NGBOT!"
        await query.edit_message_text(text, parse_mode='Markdown', reply_markup=get_main_keyboard(state))
        logger.info("🔄 Auto Scan STARTED")
        
    elif action == "stop_autoscan":
        state["auto_scan_enabled"] = False
        save_state(state)
        
        if signal_bot and signal_bot.auto_scan_task:
            signal_bot.auto_scan_task.cancel()
            
        await query.edit_message_text("⏹️ Auto Scan OFF", parse_mode='Markdown', reply_markup=get_main_keyboard(state))
        
    elif action == "start_scalpscan":
        if not state.get("is_trading_enabled"):
            await query.answer("⚠️ Enable trading first!", show_alert=True)
            return
            
        state["scalp_scan_enabled"] = True
        save_state(state)
        
        # Start scalp scan in NGBOT
        if signal_bot and hasattr(signal_bot, 'scalp_engine') and signal_bot.scalp_engine:
            try:
                if signal_bot.scalp_scan_task is None or signal_bot.scalp_scan_task.done():
                    signal_bot.scalp_scan_task = asyncio.create_task(
                        run_scalp_scan(signal_bot, ADMIN_CHAT_ID)
                    )
                
                await signal_bot.application.bot.send_message(
                    chat_id=ADMIN_CHAT_ID,
                    text="⚡ *SCALP SCAN STARTED!*\n\n"
                         "🎯 Scanning XAUUSD, BTCUSD, XAGUSD, XAUEUR\n"
                         "📊 85%+ confidence signals only\n\n"
                         "_Signals will appear here!_",
                    parse_mode='Markdown'
                )
            except Exception as e:
                logger.error(f"Error starting scalp scan: {e}")
        
        text = "⚡ *SCALP SCAN ON!*\n\nSignals flowing to NGBOT!"
        await query.edit_message_text(text, parse_mode='Markdown', reply_markup=get_main_keyboard(state))
        logger.info("⚡ Scalp Scan STARTED")
        
    elif action == "stop_scalpscan":
        state["scalp_scan_enabled"] = False
        save_state(state)
        
        if signal_bot and signal_bot.scalp_scan_task:
            signal_bot.scalp_scan_task.cancel()
            
        await query.edit_message_text("⏹️ Scalp Scan OFF", parse_mode='Markdown', reply_markup=get_main_keyboard(state))
        
    elif action == "status":
        uptime = "N/A"
        if state.get("start_time") and state.get("is_trading_enabled"):
            try:
                start = datetime.fromisoformat(state["start_time"])
                delta = datetime.now() - start
                hours = delta.seconds // 3600
                mins = (delta.seconds % 3600) // 60
                uptime = f"{hours}h {mins}m"
            except:
                pass
        
        trading = "🟢 ON" if state.get("is_trading_enabled") else "🔴 OFF"
        auto = "🔄 ON" if state.get("auto_scan_enabled") else "OFF"
        scalp = "⚡ ON" if state.get("scalp_scan_enabled") else "OFF"
        
        text = f"""
📊 *Status*

Trading: {trading}
Uptime: {uptime}
Auto Scan: {auto}
Scalp Scan: {scalp}
Trailing: {'✅' if state.get('trailing_enabled') else '❌'}
Risk: {state.get('max_risk_percent', 10)}%
        """
        await query.edit_message_text(text, parse_mode='Markdown', reply_markup=get_main_keyboard(state))
        
    elif action == "balance":
        balance = state.get("balance", 10000)
        risk = state.get("max_risk_percent", 10)
        
        text = f"""
💰 *Balance*

💵 ${balance:,.2f}

{risk}% Risk:
✅ Trade: ${balance * risk / 100:,.2f}
🛡️ Safe: ${balance * (100-risk) / 100:,.2f}

🏦 {XM_ACCOUNT}
        """
        await query.edit_message_text(text, parse_mode='Markdown', reply_markup=get_main_keyboard(state))
        
    elif action == "settings":
        await query.edit_message_text("⚙️ *Settings*", parse_mode='Markdown', reply_markup=get_settings_keyboard(state))
        
    elif action == "toggle_trailing":
        state["trailing_enabled"] = not state.get("trailing_enabled", True)
        save_state(state)
        status = "✅ ON" if state["trailing_enabled"] else "❌ OFF"
        await query.edit_message_text(f"📈 Trailing: {status}", parse_mode='Markdown', reply_markup=get_settings_keyboard(state))
        
    elif action == "show_risk":
        await query.edit_message_text(f"⚠️ Risk: {state.get('max_risk_percent', 10)}%", parse_mode='Markdown', reply_markup=get_settings_keyboard(state))
        
    elif action == "help":
        text = """
❓ *Help*

🟢 START → Enables trading
🔄 Auto Scan → Swing signals
⚡ Scalp Scan → Quick scalp signals

*All signals go to NGBOT!*

Control Bot = START/STOP
NGBOT = Receive signals
        """
        await query.edit_message_text(text, parse_mode='Markdown', reply_markup=get_main_keyboard(state))
        
    elif action in ["menu", "refresh"]:
        state = load_state()
        status = "🟢 ON" if state.get("is_trading_enabled") else "🔴 OFF"
        
        text = f"""
🎮 *NG XM Trading BOT*

Trading: {status}
Auto: {'🔄' if state.get('auto_scan_enabled') else '⏹️'}
Scalp: {'⚡' if state.get('scalp_scan_enabled') else '⏹️'}
        """
        await query.edit_message_text(text, parse_mode='Markdown', reply_markup=get_main_keyboard(state))

# ============================================================
# AUTO SCAN LOOP
# ============================================================

async def run_auto_scan(bot, chat_id: int):
    """Background auto scan task"""
    scan_count = 0
    
    while True:
        try:
            state = load_state()
            if not state.get("is_trading_enabled") or not state.get("auto_scan_enabled"):
                logger.info("Auto scan stopped - disabled from Control Bot")
                break
            
            scan_count += 1
            logger.info(f"Auto scan #{scan_count}...")
            
            signals = await bot.auto_generator.scan_all_markets()
            
            if signals:
                for signal in signals:
                    message = bot.auto_generator.format_signal_message(signal)
                    await bot.application.bot.send_message(
                        chat_id=chat_id,
                        text=message,
                        parse_mode='Markdown'
                    )
                    await asyncio.sleep(2)
                    
            await asyncio.sleep(60)
            
        except asyncio.CancelledError:
            logger.info("Auto scan cancelled")
            break
        except Exception as e:
            logger.error(f"Auto scan error: {e}")
            await asyncio.sleep(30)

async def run_scalp_scan(bot, chat_id: int):
    """Background scalp scan task"""
    scan_count = 0
    
    while True:
        try:
            state = load_state()
            if not state.get("is_trading_enabled") or not state.get("scalp_scan_enabled"):
                logger.info("Scalp scan stopped - disabled from Control Bot")
                break
            
            scan_count += 1
            logger.info(f"Scalp scan #{scan_count}...")
            
            signals = await bot.scalp_engine.scan_for_scalp_signals_async()
            
            if signals:
                for signal in signals:
                    message = bot.scalp_engine.format_scalp_signal(signal)
                    await bot.application.bot.send_message(
                        chat_id=chat_id,
                        text=message,
                        parse_mode='Markdown'
                    )
                    await asyncio.sleep(2)
                    
            await asyncio.sleep(30)
            
        except asyncio.CancelledError:
            logger.info("Scalp scan cancelled")
            break
        except Exception as e:
            logger.error(f"Scalp scan error: {e}")
            await asyncio.sleep(20)

# ============================================================
# MAIN - RUN BOTH BOTS
# ============================================================

async def main():
    global signal_bot
    
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║       🚀 COMBINED TRADING SYSTEM                              ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║                                                               ║
    ║   Running BOTH bots in ONE terminal!                          ║
    ║                                                               ║
    ║   🎮 NG XM Trading BOT - Control (Start/Stop)                 ║
    ║   📡 NGBOT - Signals (copy_trading_bot.py)                    ║
    ║                                                               ║
    ║   Press START in Control Bot → Signals flow to NGBOT!         ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Initialize state
    state = load_state()
    state["is_trading_enabled"] = False
    state["auto_scan_enabled"] = False
    state["scalp_scan_enabled"] = False
    save_state(state)
    
    # Create Control Bot
    control_app = Application.builder().token(CONTROL_BOT_TOKEN).build()
    control_app.add_handler(CommandHandler("start", control_start))
    control_app.add_handler(CallbackQueryHandler(control_callback))
    
    # Create Signal Bot (NGBOT)
    if TRADING_BOT_AVAILABLE:
        try:
            signal_bot = TradingBot()
            print("[INFO] ✅ Signal Bot (NGBOT) initialized")
        except Exception as e:
            print(f"[ERROR] Failed to initialize Signal Bot: {e}")
            signal_bot = None
    
    print("\n✅ Both bots starting...")
    print("   🎮 NG XM Trading BOT - Send /start")
    print("   📡 NGBOT - Receives signals automatically")
    print("\n   Press Ctrl+C to stop both bots\n")
    
    # Run both bots concurrently
    try:
        # Initialize both
        await control_app.initialize()
        if signal_bot:
            await signal_bot.application.initialize()
        
        # Start both
        await control_app.start()
        if signal_bot:
            await signal_bot.application.start()
            # Start scheduler if exists
            if hasattr(signal_bot, 'scheduler') and signal_bot.scheduler:
                signal_bot.scheduler.start()
        
        # Start polling for both
        await control_app.updater.start_polling(drop_pending_updates=True)
        if signal_bot:
            await signal_bot.application.updater.start_polling(drop_pending_updates=True)
        
        print("✅ Both bots are now running!")
        print("   Go to Telegram and test!")
        
        # Keep running
        while True:
            await asyncio.sleep(1)
            
    except asyncio.CancelledError:
        pass
    finally:
        # Cleanup
        print("\n🛑 Shutting down...")
        await control_app.updater.stop()
        await control_app.stop()
        await control_app.shutdown()
        
        if signal_bot:
            await signal_bot.application.updater.stop()
            await signal_bot.application.stop()
            await signal_bot.application.shutdown()
        
        print("👋 Goodbye!")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Stopped by user")
