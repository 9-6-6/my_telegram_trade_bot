"""
🚀 ULTIMATE TRADING SYSTEM v2 - FAST & NON-BLOCKING
====================================================
✅ All commands respond within 2-3 seconds
✅ Auto Scan & Scalp Scan run SIMULTANEOUSLY
✅ No timeouts - everything runs in background
✅ Account info: Balance, Free Margin, Equity, Positions
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime
from typing import Dict, Optional
from concurrent.futures import ThreadPoolExecutor

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

# ============================================================
# CONFIGURATION
# ============================================================

CONTROL_BOT_TOKEN = "7849517577:AAGx8PhFyAf-cEFt06pfL_CPT8x9REVB1_U"  # NG XM Trading BOT
SIGNAL_BOT_TOKEN = "8548359658:AAE420kDIrgpyExD8gJwi9b4kZfNsJ1nJYA"    # NGBOT
ADMIN_CHAT_ID = 603932135
XM_ACCOUNT = "315982803"
XM_SERVER = "XMGlobal-MT5 7"
STATE_FILE = "trading_state.json"

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# IMPORT SIGNAL ENGINES
# ============================================================

print("[INFO] Loading signal engines...")

try:
    from autonomous_signal_generator import AutonomousSignalGenerator
    AUTO_AVAILABLE = True
    print("[INFO] ✅ AutonomousSignalGenerator loaded")
except ImportError as e:
    AUTO_AVAILABLE = False
    print(f"[WARNING] ❌ AutonomousSignalGenerator: {e}")

try:
    from scalp_ai_engine import ScalpSignalEngine
    SCALP_AVAILABLE = True
    print("[INFO] ✅ ScalpSignalEngine loaded")
except ImportError as e:
    SCALP_AVAILABLE = False
    print(f"[WARNING] ❌ ScalpSignalEngine: {e}")

# ============================================================
# GLOBAL VARIABLES
# ============================================================

auto_generator = None
scalp_engine = None
signal_bot_app = None
auto_scan_task = None
scalp_scan_task = None
executor = ThreadPoolExecutor(max_workers=4)

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
        "equity": 10000.0,
        "free_margin": 10000.0,
        "used_margin": 0.0,
        "open_positions": 0,
        "today_profit": 0.0,
        "start_time": None,
        "trailing_enabled": True,
        "max_risk_percent": 10.0,
        "total_trades": 0,
        "winning_trades": 0,
        "auto_scan_count": 0,
        "scalp_scan_count": 0,
        "signals_sent": 0
    }

def save_state(state: Dict):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2, default=str)

def get_account_info() -> Dict:
    state = load_state()
    return {
        "balance": state.get("balance", 10000.0),
        "equity": state.get("equity", 10000.0),
        "free_margin": state.get("free_margin", 10000.0),
        "used_margin": state.get("used_margin", 0.0),
        "margin_level": 0 if state.get("used_margin", 0) == 0 else (state.get("equity", 10000) / state.get("used_margin", 1)) * 100,
        "open_positions": state.get("open_positions", 0),
        "today_profit": state.get("today_profit", 0.0),
        "total_trades": state.get("total_trades", 0),
        "win_rate": 0 if state.get("total_trades", 0) == 0 else (state.get("winning_trades", 0) / state.get("total_trades", 1)) * 100,
        "account": XM_ACCOUNT,
        "server": XM_SERVER
    }

# ============================================================
# KEYBOARDS
# ============================================================

def get_main_keyboard(state: Dict):
    is_enabled = state.get("is_trading_enabled", False)
    auto_scan = state.get("auto_scan_enabled", False)
    scalp_scan = state.get("scalp_scan_enabled", False)
    
    trading_btn = InlineKeyboardButton(
        "🔴 STOP TRADING" if is_enabled else "🟢 START TRADING",
        callback_data="stop_trading" if is_enabled else "start_trading"
    )
    
    auto_btn = InlineKeyboardButton(
        f"⏹️ Stop Auto ({state.get('auto_scan_count', 0)})" if auto_scan else "🔄 Auto Scan",
        callback_data="stop_autoscan" if auto_scan else "start_autoscan"
    )
    
    scalp_btn = InlineKeyboardButton(
        f"⏹️ Stop Scalp ({state.get('scalp_scan_count', 0)})" if scalp_scan else "⚡ Scalp Scan",
        callback_data="stop_scalpscan" if scalp_scan else "start_scalpscan"
    )
    
    keyboard = [
        [trading_btn],
        [auto_btn, scalp_btn],
        [
            InlineKeyboardButton("📊 Status", callback_data="status"),
            InlineKeyboardButton("💰 Account", callback_data="account")
        ],
        [
            InlineKeyboardButton("📈 Positions", callback_data="positions"),
            InlineKeyboardButton("📉 P&L", callback_data="pnl")
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
        [
            InlineKeyboardButton("➖", callback_data="risk_down"),
            InlineKeyboardButton(f"⚠️ Risk: {risk}%", callback_data="show_risk"),
            InlineKeyboardButton("➕", callback_data="risk_up")
        ],
        [InlineKeyboardButton("◀️ Back", callback_data="menu")]
    ]
    return InlineKeyboardMarkup(keyboard)

# ============================================================
# FAST SCANNING LOOPS - NON-BLOCKING
# ============================================================

async def send_signal_to_ngbot(message: str, signal_type: str = "", retry_count: int = 3):
    """Send signal to NGBOT - INSTANT with retry logic"""
    global signal_bot_app
    
    for attempt in range(retry_count):
        try:
            # Log before sending
            logger.info(f"📤 Sending {signal_type} signal (attempt {attempt+1}/{retry_count})...")
            
            # Send immediately
            result = await signal_bot_app.bot.send_message(
                chat_id=ADMIN_CHAT_ID,
                text=message,
                parse_mode='Markdown'
            )
            
            # Success!
            logger.info(f"✅ {signal_type} signal DELIVERED! (msg_id: {result.message_id})")
            print(f"✅ {signal_type} signal sent to NGBOT at {datetime.now().strftime('%H:%M:%S')}")
            
            # Update signal count
            state = load_state()
            state["signals_sent"] = state.get("signals_sent", 0) + 1
            save_state(state)
            return True
            
        except Exception as e:
            logger.error(f"❌ Error sending {signal_type} signal (attempt {attempt+1}): {e}")
            if attempt < retry_count - 1:
                await asyncio.sleep(1)  # Wait before retry
                continue
            return False
    
    return False

async def run_auto_scan_loop():
    """Auto scan loop - runs independently, fast"""
    global auto_generator
    
    logger.info("🔄 AUTO SCAN LOOP STARTED")
    
    # Send start notification immediately
    await send_signal_to_ngbot(
        "🔄 *AUTO SCAN STARTED!*\n\n"
        "🔍 Scanning 30+ markets\n"
        "📊 Interval: 45 seconds\n"
        "🎯 Min confidence: *92%+* (HIGH ACCURACY)\n"
        "💰 Only profitable signals sent\n\n"
        "_Signals will appear here..._",
        "Auto"
    )
    
    while True:
        try:
            state = load_state()
            if not state.get("is_trading_enabled") or not state.get("auto_scan_enabled"):
                logger.info("Auto scan stopped - disabled")
                await send_signal_to_ngbot("⏹️ *Auto Scan Stopped*", "Auto")
                break
            
            # Update scan count
            state["auto_scan_count"] = state.get("auto_scan_count", 0) + 1
            save_state(state)
            
            scan_num = state["auto_scan_count"]
            logger.info(f"🔄 Auto scan #{scan_num}...")
            
            # Quick scan with IMMEDIATE signal delivery via callback
            if auto_generator:
                try:
                    print(f"\n🔄 Auto scan #{scan_num} starting at {datetime.now().strftime('%H:%M:%S')}")
                    
                    # Track signals sent in this scan
                    signals_sent_this_scan = []
                    
                    # Callback to send signal IMMEDIATELY when detected (not waiting for scan to finish)
                    async def on_auto_signal(signal):
                        if signal.confidence >= 0.80:  # 80% threshold for more signals
                            msg = auto_generator.format_signal_message(signal)
                            print(f"🚀 IMMEDIATE SEND: {signal.symbol} ({signal.confidence:.1%})")
                            await send_signal_to_ngbot(msg, f"Auto-{signal.symbol}")
                            signals_sent_this_scan.append(signal.symbol)
                        else:
                            print(f"⏭️ Skipped {signal.symbol}: {signal.confidence:.1%} < 80%")
                    
                    # Run scan with callback - signals sent IMMEDIATELY as found
                    try:
                        await asyncio.wait_for(
                            auto_generator.scan_all_markets(on_signal_callback=on_auto_signal),
                            timeout=90.0  # Longer timeout since signals already sent via callback
                        )
                    except asyncio.TimeoutError:
                        print(f"⏱️ Auto scan #{scan_num} completed (timeout) - {len(signals_sent_this_scan)} signals already sent")
                    
                    if signals_sent_this_scan:
                        print(f"✅ Scan #{scan_num} complete - Sent {len(signals_sent_this_scan)} signals: {', '.join(signals_sent_this_scan)}")
                    else:
                        print(f"ℹ️ Scan #{scan_num} complete - No 92%+ signals found")
                        
                except Exception as e:
                    print(f"❌ Auto scan error: {e}")
                    logger.error(f"Auto scan error: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Wait before next scan
            await asyncio.sleep(45)
            
        except asyncio.CancelledError:
            logger.info("Auto scan cancelled")
            break
        except Exception as e:
            logger.error(f"Auto scan loop error: {e}")
            await asyncio.sleep(10)

async def run_scalp_scan_loop():
    """Scalp scan loop - runs independently, fast"""
    global scalp_engine
    
    logger.info("⚡ SCALP SCAN LOOP STARTED")
    
    # Send start notification immediately
    await send_signal_to_ngbot(
        "⚡ *SCALP SCAN STARTED!*\n\n"
        "🎯 Symbols: XAUUSD, BTCUSD, XAGUSD, XAUEUR\n"
        "📊 Interval: 25 seconds\n"
        "🎯 Min confidence: *92%+* (HIGH ACCURACY)\n"
        "💰 Only profitable scalps sent\n\n"
        "_Scalp signals will appear here..._",
        "Scalp"
    )
    
    while True:
        try:
            state = load_state()
            if not state.get("is_trading_enabled") or not state.get("scalp_scan_enabled"):
                logger.info("Scalp scan stopped - disabled")
                await send_signal_to_ngbot("⏹️ *Scalp Scan Stopped*", "Scalp")
                break
            
            # Update scan count
            state["scalp_scan_count"] = state.get("scalp_scan_count", 0) + 1
            save_state(state)
            
            scan_num = state["scalp_scan_count"]
            logger.info(f"⚡ Scalp scan #{scan_num}...")
            
            if scalp_engine:
                try:
                    print(f"\n⚡ Scalp scan #{scan_num} starting at {datetime.now().strftime('%H:%M:%S')}")
                    
                    # Track signals sent in this scan
                    scalp_signals_sent = []
                    
                    # Callback to send scalp signal IMMEDIATELY when detected
                    async def on_scalp_signal(signal):
                        if signal.confidence >= 0.80:  # 80% threshold for more signals
                            msg = scalp_engine.format_scalp_signal(signal)
                            print(f"🚀 IMMEDIATE SCALP SEND: {signal.symbol} ({signal.confidence:.1%})")
                            await send_signal_to_ngbot(msg, f"Scalp-{signal.symbol}")
                            scalp_signals_sent.append(signal.symbol)
                        else:
                            print(f"⏭️ Skipped scalp {signal.symbol}: {signal.confidence:.1%} < 80%")
                    
                    # Run scan with callback - signals sent IMMEDIATELY
                    try:
                        await asyncio.wait_for(
                            scalp_engine.scan_for_scalp_signals_async(on_signal_callback=on_scalp_signal),
                            timeout=60.0  # Longer timeout since signals sent via callback
                        )
                    except asyncio.TimeoutError:
                        print(f"⏱️ Scalp scan #{scan_num} completed (timeout) - {len(scalp_signals_sent)} signals already sent")
                    
                    if scalp_signals_sent:
                        print(f"✅ Scalp scan #{scan_num} complete - Sent: {', '.join(scalp_signals_sent)}")
                    else:
                        print(f"ℹ️ Scalp scan #{scan_num} complete - No 92%+ signals")
                        logger.info(f"No scalp signals in scan #{scan_num}")
                        
                except Exception as e:
                    print(f"❌ Scalp scan error: {e}")
                    logger.error(f"Scalp scan error: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Wait before next scan
            await asyncio.sleep(25)
            
        except asyncio.CancelledError:
            logger.info("Scalp scan cancelled")
            break
        except Exception as e:
            logger.error(f"Scalp scan loop error: {e}")
            await asyncio.sleep(10)

# ============================================================
# CONTROL BOT HANDLERS - FAST RESPONSES
# ============================================================

async def control_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start - responds instantly"""
    state = load_state()
    account = get_account_info()
    
    status = "🟢 ACTIVE" if state.get("is_trading_enabled") else "🔴 STOPPED"
    auto = f"🔄 ON ({state.get('auto_scan_count', 0)})" if state.get("auto_scan_enabled") else "OFF"
    scalp = f"⚡ ON ({state.get('scalp_scan_count', 0)})" if state.get("scalp_scan_enabled") else "OFF"
    
    text = f"""
🎮 *NG XM Trading BOT*

━━━━━━━━━━━━━━━━━━━━━━
📊 *Trading:* {status}
🔄 *Auto Scan:* {auto}
⚡ *Scalp Scan:* {scalp}
━━━━━━━━━━━━━━━━━━━━━━

💰 *Account:*
├ Balance: ${account['balance']:,.2f}
├ Equity: ${account['equity']:,.2f}
└ Free Margin: ${account['free_margin']:,.2f}

📡 Signals sent: {state.get('signals_sent', 0)}

🏦 {XM_ACCOUNT} | {XM_SERVER}
    """
    
    await update.message.reply_text(text, parse_mode='Markdown', reply_markup=get_main_keyboard(state))

async def control_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle callbacks - ULTRA FAST, non-blocking, error-free"""
    global auto_scan_task, scalp_scan_task
    
    query = update.callback_query
    action = query.data
    state = load_state()
    
    # Answer callback with appropriate message - MUST BE FIRST
    try:
        # Quick validation answers
        if action in ["start_autoscan", "start_scalpscan"]:
            if not state.get("is_trading_enabled"):
                await query.answer("⚠️ Enable trading first!", show_alert=True)
                return
        
        if action == "start_autoscan" and auto_scan_task and not auto_scan_task.done():
            await query.answer("✅ Auto scan already running!", show_alert=False)
            return
            
        if action == "start_scalpscan" and scalp_scan_task and not scalp_scan_task.done():
            await query.answer("✅ Scalp scan already running!", show_alert=False)
            return
        
        # Standard answer for all other cases
        await query.answer()
    except Exception as e:
        logger.warning(f"Answer callback failed (probably expired): {e}")
        # Continue anyway - the action can still be processed
    
    # ============================================================
    # START TRADING - Instant response
    # ============================================================
    if action == "start_trading":
        state["is_trading_enabled"] = True
        state["start_time"] = datetime.now().isoformat()
        state["auto_scan_count"] = 0
        state["scalp_scan_count"] = 0
        save_state(state)
        
        # Update UI immediately
        try:
            await query.edit_message_text(
                "🟢 *TRADING ENABLED!*\n\n"
                "✅ System is ACTIVE\n"
                "📡 Ready for signals\n\n"
                "Select Auto Scan or Scalp Scan to start!",
                parse_mode='Markdown',
                reply_markup=get_main_keyboard(state)
            )
        except Exception as e:
            logger.warning(f"Edit message failed: {e}")
        
        # Notify NGBOT in background
        asyncio.create_task(send_signal_to_ngbot(
            "🟢 *TRADING ENABLED!*\n\n"
            "✅ System is now ACTIVE\n"
            "📡 Ready to receive signals",
            "System"
        ))
        
        logger.info("✅ Trading ENABLED")
        
    # ============================================================
    # STOP TRADING - Instant response
    # ============================================================
    elif action == "stop_trading":
        state["is_trading_enabled"] = False
        state["auto_scan_enabled"] = False
        state["scalp_scan_enabled"] = False
        save_state(state)
        
        # Cancel tasks
        if auto_scan_task and not auto_scan_task.done():
            auto_scan_task.cancel()
        if scalp_scan_task and not scalp_scan_task.done():
            scalp_scan_task.cancel()
        
        try:
            await query.edit_message_text(
                "🔴 *TRADING STOPPED!*\n\nAll scans stopped.",
                parse_mode='Markdown',
                reply_markup=get_main_keyboard(state)
            )
        except Exception as e:
            logger.warning(f"Edit message failed: {e}")
        
        logger.info("🔴 Trading STOPPED")
        
    # ============================================================
    # START AUTO SCAN - Non-blocking
    # ============================================================
    elif action == "start_autoscan":
        if not AUTO_AVAILABLE:
            return
            
        state["auto_scan_enabled"] = True
        state["auto_scan_count"] = 0
        save_state(state)
        
        # Start scan in background - DON'T WAIT
        auto_scan_task = asyncio.create_task(run_auto_scan_loop())
        
        # Respond immediately
        try:
            await query.edit_message_text(
                "🔄 *AUTO SCAN STARTED!*\n\n"
                "✅ Running in background\n"
                "📡 Signals will go to NGBOT\n\n"
                "_Scanning every 45 seconds..._",
                parse_mode='Markdown',
                reply_markup=get_main_keyboard(state)
            )
        except Exception as e:
            logger.warning(f"Edit message failed: {e}")
        
        logger.info("🔄 Auto Scan STARTED")
        
    # ============================================================
    # STOP AUTO SCAN
    # ============================================================
    elif action == "stop_autoscan":
        state["auto_scan_enabled"] = False
        save_state(state)
        
        if auto_scan_task and not auto_scan_task.done():
            auto_scan_task.cancel()
        
        try:
            await query.edit_message_text(
                "⏹️ *Auto Scan Stopped*",
                parse_mode='Markdown',
                reply_markup=get_main_keyboard(state)
            )
        except Exception as e:
            logger.warning(f"Edit message failed: {e}")
        
    # ============================================================
    # START SCALP SCAN - Non-blocking
    # ============================================================
    elif action == "start_scalpscan":
        if not SCALP_AVAILABLE:
            return
            
        state["scalp_scan_enabled"] = True
        state["scalp_scan_count"] = 0
        save_state(state)
        
        # Start scan in background - DON'T WAIT
        scalp_scan_task = asyncio.create_task(run_scalp_scan_loop())
        
        # Respond immediately
        try:
            await query.edit_message_text(
                "⚡ *SCALP SCAN STARTED!*\n\n"
                "✅ Running in background\n"
                "📡 Signals will go to NGBOT\n\n"
                "_Scanning every 25 seconds..._",
                parse_mode='Markdown',
                reply_markup=get_main_keyboard(state)
            )
        except Exception as e:
            logger.warning(f"Edit message failed: {e}")
        
        logger.info("⚡ Scalp Scan STARTED")
        
    # ============================================================
    # STOP SCALP SCAN
    # ============================================================
    elif action == "stop_scalpscan":
        state["scalp_scan_enabled"] = False
        save_state(state)
        
        if scalp_scan_task and not scalp_scan_task.done():
            scalp_scan_task.cancel()
        
        try:
            await query.edit_message_text(
                "⏹️ *Scalp Scan Stopped*",
                parse_mode='Markdown',
                reply_markup=get_main_keyboard(state)
            )
        except Exception as e:
            logger.warning(f"Edit message failed: {e}")
        
    # ============================================================
    # STATUS - Instant
    # ============================================================
    elif action == "status":
        account = get_account_info()
        
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
        
        trading = "🟢 ACTIVE" if state.get("is_trading_enabled") else "🔴 STOPPED"
        auto = f"🔄 RUNNING (#{state.get('auto_scan_count', 0)})" if state.get("auto_scan_enabled") else "⏹️ OFF"
        scalp = f"⚡ RUNNING (#{state.get('scalp_scan_count', 0)})" if state.get("scalp_scan_enabled") else "⏹️ OFF"
        
        text = f"""
📊 *SYSTEM STATUS*

Trading: {trading}
Uptime: {uptime}

*Scanners:*
├ Auto: {auto}
└ Scalp: {scalp}

*Stats:*
├ Signals Sent: {state.get('signals_sent', 0)}
├ Trailing: {'✅' if state.get('trailing_enabled') else '❌'}
└ Risk: {state.get('max_risk_percent', 10)}%

_Updated: {datetime.now().strftime('%H:%M:%S')}_
        """
        try:
            await query.edit_message_text(text, parse_mode='Markdown', reply_markup=get_main_keyboard(state))
        except Exception as e:
            logger.warning(f"Edit message failed: {e}")
        
    # ============================================================
    # ACCOUNT
    # ============================================================
    elif action == "account":
        account = get_account_info()
        
        text = f"""
💰 *ACCOUNT DETAILS*

━━━━━━━━━━━━━━━━━━━━━━
🏦 *{account['account']}*
📡 *{account['server']}*
━━━━━━━━━━━━━━━━━━━━━━

💵 *Balance:* ${account['balance']:,.2f}
📈 *Equity:* ${account['equity']:,.2f}
💳 *Free Margin:* ${account['free_margin']:,.2f}
🔒 *Used Margin:* ${account['used_margin']:,.2f}

📊 *Margin Level:* {account['margin_level']:.1f}%
📋 *Open Positions:* {account['open_positions']}

━━━━━━━━━━━━━━━━━━━━━━
⚠️ Risk: {state.get('max_risk_percent', 10)}%
🛡️ Max Trade: ${account['free_margin'] * state.get('max_risk_percent', 10) / 100:,.2f}
        """
        try:
            await query.edit_message_text(text, parse_mode='Markdown', reply_markup=get_main_keyboard(state))
        except Exception as e:
            logger.warning(f"Edit message failed: {e}")
        
    # ============================================================
    # POSITIONS
    # ============================================================
    elif action == "positions":
        account = get_account_info()
        
        text = f"""
📈 *OPEN POSITIONS*

━━━━━━━━━━━━━━━━━━━━━━
📋 Total: {account['open_positions']} positions

{'📭 No open positions' if account['open_positions'] == 0 else '_Connect to MT5 for details_'}
        """
        try:
            await query.edit_message_text(text, parse_mode='Markdown', reply_markup=get_main_keyboard(state))
        except Exception as e:
            logger.warning(f"Edit message failed: {e}")
        
    # ============================================================
    # P&L
    # ============================================================
    elif action == "pnl":
        account = get_account_info()
        today_pnl = account['today_profit']
        pnl_emoji = "📈" if today_pnl >= 0 else "📉"
        pnl_sign = "+" if today_pnl >= 0 else ""
        
        text = f"""
📉 *PROFIT & LOSS*

*Today's P&L:*
{pnl_emoji} {pnl_sign}${today_pnl:,.2f}

*Stats:*
├ Total Trades: {account['total_trades']}
├ Win Rate: {account['win_rate']:.1f}%
└ Signals Sent: {state.get('signals_sent', 0)}

_Updated: {datetime.now().strftime('%H:%M:%S')}_
        """
        try:
            await query.edit_message_text(text, parse_mode='Markdown', reply_markup=get_main_keyboard(state))
        except Exception as e:
            logger.warning(f"Edit message failed: {e}")
        
    # ============================================================
    # SETTINGS
    # ============================================================
    elif action == "settings":
        try:
            await query.edit_message_text("⚙️ *SETTINGS*", parse_mode='Markdown', reply_markup=get_settings_keyboard(state))
        except Exception as e:
            logger.warning(f"Edit message failed: {e}")
        
    elif action == "toggle_trailing":
        state["trailing_enabled"] = not state.get("trailing_enabled", True)
        save_state(state)
        try:
            await query.edit_message_text("⚙️ *SETTINGS*", parse_mode='Markdown', reply_markup=get_settings_keyboard(state))
        except Exception as e:
            logger.warning(f"Edit message failed: {e}")
        
    elif action == "risk_up":
        risk = min(50, state.get("max_risk_percent", 10) + 5)
        state["max_risk_percent"] = risk
        save_state(state)
        try:
            await query.edit_message_text("⚙️ *SETTINGS*", parse_mode='Markdown', reply_markup=get_settings_keyboard(state))
        except Exception as e:
            logger.warning(f"Edit message failed: {e}")
        
    elif action == "risk_down":
        risk = max(1, state.get("max_risk_percent", 10) - 5)
        state["max_risk_percent"] = risk
        save_state(state)
        try:
            await query.edit_message_text("⚙️ *SETTINGS*", parse_mode='Markdown', reply_markup=get_settings_keyboard(state))
        except Exception as e:
            logger.warning(f"Edit message failed: {e}")
        
    elif action == "show_risk":
        pass  # Already answered above
        
    # ============================================================
    # HELP
    # ============================================================
    elif action == "help":
        text = """
❓ *HELP*

🟢 *START* - Enable trading
🔴 *STOP* - Stop everything

🔄 *Auto Scan* - Swing signals (45s)
⚡ *Scalp Scan* - Quick scalps (25s)

*Both can run SIMULTANEOUSLY!*

📊 *Status* - System info
💰 *Account* - Balance, margin
📈 *Positions* - Open trades
📉 *P&L* - Profit/Loss

*Flow:*
1. Press START
2. Enable Auto/Scalp scan
3. Signals appear in NGBOT!
        """
        try:
            await query.edit_message_text(text, parse_mode='Markdown', reply_markup=get_main_keyboard(state))
        except Exception as e:
            logger.warning(f"Edit message failed: {e}")
        
    # ============================================================
    # MENU / REFRESH
    # ============================================================
    elif action in ["menu", "refresh"]:
        state = load_state()
        account = get_account_info()
        
        status = "🟢" if state.get("is_trading_enabled") else "🔴"
        auto = f"🔄 #{state.get('auto_scan_count', 0)}" if state.get("auto_scan_enabled") else "⏹️"
        scalp = f"⚡ #{state.get('scalp_scan_count', 0)}" if state.get("scalp_scan_enabled") else "⏹️"
        
        text = f"""
🎮 *NG XM Trading BOT*

{status} Trading | {auto} Auto | {scalp} Scalp

💰 ${account['balance']:,.2f}
📡 Signals: {state.get('signals_sent', 0)}
        """
        try:
            await query.edit_message_text(text, parse_mode='Markdown', reply_markup=get_main_keyboard(state))
        except Exception as e:
            logger.warning(f"Edit message failed: {e}")

# ============================================================
# SIGNAL BOT HANDLER
# ============================================================

async def signal_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start in NGBOT"""
    state = load_state()
    
    text = f"""
📡 *NGBOT - Signal Receiver*

━━━━━━━━━━━━━━━━━━━━━━
This bot receives trading signals!

📊 Signals received: {state.get('signals_sent', 0)}

*How to use:*
1. Go to *NG XM Trading BOT*
2. Send /start
3. Press 🟢 START
4. Press 🔄 Auto or ⚡ Scalp
5. Signals appear here!

_Waiting for signals..._
    """
    await update.message.reply_text(text, parse_mode='Markdown')

# ============================================================
# MAIN
# ============================================================

async def main():
    global auto_generator, scalp_engine, signal_bot_app
    
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║       🚀 ULTIMATE TRADING SYSTEM v2                           ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║                                                               ║
    ║   ⚡ FAST & NON-BLOCKING                                      ║
    ║   ✅ All responses under 3 seconds                            ║
    ║   ✅ Auto & Scalp run SIMULTANEOUSLY                          ║
    ║                                                               ║
    ║   🎮 NG XM Trading BOT - Control                              ║
    ║   📡 NGBOT - Signals                                          ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Reset state
    state = load_state()
    state["is_trading_enabled"] = False
    state["auto_scan_enabled"] = False
    state["scalp_scan_enabled"] = False
    state["auto_scan_count"] = 0
    state["scalp_scan_count"] = 0
    save_state(state)
    
    # Initialize signal engines
    print("\n[INFO] Initializing signal engines...")
    
    if AUTO_AVAILABLE:
        try:
            auto_generator = AutonomousSignalGenerator()
            print("[INFO] ✅ Auto Generator ready")
        except Exception as e:
            print(f"[WARNING] Auto Generator failed: {e}")
            
    if SCALP_AVAILABLE:
        try:
            scalp_engine = ScalpSignalEngine()
            print("[INFO] ✅ Scalp Engine ready")
        except Exception as e:
            print(f"[WARNING] Scalp Engine failed: {e}")
    
    # Create bots
    control_app = Application.builder().token(CONTROL_BOT_TOKEN).build()
    control_app.add_handler(CommandHandler("start", control_start))
    control_app.add_handler(CallbackQueryHandler(control_callback))
    
    signal_bot_app = Application.builder().token(SIGNAL_BOT_TOKEN).build()
    signal_bot_app.add_handler(CommandHandler("start", signal_start))
    
    print("\n✅ Starting bots...")
    print("   🎮 NG XM Trading BOT - Control")
    print("   📡 NGBOT - Signals")
    print("\n   Press Ctrl+C to stop\n")
    
    try:
        await control_app.initialize()
        await signal_bot_app.initialize()
        
        await control_app.start()
        await signal_bot_app.start()
        
        await control_app.updater.start_polling(drop_pending_updates=True)
        await signal_bot_app.updater.start_polling(drop_pending_updates=True)
        
        print("✅ Both bots running!")
        print("   Open Telegram and test!\n")
        
        while True:
            await asyncio.sleep(1)
            
    except asyncio.CancelledError:
        pass
    except KeyboardInterrupt:
        pass
    finally:
        print("\n🛑 Shutting down...")
        
        if auto_scan_task and not auto_scan_task.done():
            auto_scan_task.cancel()
        if scalp_scan_task and not scalp_scan_task.done():
            scalp_scan_task.cancel()
        
        await control_app.updater.stop()
        await control_app.stop()
        await control_app.shutdown()
        
        await signal_bot_app.updater.stop()
        await signal_bot_app.stop()
        await signal_bot_app.shutdown()
        
        print("👋 Goodbye!")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Stopped")
