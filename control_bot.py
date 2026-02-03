"""
🎮 CONTROL BOT - NG XM Trading BOT
===================================
This bot provides START/STOP buttons to control trading.

IMPORTANT: This works WITH copy_trading_bot.py - NOT replacing it!
- Real signals come from NGBOT (copy_trading_bot.py) with TradingView data
- This bot just provides a button interface for START/STOP control

Run BOTH bots:
1. python copy_trading_bot.py  (NGBOT - signals)
2. python control_bot.py       (NG XM Trading BOT - control)
"""

import asyncio
import logging
import json
import os
from datetime import datetime
from typing import Dict

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

# ============================================================
# CONFIGURATION
# ============================================================

# NG XM Trading BOT - Control Bot Token
CONTROL_BOT_TOKEN = "7849517577:AAGx8PhFyAf-cEFt06pfL_CPT8x9REVB1_U"
ADMIN_CHAT_ID = "603932135"

# XM360 Account Info
XM_ACCOUNT = "315982803"
XM_SERVER = "XMGlobal-MT5 7"

# ============================================================
# SHARED STATE FILE
# ============================================================

STATE_FILE = "trading_state.json"

def load_state() -> Dict:
    """Load trading state"""
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
    """Save trading state"""
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2, default=str)

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================
# KEYBOARD LAYOUTS
# ============================================================

def get_main_keyboard(state: Dict):
    """Main control panel"""
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
    """Settings keyboard"""
    trailing = "✅" if state.get("trailing_enabled", True) else "❌"
    risk = state.get("max_risk_percent", 10)
    
    keyboard = [
        [InlineKeyboardButton(f"📈 Trailing: {trailing}", callback_data="toggle_trailing")],
        [InlineKeyboardButton(f"⚠️ Risk: {risk}%", callback_data="show_risk")],
        [InlineKeyboardButton("◀️ Back", callback_data="menu")]
    ]
    return InlineKeyboardMarkup(keyboard)

# ============================================================
# HANDLERS
# ============================================================

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start"""
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

*This controls your trading.*
*Signals come from NGBOT!*

*How to use:*
1️⃣ Tap START to enable
2️⃣ Tap Auto/Scalp Scan
3️⃣ Watch signals in NGBOT
4️⃣ Tap STOP when done

━━━━━━━━━━━━━━━━━━━━━━
🏦 {XM_ACCOUNT} | {XM_SERVER}
━━━━━━━━━━━━━━━━━━━━━━
    """
    
    await update.message.reply_text(
        text, parse_mode='Markdown',
        reply_markup=get_main_keyboard(state)
    )

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle button presses"""
    query = update.callback_query
    await query.answer()
    
    action = query.data
    state = load_state()
    
    if action == "start_trading":
        state["is_trading_enabled"] = True
        state["start_time"] = datetime.now().isoformat()
        save_state(state)
        
        text = """
🟢 *TRADING ENABLED!*

Now activate scanning:
• 🔄 Auto Scan - swing signals
• ⚡ Scalp Scan - quick trades

Or use NGBOT directly:
`/autoscan` or `/scalpscan`
        """
        await query.edit_message_text(text, parse_mode='Markdown', reply_markup=get_main_keyboard(state))
        logger.info("✅ Trading ENABLED")
        
    elif action == "stop_trading":
        state["is_trading_enabled"] = False
        state["auto_scan_enabled"] = False
        state["scalp_scan_enabled"] = False
        state["start_time"] = None
        save_state(state)
        
        text = "🔴 *TRADING STOPPED!*\n\nNo new trades. Tap START to resume."
        await query.edit_message_text(text, parse_mode='Markdown', reply_markup=get_main_keyboard(state))
        logger.info("🔴 Trading STOPPED")
        
    elif action == "start_autoscan":
        if not state.get("is_trading_enabled"):
            await query.answer("⚠️ Enable trading first!", show_alert=True)
            return
        state["auto_scan_enabled"] = True
        save_state(state)
        
        text = """
🔄 *AUTO SCAN ON!*

Scanning 100+ markets:
• Forex pairs
• Gold, Silver
• BTC, ETH
• Indices

Use `/autoscan` in NGBOT!
        """
        await query.edit_message_text(text, parse_mode='Markdown', reply_markup=get_main_keyboard(state))
        logger.info("🔄 Auto Scan ON")
        
    elif action == "stop_autoscan":
        state["auto_scan_enabled"] = False
        save_state(state)
        await query.edit_message_text("⏹️ Auto Scan OFF", parse_mode='Markdown', reply_markup=get_main_keyboard(state))
        
    elif action == "start_scalpscan":
        if not state.get("is_trading_enabled"):
            await query.answer("⚠️ Enable trading first!", show_alert=True)
            return
        state["scalp_scan_enabled"] = True
        save_state(state)
        
        text = """
⚡ *SCALP SCAN ON!*

AI scalp detection:
• 85%+ confidence
• 5min-1hour TF
• Quick targets

Use `/scalpscan` in NGBOT!
        """
        await query.edit_message_text(text, parse_mode='Markdown', reply_markup=get_main_keyboard(state))
        logger.info("⚡ Scalp Scan ON")
        
    elif action == "stop_scalpscan":
        state["scalp_scan_enabled"] = False
        save_state(state)
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

*This is CONTROL bot.*
Signals from NGBOT!

🟢 START - enable
🔴 STOP - disable
🔄 Auto - swing signals
⚡ Scalp - quick signals

*In NGBOT use:*
/newsignal XAUUSD
/scalp BTCUSD
/autoscan
/scalpscan
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
# MAIN
# ============================================================

def main():
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║       🎮 NG XM Trading BOT - CONTROL                          ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║                                                               ║
    ║   START/STOP buttons for trading control                      ║
    ║   Real signals from NGBOT (copy_trading_bot.py)               ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Build app and DROP all old pending updates
    app = Application.builder().token(CONTROL_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CallbackQueryHandler(button_callback))
    
    print("✅ Control Bot started!")
    print("   Press Ctrl+C to stop\n")
    
    # Drop pending updates to ignore old messages
    app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)

if __name__ == "__main__":
    main()
