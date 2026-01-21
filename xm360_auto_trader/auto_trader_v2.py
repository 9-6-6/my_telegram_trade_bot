"""
XM360 Auto Trader v2.0 - Enhanced with Trailing Stop & Multiple TPs
=====================================================================

FEATURES:
- Pending orders (trade activates at signal price)
- Trailing Stop Loss (SL moves as price moves in your favor)
- Multiple Take Profit targets (partial closes)
- Full automation of trade lifecycle
- 10% balance protection strategy

Usage:
    python auto_trader_v2.py

Author: Trading Bot Team
Version: 2.0.0
"""

import os
import sys
import time
import logging
import asyncio
from datetime import datetime
from typing import Dict, Optional, List

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules
from xm360_auto_trader import config
from xm360_auto_trader.mt5_connector import MT5Connector
from xm360_auto_trader.signal_receiver import SignalReceiver
from xm360_auto_trader.advanced_trade_manager import (
    AdvancedTradeManager, 
    TrailingStopConfig,
    AdvancedTrade
)

# Optional Telegram
try:
    import telegram
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False

# Setup logging
os.makedirs('xm360_auto_trader/logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('xm360_auto_trader/logs/auto_trader_v2.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('AutoTraderV2')


# ============================================================
# TRADING CONFIGURATION - CUSTOMIZE THESE VALUES
# ============================================================

TRAILING_STOP_CONFIG = {
    "XAUUSD": {
        "enabled": True,
        "activation_profit_dollars": 5.0,   # Start trailing after $5 profit
        "trail_distance_dollars": 3.0,      # Keep SL $3 behind price
        "step_dollars": 1.0                 # Minimum $1 to update SL
    },
    "BTCUSD": {
        "enabled": True,
        "activation_profit_dollars": 100.0,  # Start after $100 profit
        "trail_distance_dollars": 50.0,      # Keep SL $50 behind
        "step_dollars": 20.0                 # Minimum $20 to update
    },
    "EURUSD": {
        "enabled": True,
        "activation_profit_pips": 20,        # Start after 20 pips
        "trail_distance_pips": 10,           # Keep 10 pips behind
        "step_pips": 5                       # Minimum 5 pips to update
    }
}

# Multiple Take Profit Configuration
# Each TP specifies: price offset from entry and % of position to close
TAKE_PROFIT_CONFIG = {
    "XAUUSD": {
        "use_multiple_tps": True,
        "tp_levels": [
            {"offset_dollars": 10.0, "percent_close": 50},   # TP1: +$10, close 50%
            {"offset_dollars": 20.0, "percent_close": 100},  # TP2: +$20, close remaining
        ]
    },
    "BTCUSD": {
        "use_multiple_tps": True,
        "tp_levels": [
            {"offset_dollars": 200.0, "percent_close": 50},
            {"offset_dollars": 500.0, "percent_close": 100},
        ]
    },
    "EURUSD": {
        "use_multiple_tps": True,
        "tp_levels": [
            {"offset_pips": 30, "percent_close": 50},
            {"offset_pips": 50, "percent_close": 100},
        ]
    }
}

# Use pending orders (activate at signal price) or market orders (immediate)
USE_PENDING_ORDERS = True


class XM360AutoTraderV2:
    """
    Enhanced Auto Trader with Trailing Stops and Multiple Take Profits
    """
    
    def __init__(self):
        self.mt5 = MT5Connector()
        self.advanced_manager: Optional[AdvancedTradeManager] = None
        self.signal_receiver = None
        self.telegram_bot = None
        self.running = False
        self.monitor_interval = 1  # Check prices every 1 second
        
        # Initialize Telegram
        if TELEGRAM_AVAILABLE and config.SEND_TELEGRAM_NOTIFICATIONS:
            try:
                self.telegram_bot = telegram.Bot(token=config.TELEGRAM_BOT_TOKEN)
            except Exception as e:
                logger.warning(f"Telegram disabled: {e}")
    
    async def send_notification(self, message: str):
        """Send Telegram notification"""
        if self.telegram_bot and config.NOTIFICATION_CHAT_ID:
            try:
                await self.telegram_bot.send_message(
                    chat_id=config.NOTIFICATION_CHAT_ID,
                    text=message,
                    parse_mode='Markdown'
                )
            except Exception as e:
                logger.error(f"Notification failed: {e}")
    
    def _get_trailing_config(self, symbol: str) -> TrailingStopConfig:
        """Get trailing stop configuration for symbol"""
        
        cfg = TRAILING_STOP_CONFIG.get(symbol, {})
        
        if not cfg.get("enabled", True):
            return TrailingStopConfig(enabled=False)
        
        # Convert dollars to pips for gold/crypto
        if symbol == "XAUUSD":
            pip_value = 0.01
            return TrailingStopConfig(
                enabled=True,
                activation_profit_pips=cfg.get("activation_profit_dollars", 5) / pip_value,
                trail_distance_pips=cfg.get("trail_distance_dollars", 3) / pip_value,
                step_pips=cfg.get("step_dollars", 1) / pip_value
            )
        elif symbol == "BTCUSD":
            pip_value = 1.0
            return TrailingStopConfig(
                enabled=True,
                activation_profit_pips=cfg.get("activation_profit_dollars", 100) / pip_value,
                trail_distance_pips=cfg.get("trail_distance_dollars", 50) / pip_value,
                step_pips=cfg.get("step_dollars", 20) / pip_value
            )
        else:  # Forex
            return TrailingStopConfig(
                enabled=True,
                activation_profit_pips=cfg.get("activation_profit_pips", 20),
                trail_distance_pips=cfg.get("trail_distance_pips", 10),
                step_pips=cfg.get("step_pips", 5)
            )
    
    def _calculate_take_profits(
        self, 
        symbol: str, 
        direction: str, 
        entry_price: float,
        signal_tp: Optional[float] = None
    ) -> List[Dict]:
        """Calculate multiple take profit levels"""
        
        cfg = TAKE_PROFIT_CONFIG.get(symbol, {})
        
        # If not using multiple TPs, use single TP from signal
        if not cfg.get("use_multiple_tps", False):
            if signal_tp:
                return [{"price": signal_tp, "percent": 100}]
            return []
        
        take_profits = []
        tp_levels = cfg.get("tp_levels", [])
        
        for tp in tp_levels:
            if symbol in ["XAUUSD", "BTCUSD"]:
                # Dollar offset
                offset = tp.get("offset_dollars", 10)
            else:
                # Pip offset (forex)
                offset = tp.get("offset_pips", 20) * 0.0001  # Convert pips to price
            
            if direction == "BUY":
                tp_price = entry_price + offset
            else:
                tp_price = entry_price - offset
            
            take_profits.append({
                "price": tp_price,
                "percent": tp.get("percent_close", 50)
            })
        
        return take_profits
    
    def on_signal_received(self, signal: Dict) -> Dict:
        """
        Process incoming signal with advanced features
        """
        logger.info(f"📥 Signal: {signal['symbol']} {signal['direction']} @ ${signal.get('entry_price', 'Market')}")
        
        # Check if symbol is allowed
        if not self._should_trade_symbol(signal['symbol']):
            return {'skipped': True, 'reason': 'Symbol not allowed'}
        
        symbol = signal['symbol']
        direction = signal['direction']
        entry_price = signal.get('entry_price', self._get_current_price(symbol))
        stop_loss = signal.get('stop_loss', entry_price - 10 if direction == "BUY" else entry_price + 10)
        signal_tp = signal.get('take_profit')
        lot_size = signal.get('lot_size', config.DEFAULT_LOT_SIZE)
        
        # Get trailing stop configuration
        trailing_config = self._get_trailing_config(symbol)
        
        # Calculate multiple take profit levels
        take_profits = self._calculate_take_profits(symbol, direction, entry_price, signal_tp)
        
        # Check balance before trading
        if not self._check_balance():
            return {'skipped': True, 'reason': 'Insufficient balance or risk limit reached'}
        
        # Create trade with advanced features
        try:
            if USE_PENDING_ORDERS:
                # Create pending order (activates at signal price)
                trade = self.advanced_manager.create_pending_order(
                    symbol=symbol,
                    direction=direction,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profits=take_profits,
                    lot_size=lot_size,
                    trailing_config=trailing_config
                )
                order_type = "PENDING"
            else:
                # Create immediate market order
                trade = self.advanced_manager.create_market_order(
                    symbol=symbol,
                    direction=direction,
                    stop_loss=stop_loss,
                    take_profits=take_profits,
                    lot_size=lot_size,
                    trailing_config=trailing_config
                )
                order_type = "MARKET"
            
            # Send notification
            tp_str = "\n".join([f"   TP{i+1}: ${tp['price']:.2f} ({tp['percent']}%)" 
                               for i, tp in enumerate(take_profits)])
            
            message = f"""
✅ *Trade Created ({order_type})*

📊 Symbol: `{symbol}`
📈 Direction: `{direction}`
💰 Entry: `${entry_price:.2f}`
🛑 Stop Loss: `${stop_loss:.2f}`
🎯 Take Profits:
{tp_str}
📈 Trailing Stop: `{'Enabled' if trailing_config.enabled else 'Disabled'}`
🎫 Ticket: `#{trade.ticket}`

_Trade will auto-manage: SL will trail profits, TPs will auto-close_
            """
            
            if config.SEND_TELEGRAM_NOTIFICATIONS:
                asyncio.create_task(self.send_notification(message))
            
            return {'success': True, 'ticket': trade.ticket}
            
        except Exception as e:
            logger.error(f"Trade creation failed: {e}")
            return {'error': str(e)}
    
    def _should_trade_symbol(self, symbol: str) -> bool:
        """Check if symbol is allowed for trading"""
        if config.BLOCKED_SYMBOLS and symbol in config.BLOCKED_SYMBOLS:
            return False
        if config.ALLOWED_SYMBOLS and symbol not in config.ALLOWED_SYMBOLS:
            return False
        return True
    
    def _check_balance(self) -> bool:
        """Check if we can trade based on balance rules"""
        try:
            account = self.mt5.get_account_info()
            balance = account.get('balance', 0)
            margin_used = account.get('margin', 0)
            
            # 10% rule
            max_trading_margin = balance * (config.MAX_BALANCE_USAGE_PERCENT / 100)
            available = max_trading_margin - margin_used
            
            if balance < config.MIN_BALANCE_TO_TRADE:
                logger.warning(f"❌ Balance ${balance:.2f} below minimum ${config.MIN_BALANCE_TO_TRADE}")
                return False
            
            if available <= 0:
                logger.warning(f"❌ 10% limit reached. Used ${margin_used:.2f} of ${max_trading_margin:.2f}")
                return False
            
            return True
        except:
            return True  # Allow if we can't check
    
    def _get_current_price(self, symbol: str) -> float:
        """Get current market price"""
        try:
            return self.mt5.get_current_price(symbol)
        except:
            # Fallback prices
            return {"XAUUSD": 2650.0, "BTCUSD": 101000.0, "EURUSD": 1.0850}.get(symbol, 100)
    
    def connect(self) -> bool:
        """Connect to MT5"""
        logger.info("=" * 60)
        logger.info("XM360 Auto Trader v2.0 (Enhanced)")
        logger.info("=" * 60)
        
        if not config.ACCOUNT or not config.PASSWORD:
            logger.error("❌ Account not configured in config.py")
            return False
        
        logger.info("Connecting to MetaTrader 5...")
        if not self.mt5.connect():
            logger.error(f"❌ Connection failed: {self.mt5.last_error}")
            return False
        
        # Initialize advanced trade manager
        self.advanced_manager = AdvancedTradeManager(self.mt5)
        
        # Initialize signal receiver
        self.signal_receiver = SignalReceiver(on_signal_callback=self.on_signal_received)
        
        # Print account info
        account = self.mt5.get_account_info()
        logger.info("=" * 60)
        logger.info(f"✅ Connected to {account.get('trade_mode', 'Demo')} Account")
        logger.info(f"   Account: {account.get('login', config.ACCOUNT)}")
        logger.info(f"   Balance: ${account.get('balance', 0):.2f}")
        logger.info(f"   Leverage: 1:{account.get('leverage', 500)}")
        logger.info("=" * 60)
        logger.info("")
        logger.info("📋 ADVANCED FEATURES ENABLED:")
        logger.info(f"   ✅ Pending Orders: {USE_PENDING_ORDERS}")
        logger.info(f"   ✅ Trailing Stop Loss: Auto-adjusts as profit grows")
        logger.info(f"   ✅ Multiple Take Profits: Partial closes at targets")
        logger.info(f"   ✅ 10% Balance Rule: Protected risk management")
        logger.info("")
        
        return True
    
    def start(self):
        """Start the enhanced auto trader"""
        if not self.mt5.is_connected():
            if not self.connect():
                return
        
        self.running = True
        
        logger.info("🚀 Enhanced Auto Trader Started!")
        logger.info("   Features: Trailing SL | Multiple TPs | Pending Orders")
        logger.info("   Press Ctrl+C to stop")
        logger.info("")
        
        # Start signal receiver
        self.signal_receiver.start_listening()
        
        try:
            last_status_time = 0
            
            while self.running:
                # Monitor and manage trades (trailing stops, TPs, pending activations)
                self.advanced_manager.monitor_and_manage_trades()
                
                # Check connection
                if not self.mt5.is_connected():
                    logger.warning("⚠️ Reconnecting...")
                    if not self.mt5.connect():
                        time.sleep(10)
                        continue
                
                # Print status every 60 seconds
                if time.time() - last_status_time > 60:
                    self._print_status()
                    last_status_time = time.time()
                
                # Monitor interval (check prices frequently for trailing)
                time.sleep(self.monitor_interval)
                
        except KeyboardInterrupt:
            logger.info("\n⏹️ Stopping...")
        finally:
            self.stop()
    
    def _print_status(self):
        """Print status update"""
        logger.info("-" * 60)
        logger.info(f"📊 Status - {datetime.now().strftime('%H:%M:%S')}")
        
        # Account info
        try:
            account = self.mt5.get_account_info()
            balance = account.get('balance', 0)
            equity = account.get('equity', 0)
            margin = account.get('margin', 0)
            
            max_margin = balance * (config.MAX_BALANCE_USAGE_PERCENT / 100)
            usage_pct = (margin / max_margin * 100) if max_margin > 0 else 0
            
            logger.info(f"   Balance: ${balance:.2f} | Equity: ${equity:.2f}")
            logger.info(f"   10% Limit: ${margin:.2f} / ${max_margin:.2f} ({usage_pct:.1f}%)")
        except:
            pass
        
        # Active trades
        active_count = len(self.advanced_manager.active_trades)
        pending_count = sum(1 for t in self.advanced_manager.active_trades.values() if t.is_pending)
        
        logger.info(f"   Active Trades: {active_count} ({pending_count} pending)")
        
        for ticket, trade in self.advanced_manager.active_trades.items():
            status = "⏳ PENDING" if trade.is_pending else "🟢 ACTIVE"
            trail = "📈 Trailing" if trade.trailing_activated else ""
            logger.info(f"     {status} #{ticket}: {trade.symbol} {trade.direction} @ ${trade.entry_price:.2f} SL:${trade.current_stop_loss:.2f} {trail}")
        
        logger.info("-" * 60)
    
    def stop(self):
        """Stop the auto trader"""
        self.running = False
        
        if self.signal_receiver:
            self.signal_receiver.stop_listening()
        
        if self.mt5:
            self.mt5.disconnect()
        
        logger.info("Auto Trader stopped")


def main():
    """Main entry point"""
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║           XM360 AUTO TRADER v2.0 (ENHANCED)                   ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║  NEW FEATURES:                                                ║
    ║  ✅ Pending Orders - Trade activates at signal price          ║
    ║  ✅ Trailing Stop Loss - SL moves as profit grows             ║
    ║  ✅ Multiple Take Profits - Partial closes at targets         ║
    ║  ✅ Full Automation - Everything managed automatically        ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Check MetaTrader5
    try:
        import MetaTrader5
        print("✅ MetaTrader5 library found")
    except ImportError:
        print("❌ MetaTrader5 not installed!")
        print("   Run: pip install MetaTrader5")
        sys.exit(1)
    
    # Check config
    if not config.ACCOUNT or not config.PASSWORD:
        print("\n❌ Account not configured!")
        print("   Update xm360_auto_trader/config.py")
        sys.exit(1)
    
    # Start
    trader = XM360AutoTraderV2()
    trader.start()


if __name__ == '__main__':
    main()
