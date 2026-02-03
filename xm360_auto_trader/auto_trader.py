"""
XM360 Auto Trader - Main Script

Automatically executes trades on XM360 broker based on
signals from the Telegram Trading Bot.

Usage:
    python auto_trader.py

Before running:
1. Install MetaTrader 5 and connect to XM360
2. Update config.py with your account credentials
3. Ensure the main trading bot is running

Author: Trading Bot Team
Version: 1.0.0
"""

import os
import sys
import time
import logging
import asyncio
from datetime import datetime
from typing import Dict, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from xm360_auto_trader import config
from xm360_auto_trader.mt5_connector import MT5Connector
from xm360_auto_trader.signal_receiver import SignalReceiver
from xm360_auto_trader.trade_manager import TradeManager

# Optional: Telegram notifications
try:
    import telegram
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False

# Setup logging
os.makedirs('xm360_auto_trader/logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('AutoTrader')


class XM360AutoTrader:
    """
    Main auto trading system that connects to XM360 via MT5
    and executes trades based on signals from the Telegram bot.
    """
    
    def __init__(self):
        self.mt5 = MT5Connector()
        self.trade_manager = None
        self.signal_receiver = None
        self.telegram_bot = None
        self.running = False
        
        # Initialize Telegram for notifications
        if TELEGRAM_AVAILABLE and config.SEND_TELEGRAM_NOTIFICATIONS:
            try:
                self.telegram_bot = telegram.Bot(token=config.TELEGRAM_BOT_TOKEN)
            except Exception as e:
                logger.warning(f"Telegram notifications disabled: {e}")
    
    async def send_notification(self, message: str):
        """Send notification to Telegram."""
        if self.telegram_bot and config.NOTIFICATION_CHAT_ID:
            try:
                await self.telegram_bot.send_message(
                    chat_id=config.NOTIFICATION_CHAT_ID,
                    text=message,
                    parse_mode='Markdown'
                )
            except Exception as e:
                logger.error(f"Failed to send notification: {e}")
    
    def on_signal_received(self, signal: Dict) -> Dict:
        """
        Callback when a new signal is received.
        
        Args:
            signal: Processed trading signal
            
        Returns:
            dict: Trade result
        """
        logger.info(f"📥 Signal received: {signal['symbol']} {signal['direction']}")
        
        # Check if we should trade this symbol
        if not self._should_trade_symbol(signal['symbol']):
            return {'skipped': True, 'reason': 'Symbol not in allowed list'}
        
        # Execute the trade
        success, result = self.trade_manager.execute_signal(signal)
        
        # Format result message
        if success:
            # Get price validation info from result
            signal_price = result.get('signal', {}).get('original_signal_price', signal.get('entry_price', 'N/A'))
            executed_price = result.get('price', 'Market')
            
            message = (
                f"✅ *Trade Executed*\n\n"
                f"Symbol: `{signal['symbol']}`\n"
                f"Direction: `{signal['direction']}`\n"
                f"Lot Size: `{signal.get('lot_size', config.DEFAULT_LOT_SIZE)}`\n"
                f"Signal Price: `{signal_price}`\n"
                f"Executed At: `{executed_price}`\n"
                f"SL: `{signal.get('stop_loss', 'None')}`\n"
                f"TP: `{signal.get('take_profit', 'None')}`\n"
                f"Ticket: `#{result.get('ticket')}`\n"
                f"✅ Price validated - trending near signal"
            )
        else:
            error_msg = result.get('error', 'Unknown')
            # Check if it's a price validation error
            is_price_error = 'price' in error_msg.lower() or 'deviation' in error_msg.lower()
            
            message = (
                f"❌ *Trade Rejected*\n\n"
                f"Symbol: `{signal['symbol']}`\n"
                f"Direction: `{signal['direction']}`\n"
                f"Signal Price: `{signal.get('entry_price', 'N/A')}`\n"
                f"Reason: `{error_msg}`"
            )
            
            if is_price_error:
                message += f"\n\n⚠️ *Price not trending near signal value*"
        
        # Send notification
        if config.SEND_TELEGRAM_NOTIFICATIONS:
            asyncio.create_task(self.send_notification(message))
        
        return result
    
    def _should_trade_symbol(self, symbol: str) -> bool:
        """Check if we should trade this symbol."""
        if config.BLOCKED_SYMBOLS and symbol in config.BLOCKED_SYMBOLS:
            return False
        if config.ALLOWED_SYMBOLS and symbol not in config.ALLOWED_SYMBOLS:
            return False
        return True
    
    def connect(self) -> bool:
        """
        Connect to MT5 and initialize trading.
        
        Returns:
            bool: True if successful
        """
        logger.info("=" * 60)
        logger.info("XM360 Auto Trader v1.0.0")
        logger.info("=" * 60)
        
        # Validate configuration
        if not config.ACCOUNT or not config.PASSWORD:
            logger.error("❌ Account credentials not configured!")
            logger.error("   Please update config.py with your XM360 account details.")
            return False
        
        # Connect to MT5
        logger.info("Connecting to MetaTrader 5...")
        if not self.mt5.connect():
            logger.error(f"❌ Failed to connect: {self.mt5.last_error}")
            return False
        
        # Initialize trade manager
        self.trade_manager = TradeManager(self.mt5)
        
        # Initialize signal receiver
        self.signal_receiver = SignalReceiver(on_signal_callback=self.on_signal_received)
        
        # Print account info
        account = self.mt5.get_account_info()
        logger.info("=" * 60)
        logger.info(f"✅ Connected to {account['trade_mode']} Account")
        logger.info(f"   Account: {account['login']}")
        logger.info(f"   Server: {account['server']}")
        logger.info(f"   Balance: ${account['balance']:.2f}")
        logger.info(f"   Equity: ${account['equity']:.2f}")
        logger.info(f"   Leverage: 1:{account['leverage']}")
        logger.info("=" * 60)
        
        return True
    
    def start(self):
        """Start the auto trader."""
        if not self.mt5.is_connected():
            if not self.connect():
                return
        
        self.running = True
        
        logger.info("🚀 Auto Trader started!")
        logger.info(f"   Checking for signals every {config.SIGNAL_CHECK_INTERVAL}s")
        logger.info(f"   Max risk per trade: {config.MAX_RISK_PERCENT}%")
        logger.info(f"   Default lot size: {config.DEFAULT_LOT_SIZE}")
        logger.info("")
        logger.info("Waiting for signals from Telegram Trading Bot...")
        logger.info("Press Ctrl+C to stop")
        logger.info("")
        
        # Start signal receiver
        self.signal_receiver.start_listening()
        
        try:
            # Main loop - keep running and monitor
            while self.running:
                # Check connection
                if not self.mt5.is_connected():
                    logger.warning("⚠️ Connection lost, reconnecting...")
                    if not self.mt5.connect():
                        time.sleep(10)
                        continue
                
                # Print periodic status
                self._print_status()
                
                time.sleep(60)  # Status update every minute
                
        except KeyboardInterrupt:
            logger.info("\n⏹️ Stopping auto trader...")
        
        finally:
            self.stop()
    
    def _print_status(self):
        """Print periodic status update."""
        summary = self.trade_manager.get_daily_summary()
        positions = self.trade_manager.get_open_positions()
        
        logger.info("-" * 50)
        logger.info(f"📊 Status Update - {datetime.now().strftime('%H:%M:%S')}")
        logger.info(f"   Balance: ${summary.get('balance', 0):.2f}")
        logger.info(f"   Equity: ${summary.get('equity', 0):.2f}")
        logger.info("-" * 50)
        logger.info(f"   💰 10% STRATEGY:")
        logger.info(f"      Max Trading Margin: ${summary.get('max_trading_margin', 0):.2f}")
        logger.info(f"      Margin Used: ${summary.get('margin_used', 0):.2f}")
        logger.info(f"      Available: ${summary.get('available_trading_balance', 0):.2f}")
        logger.info(f"      Usage: {summary.get('balance_usage_percent', 0):.1f}%")
        logger.info("-" * 50)
        logger.info(f"   Open Positions: {len(positions)}/{config.MAX_OPEN_POSITIONS}")
        logger.info(f"   Unrealized P/L: ${summary['unrealized_profit']:.2f}")
        logger.info(f"   Daily Trades: {summary['trades_executed']}/{config.MAX_DAILY_TRADES}")
        logger.info(f"   Daily Loss: ${summary['daily_loss']:.2f}/${summary['max_daily_loss']:.2f}")
        logger.info(f"   Can Trade: {'✅ Yes' if summary['can_trade'] else '❌ No'}")
        
        if not summary['can_trade']:
            logger.info(f"   Reasons: {', '.join(summary.get('status_reasons', []))}")
        
        if positions:
            logger.info("   📈 Positions:")
            for pos in positions:
                profit_emoji = "🟢" if pos['profit'] >= 0 else "🔴"
                logger.info(f"     {profit_emoji} {pos['symbol']} {pos['type']} {pos['volume']} lots @ {pos['open_price']:.5f} ({pos['profit']:+.2f})")
    
    def stop(self):
        """Stop the auto trader."""
        self.running = False
        
        if self.signal_receiver:
            self.signal_receiver.stop_listening()
        
        if self.mt5:
            self.mt5.disconnect()
        
        logger.info("Auto Trader stopped")


def main():
    """Main entry point."""
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║             XM360 AUTO TRADER v1.0.0                       ║
    ║     Automatic Trade Execution from Telegram Signals        ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    # Check for MetaTrader5 library
    try:
        import MetaTrader5
        print("✅ MetaTrader5 library found")
    except ImportError:
        print("❌ MetaTrader5 library not installed!")
        print("   Run: pip install MetaTrader5")
        print("\n   Note: MetaTrader5 only works on Windows with MT5 installed.")
        sys.exit(1)
    
    # Check configuration
    if not config.ACCOUNT or not config.PASSWORD:
        print("\n❌ Account not configured!")
        print("   Please update xm360_auto_trader/config.py with:")
        print("   - ACCOUNT = Your XM360 account number")
        print("   - PASSWORD = Your MT5 password")
        print("   - SERVER = Your MT5 server name")
        sys.exit(1)
    
    # Create and start auto trader
    trader = XM360AutoTrader()
    trader.start()


if __name__ == '__main__':
    main()
