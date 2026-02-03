"""
Test Trade Script - XM360 Demo Account

This script tests placing a trade on your XM360 account.
Since MT5 desktop is not available, it runs in SIMULATION mode
to demonstrate how the auto trader works.

When MT5 is installed, it will automatically use real execution.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xm360_auto_trader import config
from xm360_auto_trader.xm_api_connector import XMWebConnector, get_connector
from xm360_auto_trader.trade_manager import TradeManager
from xm360_auto_trader.risk_manager import RiskManager


def test_simulated_trade():
    """Test a simulated trade on XM360."""
    
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║         XM360 TEST TRADE (SIMULATION MODE)                 ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    # Use simulation connector
    connector = XMWebConnector(simulation_mode=True)
    
    # Connect
    print("\n1️⃣ Connecting to XM360...")
    if not connector.connect():
        print("❌ Connection failed!")
        return
    
    print("✅ Connected!")
    
    # Get account info
    print("\n2️⃣ Account Information:")
    account = connector.get_account_info()
    print(f"   Account: {account['login']}")
    print(f"   Balance: ${account['balance']:.2f}")
    print(f"   Mode: {account['trade_mode']}")
    
    # Get current price for XAUUSD (Gold)
    print("\n3️⃣ Getting current price for XAUUSD (Gold)...")
    
    # Update with approximate real prices
    connector.update_price('XAUUSD', 2650.00, 2650.50)
    
    prices = connector.get_current_price('XAUUSD')
    if prices:
        bid, ask = prices
        print(f"   Bid: ${bid:.2f}")
        print(f"   Ask: ${ask:.2f}")
    
    # Place a test BUY order
    print("\n4️⃣ Placing TEST BUY order for XAUUSD...")
    print("   Symbol: XAUUSD (Gold)")
    print("   Direction: BUY")
    print("   Lot Size: 0.01")
    print("   Stop Loss: $2645.00")
    print("   Take Profit: $2660.00")
    
    success, result = connector.place_order(
        symbol='XAUUSD',
        order_type='BUY',
        lot_size=0.01,
        stop_loss=2645.00,
        take_profit=2660.00,
        comment='Test Trade from Bot'
    )
    
    if success:
        print("\n✅ ORDER PLACED SUCCESSFULLY!")
        print(f"   Ticket: #{result['ticket']}")
        print(f"   Entry Price: ${result['price']:.2f}")
        print(f"   ⚠️ This is a SIMULATED trade (not real)")
    else:
        print(f"\n❌ Order failed: {result.get('error')}")
        return
    
    # Check open positions
    print("\n5️⃣ Open Positions:")
    positions = connector.get_open_positions()
    for pos in positions:
        print(f"   • #{pos['ticket']} {pos['symbol']} {pos['type']} {pos['volume']} lots")
        print(f"     Open: ${pos['open_price']:.2f}, Current: ${pos['current_price']:.2f}")
        print(f"     P/L: ${pos['profit']:.2f}")
    
    # Close the position
    print("\n6️⃣ Closing the test position...")
    success, result = connector.close_position(positions[0]['ticket'])
    
    if success:
        print(f"   ✅ Position closed!")
        print(f"   Closed at: ${result['closed_at']:.2f}")
        print(f"   Profit: ${result['profit']:.2f}")
    
    # Final account state
    print("\n7️⃣ Final Account State:")
    account = connector.get_account_info()
    print(f"   Balance: ${account['balance']:.2f}")
    print(f"   Equity: ${account['equity']:.2f}")
    print(f"   Open Positions: {len(connector.get_open_positions())}")
    
    # Disconnect
    connector.disconnect()
    
    print("\n" + "="*60)
    print("✅ SIMULATION TEST COMPLETE!")
    print("="*60)
    print("""
    📌 SUMMARY:
    This was a SIMULATED trade to demonstrate how the auto trader works.
    
    To execute REAL trades, you need to:
    1. Install MetaTrader 5 desktop application
    2. Login to your XM360 account in MT5
    3. Keep MT5 running while the auto trader operates
    
    Download MT5 from: https://www.xm.com/mt5
    """)


def test_signal_to_trade():
    """Test the full signal-to-trade flow."""
    
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║       SIGNAL → TRADE TEST (SIMULATION MODE)                ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    # Create connector
    connector = XMWebConnector(simulation_mode=True)
    connector.connect()
    
    # Update with realistic prices
    connector.update_price('XAUUSD', 2650.00, 2650.50)
    connector.update_price('EURUSD', 1.0850, 1.0852)
    
    # Simulate receiving a signal from the trading bot
    signal = {
        'symbol': 'XAUUSD',
        'direction': 'BUY',
        'entry_price': 2650.50,  # Close to current ask
        'stop_loss': 2645.00,
        'take_profit': 2660.00,
        'confidence': 85,
        'source': 'TelegramBot'
    }
    
    print("\n📥 SIGNAL RECEIVED FROM BOT:")
    print(f"   Symbol: {signal['symbol']}")
    print(f"   Direction: {signal['direction']}")
    print(f"   Entry Price: ${signal['entry_price']:.2f}")
    print(f"   Stop Loss: ${signal['stop_loss']:.2f}")
    print(f"   Take Profit: ${signal['take_profit']:.2f}")
    print(f"   Confidence: {signal['confidence']}%")
    
    # Validate price
    print("\n📊 PRICE VALIDATION:")
    bid, ask = connector.get_current_price('XAUUSD')
    print(f"   Signal Price: ${signal['entry_price']:.2f}")
    print(f"   Current Ask: ${ask:.2f}")
    
    deviation = abs(ask - signal['entry_price'])
    max_deviation = config.MAX_PRICE_DEVIATION_GOLD
    
    print(f"   Deviation: ${deviation:.2f}")
    print(f"   Max Allowed: ${max_deviation:.2f}")
    
    if deviation <= max_deviation:
        print("   ✅ Price VALID - matches market!")
        
        # Execute trade
        print("\n🚀 EXECUTING TRADE...")
        success, result = connector.place_order(
            symbol=signal['symbol'],
            order_type=signal['direction'],
            lot_size=0.01,
            stop_loss=signal['stop_loss'],
            take_profit=signal['take_profit'],
            comment=f"Signal: {signal['source']}"
        )
        
        if success:
            print(f"\n✅ TRADE EXECUTED!")
            print(f"   Ticket: #{result['ticket']}")
            print(f"   Executed at: ${result['price']:.2f}")
        else:
            print(f"\n❌ Trade failed: {result.get('error')}")
    else:
        print("   ❌ Price INVALID - deviation too high!")
        print("   ⚠️ Trade NOT executed")
    
    connector.disconnect()
    print("\n" + "="*60)


if __name__ == '__main__':
    print("Choose a test:")
    print("1. Basic Trade Test")
    print("2. Signal-to-Trade Test")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == '2':
        test_signal_to_trade()
    else:
        test_simulated_trade()
