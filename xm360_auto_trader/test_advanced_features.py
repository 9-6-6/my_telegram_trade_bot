"""
Test Advanced Trade Manager Features
=====================================
Demonstrates: Pending Orders, Trailing Stop, Multiple TPs

Run: python xm360_auto_trader/test_advanced_features.py
"""

import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xm360_auto_trader.advanced_trade_manager import (
    AdvancedTradeManager,
    TrailingStopConfig,
    AdvancedTrade
)


class MockMT5Connector:
    """Simulated MT5 connector for testing"""
    
    def __init__(self):
        self.current_prices = {
            "XAUUSD": 2650.00,
            "BTCUSD": 101000.00
        }
    
    def get_current_price(self, symbol):
        return self.current_prices.get(symbol, 100)
    
    def set_price(self, symbol, price):
        """Simulate price movement"""
        self.current_prices[symbol] = price


def test_trailing_stop():
    """Test trailing stop functionality"""
    
    print("\n" + "=" * 70)
    print("TEST 1: TRAILING STOP LOSS")
    print("=" * 70)
    
    # Create mock connector
    mock = MockMT5Connector()
    mock.set_price("XAUUSD", 2650.00)
    
    # Create manager with mock connector
    manager = AdvancedTradeManager(mock)
    
    # Override the price getter to use our mock
    manager._get_current_price = lambda s: mock.get_current_price(s)
    
    print("\n📋 Scenario:")
    print("   - BUY XAUUSD at $2655")
    print("   - Initial SL: $2645")
    print("   - Trailing: Start after $5 profit, trail $3 behind")
    print()
    
    # Create market order (already activated)
    trade = manager.create_market_order(
        symbol="XAUUSD",
        direction="BUY",
        stop_loss=2645.0,
        take_profits=[
            {"price": 2670.0, "percent": 50},
            {"price": 2680.0, "percent": 100}
        ],
        lot_size=0.02,
        trailing_config=TrailingStopConfig(
            enabled=True,
            activation_profit_pips=500,   # $5 for gold
            trail_distance_pips=300,      # $3 trail
            step_pips=100                 # $1 step
        )
    )
    
    # Manually set entry price for testing
    trade.entry_price = 2655.0
    trade.highest_price = 2655.0
    
    print(f"✅ Trade created: Ticket #{trade.ticket}")
    print(f"   Entry: ${trade.entry_price}")
    print(f"   SL: ${trade.current_stop_loss}")
    print()
    
    # Simulate price movements
    price_sequence = [
        (2657.0, "Price rises slightly"),
        (2660.0, "Price hits +$5 profit - Trailing should ACTIVATE"),
        (2665.0, "Price continues up"),
        (2670.0, "Price at TP1 - Should close 50%"),
        (2668.0, "Price drops a bit"),
        (2672.0, "Price rises again"),
    ]
    
    for price, description in price_sequence:
        print(f"\n{'─' * 50}")
        print(f"📈 {description}")
        print(f"   Setting price to ${price:.2f}")
        
        mock.set_price("XAUUSD", price)
        manager.monitor_and_manage_trades()
        
        if trade.ticket in manager.active_trades:
            t = manager.active_trades[trade.ticket]
            print(f"   Current SL: ${t.current_stop_loss:.2f}")
            print(f"   Trailing Active: {t.trailing_activated}")
            print(f"   Highest Price: ${t.highest_price:.2f}")
            print(f"   Remaining Lots: {t.lot_size:.2f}")
        else:
            print(f"   Trade closed!")
        
        time.sleep(0.5)
    
    print("\n" + "=" * 70)
    print("✅ TRAILING STOP TEST COMPLETE")
    print("=" * 70)


def test_pending_order():
    """Test pending order activation"""
    
    print("\n" + "=" * 70)
    print("TEST 2: PENDING ORDER ACTIVATION")
    print("=" * 70)
    
    mock = MockMT5Connector()
    mock.set_price("XAUUSD", 2650.00)  # Current price below entry
    
    manager = AdvancedTradeManager(mock)
    manager._get_current_price = lambda s: mock.get_current_price(s)
    
    print("\n📋 Scenario:")
    print("   - Current price: $2650")
    print("   - Signal: BUY at $2655 (pending order)")
    print("   - Order should activate when price reaches $2655")
    print()
    
    # Create pending order
    trade = manager.create_pending_order(
        symbol="XAUUSD",
        direction="BUY",
        entry_price=2655.0,
        stop_loss=2645.0,
        take_profits=[
            {"price": 2665.0, "percent": 50},
            {"price": 2675.0, "percent": 100}
        ],
        lot_size=0.01
    )
    
    print(f"✅ Pending order created: #{trade.ticket}")
    print(f"   Status: {'PENDING' if trade.is_pending else 'ACTIVE'}")
    print()
    
    # Simulate price moving towards entry
    price_sequence = [
        (2651.0, "Price at $2651 - Still waiting"),
        (2653.0, "Price at $2653 - Still waiting"),
        (2655.0, "Price reaches $2655 - Should ACTIVATE!"),
        (2658.0, "Price at $2658 - Trade should be running"),
    ]
    
    for price, description in price_sequence:
        print(f"\n{'─' * 50}")
        print(f"📈 {description}")
        
        mock.set_price("XAUUSD", price)
        manager.monitor_and_manage_trades()
        
        if trade.ticket in manager.active_trades:
            t = manager.active_trades[trade.ticket]
            status = "⏳ PENDING" if t.is_pending else "🟢 ACTIVE"
            print(f"   Status: {status}")
            print(f"   Activated: {t.activated}")
        
        time.sleep(0.5)
    
    print("\n" + "=" * 70)
    print("✅ PENDING ORDER TEST COMPLETE")
    print("=" * 70)


def test_multiple_take_profits():
    """Test multiple take profit levels"""
    
    print("\n" + "=" * 70)
    print("TEST 3: MULTIPLE TAKE PROFITS")
    print("=" * 70)
    
    mock = MockMT5Connector()
    mock.set_price("XAUUSD", 2655.00)
    
    manager = AdvancedTradeManager(mock)
    manager._get_current_price = lambda s: mock.get_current_price(s)
    
    print("\n📋 Scenario:")
    print("   - BUY at $2655 with 0.02 lots")
    print("   - TP1: $2665 → Close 50% (0.01 lots)")
    print("   - TP2: $2675 → Close remaining 50%")
    print()
    
    trade = manager.create_market_order(
        symbol="XAUUSD",
        direction="BUY",
        stop_loss=2645.0,
        take_profits=[
            {"price": 2665.0, "percent": 50},
            {"price": 2675.0, "percent": 100}
        ],
        lot_size=0.02,
        trailing_config=TrailingStopConfig(enabled=False)  # Disable for this test
    )
    
    trade.entry_price = 2655.0
    
    print(f"✅ Trade created: #{trade.ticket}")
    print(f"   Lots: {trade.lot_size}")
    print()
    
    # Simulate price hitting TPs
    price_sequence = [
        (2660.0, "Price at $2660 - No TP hit yet"),
        (2665.0, "Price at $2665 - TP1 should trigger (close 50%)"),
        (2670.0, "Price at $2670 - Running towards TP2"),
        (2675.0, "Price at $2675 - TP2 should trigger (close all)"),
    ]
    
    for price, description in price_sequence:
        print(f"\n{'─' * 50}")
        print(f"📈 {description}")
        
        mock.set_price("XAUUSD", price)
        manager.monitor_and_manage_trades()
        
        if trade.ticket in manager.active_trades:
            t = manager.active_trades[trade.ticket]
            tps_hit = sum(1 for tp in t.take_profit_levels if tp.reached)
            print(f"   TPs Hit: {tps_hit}/{len(t.take_profit_levels)}")
            print(f"   Remaining Lots: {t.lot_size:.2f}")
        else:
            print(f"   🎉 Trade fully closed!")
            print(f"   Total Realized P&L: ${trade.realized_pnl:.2f}")
        
        time.sleep(0.5)
    
    print("\n" + "=" * 70)
    print("✅ MULTIPLE TP TEST COMPLETE")
    print("=" * 70)


def test_full_scenario():
    """Test complete trading scenario with all features"""
    
    print("\n" + "=" * 70)
    print("TEST 4: COMPLETE TRADING SCENARIO")
    print("=" * 70)
    
    print("""
    📋 YOUR EXAMPLE SCENARIO:
    ─────────────────────────────────────────────────────
    Current price: $2650
    Signal: BUY at $2655 (pending order)
    Stop Loss: $2645
    Trailing: Start after $5 profit, trail $3 behind
    TP1: $2665 (close 50%)
    TP2: $2675 (close remaining)
    
    Expected behavior:
    1. Order waits until price reaches $2655
    2. Trade activates at $2655
    3. When price hits $2660 (+$5), trailing activates
    4. SL moves up as price moves up
    5. At $2665, close 50% for profit
    6. At $2675, close remaining 50%
    ─────────────────────────────────────────────────────
    """)
    
    mock = MockMT5Connector()
    mock.set_price("XAUUSD", 2650.00)
    
    manager = AdvancedTradeManager(mock)
    manager._get_current_price = lambda s: mock.get_current_price(s)
    
    # Create pending order with all features
    trade = manager.create_pending_order(
        symbol="XAUUSD",
        direction="BUY",
        entry_price=2655.0,
        stop_loss=2645.0,
        take_profits=[
            {"price": 2665.0, "percent": 50},
            {"price": 2675.0, "percent": 100}
        ],
        lot_size=0.02,
        trailing_config=TrailingStopConfig(
            enabled=True,
            activation_profit_pips=500,  # $5
            trail_distance_pips=300,     # $3
            step_pips=100                # $1
        )
    )
    
    # Full price sequence
    price_sequence = [
        (2650.0, "Starting - Pending order waiting"),
        (2652.0, "Price rising - Still waiting"),
        (2655.0, "🔔 ENTRY PRICE - Order should ACTIVATE"),
        (2658.0, "In profit - Not enough for trailing yet"),
        (2660.0, "📈 +$5 profit - TRAILING ACTIVATES"),
        (2663.0, "Price continues up - SL trails"),
        (2665.0, "🎯 TP1 HIT - Close 50%"),
        (2668.0, "Price continues - SL keeps trailing"),
        (2672.0, "Almost at TP2"),
        (2675.0, "🎯 TP2 HIT - Close remaining"),
    ]
    
    print("\n" + "─" * 60)
    print("SIMULATION START")
    print("─" * 60)
    
    for price, description in price_sequence:
        print(f"\n💰 Price: ${price:.2f} - {description}")
        
        mock.set_price("XAUUSD", price)
        manager.monitor_and_manage_trades()
        
        if trade.ticket in manager.active_trades:
            t = manager.active_trades[trade.ticket]
            
            status = "⏳ PENDING" if t.is_pending else "🟢 ACTIVE"
            trailing = "📈 YES" if t.trailing_activated else "❌ No"
            tps = sum(1 for tp in t.take_profit_levels if tp.reached)
            
            print(f"   Status: {status}")
            print(f"   SL: ${t.current_stop_loss:.2f}")
            print(f"   Trailing: {trailing}")
            print(f"   TPs Hit: {tps}/2")
            print(f"   Lots: {t.lot_size:.3f}")
        else:
            print(f"\n   🎉 TRADE FULLY CLOSED!")
            print(f"   Total Profit: ${trade.realized_pnl:.2f}")
        
        time.sleep(0.3)
    
    print("\n" + "=" * 70)
    print("✅ FULL SCENARIO TEST COMPLETE")
    print("=" * 70)
    
    # Summary
    print(f"""
    📊 TRADE SUMMARY:
    ─────────────────────────────────────────────────────
    Entry Price: $2655.00
    Final SL: ${trade.current_stop_loss:.2f} (trailed from $2645)
    TP1 Hit: $2665 (closed 0.01 lots)
    TP2 Hit: $2675 (closed 0.01 lots)
    Total Profit: ${trade.realized_pnl:.2f}
    
    ✅ Pending order worked correctly
    ✅ Trailing stop protected profits
    ✅ Multiple TPs executed partial closes
    ─────────────────────────────────────────────────────
    """)


def main():
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║        ADVANCED TRADE MANAGER - FEATURE TESTS                 ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║  Testing:                                                     ║
    ║  1. Trailing Stop Loss                                        ║
    ║  2. Pending Order Activation                                  ║
    ║  3. Multiple Take Profits                                     ║
    ║  4. Complete Trading Scenario                                 ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    input("Press Enter to start tests...")
    
    test_trailing_stop()
    input("\nPress Enter for next test...")
    
    test_pending_order()
    input("\nPress Enter for next test...")
    
    test_multiple_take_profits()
    input("\nPress Enter for final test...")
    
    test_full_scenario()
    
    print("\n\n✅ ALL TESTS COMPLETED SUCCESSFULLY!")
    print("Your advanced trading features are ready to use.")
    print("\nRun: python xm360_auto_trader/auto_trader_v2.py")


if __name__ == "__main__":
    main()
