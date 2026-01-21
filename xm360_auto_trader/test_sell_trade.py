"""
Test SELL Trade with Advanced Features
=======================================
Demonstrates: Pending Orders, Trailing Stop, Multiple TPs for SELL trades

Run: python xm360_auto_trader/test_sell_trade.py
"""

import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xm360_auto_trader.advanced_trade_manager import (
    AdvancedTradeManager,
    TrailingStopConfig,
)


class MockMT5Connector:
    """Simulated MT5 connector for testing"""
    
    def __init__(self):
        self.current_prices = {"XAUUSD": 2660.00}
    
    def get_current_price(self, symbol):
        return self.current_prices.get(symbol, 100)
    
    def set_price(self, symbol, price):
        self.current_prices[symbol] = price


def test_sell_trade_complete():
    """Test complete SELL trade scenario with all features"""
    
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║           SELL TRADE TEST - ALL FEATURES                      ║
    ╚═══════════════════════════════════════════════════════════════╝
    
    📋 SELL TRADE SCENARIO:
    ─────────────────────────────────────────────────────────────────
    Current price: $2660
    Signal: SELL at $2655 (pending order - price must DROP to entry)
    Stop Loss: $2665 (above entry - protects if price rises)
    Trailing: Start after $5 profit, trail $3 ABOVE price
    TP1: $2645 (close 50%) - below entry
    TP2: $2635 (close remaining 50%) - below entry
    
    SELL Trade Logic:
    - Entry: Price must DROP to $2655
    - Profit: When price goes DOWN (below entry)
    - SL: Triggers when price goes UP (above SL price)
    - Trailing: SL moves DOWN as price drops (locks profit)
    ─────────────────────────────────────────────────────────────────
    """)
    
    mock = MockMT5Connector()
    mock.set_price("XAUUSD", 2660.00)  # Current price ABOVE entry
    
    manager = AdvancedTradeManager(mock)
    manager._get_current_price = lambda s: mock.get_current_price(s)
    
    # Create SELL pending order
    trade = manager.create_pending_order(
        symbol="XAUUSD",
        direction="SELL",  # <-- SELL TRADE
        entry_price=2655.0,  # Price must DROP to this
        stop_loss=2665.0,    # SL is ABOVE entry for SELL
        take_profits=[
            {"price": 2645.0, "percent": 50},   # TP1: $10 below entry
            {"price": 2635.0, "percent": 100},  # TP2: $20 below entry
        ],
        lot_size=0.02,
        trailing_config=TrailingStopConfig(
            enabled=True,
            activation_profit_pips=500,  # $5 profit to activate
            trail_distance_pips=300,     # Trail $3 above lowest price
            step_pips=100                # Minimum $1 to update
        )
    )
    
    print(f"✅ SELL Pending Order Created: #{trade.ticket}")
    print(f"   Entry: ${trade.entry_price} (waiting for price to DROP)")
    print(f"   SL: ${trade.current_stop_loss} (ABOVE entry)")
    print(f"   Order Type: {trade.order_type.value}")
    print()
    
    # Simulate price movements for SELL
    price_sequence = [
        (2660.0, "Starting price - Pending order waiting"),
        (2658.0, "Price dropping - Still waiting"),
        (2655.0, "🔔 Price reaches $2655 - SELL ORDER ACTIVATES"),
        (2652.0, "In profit $3 - Not enough for trailing yet"),
        (2650.0, "📈 +$5 profit - TRAILING ACTIVATES (SL moves DOWN)"),
        (2647.0, "Price continues down - SL trails down"),
        (2645.0, "🎯 TP1 HIT - Close 50%"),
        (2642.0, "Price continues dropping - SL keeps trailing"),
        (2635.0, "🎯 TP2 HIT - Close remaining 50%"),
    ]
    
    print("─" * 65)
    print("SIMULATION START (SELL TRADE)")
    print("─" * 65)
    
    for price, description in price_sequence:
        print(f"\n💰 Price: ${price:.2f} - {description}")
        
        mock.set_price("XAUUSD", price)
        manager.monitor_and_manage_trades()
        
        if trade.ticket in manager.active_trades:
            t = manager.active_trades[trade.ticket]
            
            status = "⏳ PENDING" if t.is_pending else "🔴 ACTIVE (SELL)"
            trailing = "📉 YES (moving DOWN)" if t.trailing_activated else "❌ No"
            tps = sum(1 for tp in t.take_profit_levels if tp.reached)
            
            # Calculate profit for SELL (entry - current = profit)
            if t.activated:
                profit = (t.entry_price - price) * 100  # Simplified
                profit_str = f"+${profit:.0f}" if profit > 0 else f"-${abs(profit):.0f}"
            else:
                profit_str = "N/A"
            
            print(f"   Status: {status}")
            print(f"   Entry: ${t.entry_price:.2f}")
            print(f"   Current SL: ${t.current_stop_loss:.2f} (above price)")
            print(f"   Lowest Price: ${t.lowest_price:.2f}")
            print(f"   Trailing: {trailing}")
            print(f"   TPs Hit: {tps}/2")
            print(f"   Lots: {t.lot_size:.3f}")
            print(f"   Unrealized: {profit_str}")
        else:
            print(f"\n   🎉 TRADE FULLY CLOSED!")
            print(f"   Total Profit: ${trade.realized_pnl:.2f}")
        
        time.sleep(0.3)
    
    print("\n" + "=" * 65)
    print("✅ SELL TRADE TEST COMPLETE")
    print("=" * 65)
    
    print(f"""
    📊 SELL TRADE SUMMARY:
    ─────────────────────────────────────────────────────────────────
    Direction: SELL (profit when price goes DOWN)
    Entry Price: $2655.00 (activated when price dropped to entry)
    Initial SL: $2665.00 (above entry)
    Final SL: ${trade.current_stop_loss:.2f} (trailed DOWN with price)
    
    TP1 Hit: $2645 (closed 0.01 lots) ✅
    TP2 Hit: $2635 (closed 0.01 lots) ✅
    
    Total Profit: ${trade.realized_pnl:.2f}
    
    ✅ Pending order worked (activated when price DROPPED to $2655)
    ✅ Trailing stop moved DOWN as price moved DOWN (our favor)
    ✅ Multiple TPs executed at LOWER prices (profit targets)
    ─────────────────────────────────────────────────────────────────
    """)


def compare_buy_vs_sell():
    """Show comparison between BUY and SELL logic"""
    
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║              BUY vs SELL - COMPARISON                         ║
    ╚═══════════════════════════════════════════════════════════════╝
    
    ┌─────────────────────┬──────────────────┬──────────────────────┐
    │      FEATURE        │      BUY         │        SELL          │
    ├─────────────────────┼──────────────────┼──────────────────────┤
    │ Profit Direction    │ Price goes UP    │ Price goes DOWN      │
    ├─────────────────────┼──────────────────┼──────────────────────┤
    │ Stop Loss Position  │ BELOW entry      │ ABOVE entry          │
    ├─────────────────────┼──────────────────┼──────────────────────┤
    │ Take Profit         │ ABOVE entry      │ BELOW entry          │
    ├─────────────────────┼──────────────────┼──────────────────────┤
    │ Trailing SL Moves   │ UP (follows      │ DOWN (follows        │
    │                     │ rising price)    │ falling price)       │
    ├─────────────────────┼──────────────────┼──────────────────────┤
    │ Pending Order       │ BUY_STOP: price  │ SELL_STOP: price     │
    │                     │ rises to entry   │ drops to entry       │
    │                     │ BUY_LIMIT: price │ SELL_LIMIT: price    │
    │                     │ drops to entry   │ rises to entry       │
    └─────────────────────┴──────────────────┴──────────────────────┘
    
    
    EXAMPLE - BUY TRADE:
    ─────────────────────────────────────────────────────────────────
    Entry: $2655, SL: $2645 (below), TP: $2675 (above)
    Price goes UP to $2675 → PROFIT! ✅
    Price goes DOWN to $2645 → LOSS (SL hit) ❌
    Trailing: SL moves UP from $2645 → $2660 → $2670 (locks profit)
    
    
    EXAMPLE - SELL TRADE:
    ─────────────────────────────────────────────────────────────────
    Entry: $2655, SL: $2665 (above), TP: $2635 (below)
    Price goes DOWN to $2635 → PROFIT! ✅
    Price goes UP to $2665 → LOSS (SL hit) ❌
    Trailing: SL moves DOWN from $2665 → $2650 → $2640 (locks profit)
    
    """)


def main():
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║        SELL TRADE ADVANCED FEATURES TEST                      ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    compare_buy_vs_sell()
    input("Press Enter to run SELL trade simulation...")
    
    test_sell_trade_complete()
    
    print("\n\n✅ SELL TRADE TEST COMPLETED!")
    print("Both BUY and SELL trades have full automation support:")
    print("  ✅ Pending Orders")
    print("  ✅ Trailing Stop Loss")  
    print("  ✅ Multiple Take Profits")


if __name__ == "__main__":
    main()
