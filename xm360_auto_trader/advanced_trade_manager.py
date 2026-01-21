"""
Advanced Trade Manager with Trailing Stop and Multiple Take Profits
====================================================================
Features:
- Pending orders (trade activates at specific price)
- Trailing Stop Loss (SL moves as price moves in your favor)
- Multiple Take Profit targets (partial closes)
- Full automation of trade lifecycle
"""

import time
import logging
from datetime import datetime
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types for pending orders"""
    MARKET = "market"           # Execute immediately
    BUY_LIMIT = "buy_limit"     # Buy when price drops to entry
    BUY_STOP = "buy_stop"       # Buy when price rises to entry
    SELL_LIMIT = "sell_limit"   # Sell when price rises to entry
    SELL_STOP = "sell_stop"     # Sell when price drops to entry


@dataclass
class TakeProfitLevel:
    """Individual take profit target"""
    price: float
    percent_to_close: float  # Percentage of position to close (e.g., 50%)
    reached: bool = False
    closed_at: Optional[datetime] = None


@dataclass
class TrailingStopConfig:
    """Configuration for trailing stop loss"""
    enabled: bool = True
    activation_profit_pips: float = 5.0    # Start trailing after X pips profit
    trail_distance_pips: float = 3.0       # Keep SL this many pips behind price
    step_pips: float = 1.0                 # Minimum movement to update SL


@dataclass
class AdvancedTrade:
    """
    Advanced trade with all automation features
    """
    # Basic trade info
    ticket: int
    symbol: str
    direction: str  # "BUY" or "SELL"
    entry_price: float
    lot_size: float
    
    # Stop Loss
    initial_stop_loss: float
    current_stop_loss: float
    
    # Take Profit levels
    take_profit_levels: List[TakeProfitLevel] = field(default_factory=list)
    
    # Trailing Stop
    trailing_stop: TrailingStopConfig = field(default_factory=TrailingStopConfig)
    trailing_activated: bool = False
    highest_price: float = 0.0  # For BUY trades
    lowest_price: float = 999999.0  # For SELL trades
    
    # Order type
    order_type: OrderType = OrderType.MARKET
    is_pending: bool = False
    activated: bool = False
    activation_time: Optional[datetime] = None
    
    # Status
    is_open: bool = True
    closed_at: Optional[datetime] = None
    close_reason: str = ""
    realized_pnl: float = 0.0
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)


class AdvancedTradeManager:
    """
    Manages trades with trailing stops, multiple TPs, and pending orders
    """
    
    def __init__(self, mt5_connector=None):
        self.connector = mt5_connector
        self.active_trades: Dict[int, AdvancedTrade] = {}
        self.closed_trades: List[AdvancedTrade] = []
        self.pip_values = {
            "XAUUSD": 0.01,    # Gold: 1 pip = $0.01
            "BTCUSD": 1.0,     # Bitcoin: 1 pip = $1
            "EURUSD": 0.0001,  # Forex: 1 pip = 0.0001
        }
        
    def get_pip_value(self, symbol: str) -> float:
        """Get pip value for symbol"""
        return self.pip_values.get(symbol, 0.01)
    
    def calculate_pips(self, symbol: str, price_diff: float) -> float:
        """Convert price difference to pips"""
        pip_value = self.get_pip_value(symbol)
        return abs(price_diff) / pip_value
    
    def pips_to_price(self, symbol: str, pips: float) -> float:
        """Convert pips to price difference"""
        pip_value = self.get_pip_value(symbol)
        return pips * pip_value

    def create_pending_order(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profits: List[Dict[str, float]],  # [{"price": 2660, "percent": 50}, ...]
        lot_size: float = 0.01,
        trailing_config: Optional[TrailingStopConfig] = None
    ) -> AdvancedTrade:
        """
        Create a pending order that activates when price reaches entry
        
        Args:
            symbol: Trading symbol (XAUUSD, BTCUSD, etc.)
            direction: "BUY" or "SELL"
            entry_price: Price at which to enter the trade
            stop_loss: Initial stop loss price
            take_profits: List of TP levels with price and percent to close
            lot_size: Position size
            trailing_config: Trailing stop configuration
        
        Example:
            create_pending_order(
                symbol="XAUUSD",
                direction="BUY",
                entry_price=4805.0,
                stop_loss=4790.0,
                take_profits=[
                    {"price": 4815.0, "percent": 50},  # Close 50% at first TP
                    {"price": 4825.0, "percent": 100}  # Close remaining at second TP
                ],
                trailing_config=TrailingStopConfig(
                    enabled=True,
                    activation_profit_pips=5,
                    trail_distance_pips=3
                )
            )
        """
        
        # Determine order type based on current price and direction
        current_price = self._get_current_price(symbol)
        
        if direction == "BUY":
            if entry_price > current_price:
                order_type = OrderType.BUY_STOP
            else:
                order_type = OrderType.BUY_LIMIT
        else:  # SELL
            if entry_price < current_price:
                order_type = OrderType.SELL_STOP
            else:
                order_type = OrderType.SELL_LIMIT
        
        # Create take profit levels
        tp_levels = [
            TakeProfitLevel(price=tp["price"], percent_to_close=tp["percent"])
            for tp in take_profits
        ]
        
        # Create the trade object
        trade = AdvancedTrade(
            ticket=self._generate_ticket(),
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            lot_size=lot_size,
            initial_stop_loss=stop_loss,
            current_stop_loss=stop_loss,
            take_profit_levels=tp_levels,
            trailing_stop=trailing_config or TrailingStopConfig(),
            order_type=order_type,
            is_pending=True,
            activated=False,
            highest_price=entry_price if direction == "BUY" else 0,
            lowest_price=entry_price if direction == "SELL" else 999999
        )
        
        # Store the trade
        self.active_trades[trade.ticket] = trade
        
        logger.info(f"""
╔══════════════════════════════════════════════════════════════╗
║  📋 PENDING ORDER CREATED                                    ║
╠══════════════════════════════════════════════════════════════╣
║  Ticket: {trade.ticket}
║  Symbol: {symbol}
║  Type: {order_type.value.upper()}
║  Direction: {direction}
║  Entry Price: ${entry_price:.2f}
║  Current Price: ${current_price:.2f}
║  Stop Loss: ${stop_loss:.2f}
║  Take Profits: {len(tp_levels)} levels
║  Trailing Stop: {'Enabled' if trade.trailing_stop.enabled else 'Disabled'}
╚══════════════════════════════════════════════════════════════╝
        """)
        
        return trade
    
    def create_market_order(
        self,
        symbol: str,
        direction: str,
        stop_loss: float,
        take_profits: List[Dict[str, float]],
        lot_size: float = 0.01,
        trailing_config: Optional[TrailingStopConfig] = None
    ) -> AdvancedTrade:
        """Create immediate market order with trailing stop and multiple TPs"""
        
        current_price = self._get_current_price(symbol)
        
        tp_levels = [
            TakeProfitLevel(price=tp["price"], percent_to_close=tp["percent"])
            for tp in take_profits
        ]
        
        trade = AdvancedTrade(
            ticket=self._generate_ticket(),
            symbol=symbol,
            direction=direction,
            entry_price=current_price,
            lot_size=lot_size,
            initial_stop_loss=stop_loss,
            current_stop_loss=stop_loss,
            take_profit_levels=tp_levels,
            trailing_stop=trailing_config or TrailingStopConfig(),
            order_type=OrderType.MARKET,
            is_pending=False,
            activated=True,
            activation_time=datetime.now(),
            highest_price=current_price if direction == "BUY" else 0,
            lowest_price=current_price if direction == "SELL" else 999999
        )
        
        # Place order on MT5
        if self.connector:
            result = self._place_mt5_order(trade)
            if result:
                trade.ticket = result.get("ticket", trade.ticket)
        
        self.active_trades[trade.ticket] = trade
        
        logger.info(f"✅ MARKET ORDER PLACED: {direction} {symbol} @ ${current_price:.2f}")
        
        return trade
    
    def monitor_and_manage_trades(self):
        """
        Main loop to monitor all trades and manage:
        - Pending order activation
        - Stop loss hits
        - Take profit hits (partial closes)
        - Trailing stop updates
        
        Call this method continuously in your main loop!
        """
        
        for ticket, trade in list(self.active_trades.items()):
            if not trade.is_open:
                continue
                
            current_price = self._get_current_price(trade.symbol)
            
            # 1. Check pending order activation
            if trade.is_pending and not trade.activated:
                if self._check_pending_activation(trade, current_price):
                    self._activate_pending_order(trade, current_price)
                continue
            
            # 2. Check stop loss hit
            if self._check_stop_loss_hit(trade, current_price):
                self._close_trade(trade, current_price, "STOP_LOSS")
                continue
            
            # 3. Check take profit hits (partial closes)
            self._check_take_profit_hits(trade, current_price)
            
            # 4. Update trailing stop
            if trade.trailing_stop.enabled:
                self._update_trailing_stop(trade, current_price)
            
            trade.last_updated = datetime.now()
    
    def _check_pending_activation(self, trade: AdvancedTrade, current_price: float) -> bool:
        """Check if pending order should be activated"""
        
        if trade.direction == "BUY":
            if trade.order_type == OrderType.BUY_STOP:
                # Activate when price rises to entry
                return current_price >= trade.entry_price
            elif trade.order_type == OrderType.BUY_LIMIT:
                # Activate when price drops to entry
                return current_price <= trade.entry_price
        else:  # SELL
            if trade.order_type == OrderType.SELL_STOP:
                # Activate when price drops to entry
                return current_price <= trade.entry_price
            elif trade.order_type == OrderType.SELL_LIMIT:
                # Activate when price rises to entry
                return current_price >= trade.entry_price
        
        return False
    
    def _activate_pending_order(self, trade: AdvancedTrade, current_price: float):
        """Activate a pending order"""
        
        trade.is_pending = False
        trade.activated = True
        trade.activation_time = datetime.now()
        trade.entry_price = current_price  # Use actual fill price
        
        # Place on MT5
        if self.connector:
            result = self._place_mt5_order(trade)
            if result:
                trade.ticket = result.get("ticket", trade.ticket)
        
        logger.info(f"""
╔══════════════════════════════════════════════════════════════╗
║  🟢 PENDING ORDER ACTIVATED!                                 ║
╠══════════════════════════════════════════════════════════════╣
║  Ticket: {trade.ticket}
║  Symbol: {trade.symbol}
║  Direction: {trade.direction}
║  Entry Price: ${current_price:.2f}
║  Stop Loss: ${trade.current_stop_loss:.2f}
║  Take Profits: {len(trade.take_profit_levels)} levels
╚══════════════════════════════════════════════════════════════╝
        """)
    
    def _check_stop_loss_hit(self, trade: AdvancedTrade, current_price: float) -> bool:
        """Check if stop loss is hit"""
        
        if trade.direction == "BUY":
            return current_price <= trade.current_stop_loss
        else:  # SELL
            return current_price >= trade.current_stop_loss
    
    def _check_take_profit_hits(self, trade: AdvancedTrade, current_price: float):
        """Check and execute take profit levels"""
        
        for tp in trade.take_profit_levels:
            if tp.reached:
                continue
            
            hit = False
            if trade.direction == "BUY":
                hit = current_price >= tp.price
            else:  # SELL
                hit = current_price <= tp.price
            
            if hit:
                tp.reached = True
                tp.closed_at = datetime.now()
                
                # Calculate lots to close
                lots_to_close = trade.lot_size * (tp.percent_to_close / 100)
                
                # Close partial position
                self._close_partial(trade, lots_to_close, current_price, f"TP @ ${tp.price:.2f}")
                
                # Update remaining lot size
                trade.lot_size -= lots_to_close
                
                logger.info(f"""
╔══════════════════════════════════════════════════════════════╗
║  🎯 TAKE PROFIT HIT!                                         ║
╠══════════════════════════════════════════════════════════════╣
║  Ticket: {trade.ticket}
║  TP Price: ${tp.price:.2f}
║  Closed: {tp.percent_to_close}% ({lots_to_close:.2f} lots)
║  Remaining: {trade.lot_size:.2f} lots
╚══════════════════════════════════════════════════════════════╝
                """)
                
                # Check if fully closed
                if trade.lot_size <= 0.001:
                    self._close_trade(trade, current_price, "ALL_TPS_HIT")
    
    def _update_trailing_stop(self, trade: AdvancedTrade, current_price: float):
        """Update trailing stop loss as price moves favorably"""
        
        ts = trade.trailing_stop
        pip_value = self.get_pip_value(trade.symbol)
        
        if trade.direction == "BUY":
            # Track highest price
            if current_price > trade.highest_price:
                trade.highest_price = current_price
            
            # Calculate profit in pips
            profit_pips = (current_price - trade.entry_price) / pip_value
            
            # Check if trailing should be activated
            if profit_pips >= ts.activation_profit_pips:
                if not trade.trailing_activated:
                    trade.trailing_activated = True
                    logger.info(f"🔄 Trailing stop ACTIVATED for ticket {trade.ticket}")
                
                # Calculate new stop loss (trail behind highest price)
                trail_distance = ts.trail_distance_pips * pip_value
                new_sl = trade.highest_price - trail_distance
                
                # Only move SL up (never down)
                if new_sl > trade.current_stop_loss:
                    # Check minimum step
                    sl_change_pips = (new_sl - trade.current_stop_loss) / pip_value
                    if sl_change_pips >= ts.step_pips:
                        old_sl = trade.current_stop_loss
                        trade.current_stop_loss = new_sl
                        
                        # Update on MT5
                        if self.connector:
                            self._modify_sl_on_mt5(trade)
                        
                        logger.info(f"""
╔══════════════════════════════════════════════════════════════╗
║  📈 TRAILING STOP UPDATED (BUY)                              ║
╠══════════════════════════════════════════════════════════════╣
║  Ticket: {trade.ticket}
║  Current Price: ${current_price:.2f}
║  Highest Price: ${trade.highest_price:.2f}
║  Old SL: ${old_sl:.2f}
║  New SL: ${new_sl:.2f} (+${new_sl - old_sl:.2f})
║  Profit Locked: ${new_sl - trade.entry_price:.2f}
╚══════════════════════════════════════════════════════════════╝
                        """)
        
        else:  # SELL
            # Track lowest price
            if current_price < trade.lowest_price:
                trade.lowest_price = current_price
            
            # Calculate profit in pips
            profit_pips = (trade.entry_price - current_price) / pip_value
            
            # Check if trailing should be activated
            if profit_pips >= ts.activation_profit_pips:
                if not trade.trailing_activated:
                    trade.trailing_activated = True
                    logger.info(f"🔄 Trailing stop ACTIVATED for ticket {trade.ticket}")
                
                # Calculate new stop loss (trail above lowest price)
                trail_distance = ts.trail_distance_pips * pip_value
                new_sl = trade.lowest_price + trail_distance
                
                # Only move SL down (never up for SELL)
                if new_sl < trade.current_stop_loss:
                    sl_change_pips = (trade.current_stop_loss - new_sl) / pip_value
                    if sl_change_pips >= ts.step_pips:
                        old_sl = trade.current_stop_loss
                        trade.current_stop_loss = new_sl
                        
                        if self.connector:
                            self._modify_sl_on_mt5(trade)
                        
                        logger.info(f"""
╔══════════════════════════════════════════════════════════════╗
║  📉 TRAILING STOP UPDATED (SELL)                             ║
╠══════════════════════════════════════════════════════════════╣
║  Ticket: {trade.ticket}
║  Current Price: ${current_price:.2f}
║  Lowest Price: ${trade.lowest_price:.2f}
║  Old SL: ${old_sl:.2f}
║  New SL: ${new_sl:.2f} (-${old_sl - new_sl:.2f})
║  Profit Locked: ${trade.entry_price - new_sl:.2f}
╚══════════════════════════════════════════════════════════════╝
                        """)
    
    def _close_trade(self, trade: AdvancedTrade, price: float, reason: str):
        """Close entire trade"""
        
        trade.is_open = False
        trade.closed_at = datetime.now()
        trade.close_reason = reason
        
        # Calculate P&L
        if trade.direction == "BUY":
            trade.realized_pnl = (price - trade.entry_price) * trade.lot_size * 100
        else:
            trade.realized_pnl = (trade.entry_price - price) * trade.lot_size * 100
        
        # Close on MT5
        if self.connector:
            self._close_on_mt5(trade)
        
        # Move to closed trades
        self.closed_trades.append(trade)
        del self.active_trades[trade.ticket]
        
        emoji = "🟢" if trade.realized_pnl >= 0 else "🔴"
        
        logger.info(f"""
╔══════════════════════════════════════════════════════════════╗
║  {emoji} TRADE CLOSED                                          ║
╠══════════════════════════════════════════════════════════════╣
║  Ticket: {trade.ticket}
║  Symbol: {trade.symbol}
║  Direction: {trade.direction}
║  Entry: ${trade.entry_price:.2f}
║  Exit: ${price:.2f}
║  Reason: {reason}
║  P&L: ${trade.realized_pnl:.2f}
╚══════════════════════════════════════════════════════════════╝
        """)
    
    def _close_partial(self, trade: AdvancedTrade, lots: float, price: float, reason: str):
        """Close partial position"""
        if self.connector:
            self._partial_close_on_mt5(trade, lots)
        
        # Calculate partial P&L
        if trade.direction == "BUY":
            pnl = (price - trade.entry_price) * lots * 100
        else:
            pnl = (trade.entry_price - price) * lots * 100
        
        trade.realized_pnl += pnl
    
    # ============ MT5 Integration Methods ============
    
    def _place_mt5_order(self, trade: AdvancedTrade) -> Optional[Dict]:
        """Place order on MT5"""
        if not self.connector:
            return {"ticket": trade.ticket}
        
        try:
            # Get the first TP for MT5 (MT5 only supports single TP)
            first_tp = trade.take_profit_levels[0].price if trade.take_profit_levels else 0
            
            result = self.connector.place_order(
                symbol=trade.symbol,
                direction=trade.direction,
                lot_size=trade.lot_size,
                stop_loss=trade.current_stop_loss,
                take_profit=first_tp
            )
            return result
        except Exception as e:
            logger.error(f"MT5 order failed: {e}")
            return None
    
    def _modify_sl_on_mt5(self, trade: AdvancedTrade):
        """Modify stop loss on MT5"""
        if not self.connector:
            return
        
        try:
            self.connector.modify_position(
                ticket=trade.ticket,
                stop_loss=trade.current_stop_loss
            )
        except Exception as e:
            logger.error(f"MT5 SL modification failed: {e}")
    
    def _close_on_mt5(self, trade: AdvancedTrade):
        """Close position on MT5"""
        if not self.connector:
            return
        
        try:
            self.connector.close_position(trade.ticket)
        except Exception as e:
            logger.error(f"MT5 close failed: {e}")
    
    def _partial_close_on_mt5(self, trade: AdvancedTrade, lots: float):
        """Partial close on MT5"""
        if not self.connector:
            return
        
        try:
            self.connector.close_position(trade.ticket, lots)
        except Exception as e:
            logger.error(f"MT5 partial close failed: {e}")
    
    def _get_current_price(self, symbol: str) -> float:
        """Get current price for symbol"""
        if self.connector:
            try:
                return self.connector.get_current_price(symbol)
            except:
                pass
        
        # Fallback simulation prices
        prices = {
            "XAUUSD": 2650.50,
            "BTCUSD": 101250.00,
            "EURUSD": 1.0850
        }
        return prices.get(symbol, 100.0)
    
    def _generate_ticket(self) -> int:
        """Generate unique ticket number"""
        return int(time.time() * 1000) % 1000000000
    
    # ============ Status Methods ============
    
    def get_active_trades_summary(self) -> str:
        """Get summary of all active trades"""
        if not self.active_trades:
            return "No active trades"
        
        lines = ["═" * 60, "ACTIVE TRADES SUMMARY", "═" * 60]
        
        for ticket, trade in self.active_trades.items():
            current_price = self._get_current_price(trade.symbol)
            
            if trade.direction == "BUY":
                pnl = (current_price - trade.entry_price) * trade.lot_size * 100
            else:
                pnl = (trade.entry_price - current_price) * trade.lot_size * 100
            
            status = "PENDING" if trade.is_pending else "ACTIVE"
            emoji = "🟢" if pnl >= 0 else "🔴"
            
            lines.append(f"""
Ticket: {ticket} [{status}]
{trade.symbol} {trade.direction} @ ${trade.entry_price:.2f}
Current: ${current_price:.2f} | SL: ${trade.current_stop_loss:.2f}
Lots: {trade.lot_size:.2f} | {emoji} P&L: ${pnl:.2f}
Trailing: {'Active' if trade.trailing_activated else 'Waiting'}
            """)
        
        lines.append("═" * 60)
        return "\n".join(lines)


# ============ Example Usage ============

def example_usage():
    """Demonstrate the advanced trade manager"""
    
    print("\n" + "=" * 70)
    print("ADVANCED TRADE MANAGER - DEMONSTRATION")
    print("=" * 70)
    
    # Create manager
    manager = AdvancedTradeManager()
    
    # Example: User's scenario
    # XAUUSD at $4800, signal says BUY at $4805
    # SL at $4790
    # TP1 at $4815 (close 50%)
    # TP2 at $4825 (close remaining)
    # Trailing stop: activate after 5 pips profit, trail 3 pips behind
    
    print("\n📊 Scenario: XAUUSD currently at $2650")
    print("📋 Signal: BUY at $2655 (pending order)")
    print("🛑 Stop Loss: $2645")
    print("🎯 TP1: $2665 (close 50%)")
    print("🎯 TP2: $2675 (close remaining 50%)")
    print("📈 Trailing: Activate after +$5, trail $3 behind")
    
    # Create pending order
    trade = manager.create_pending_order(
        symbol="XAUUSD",
        direction="BUY",
        entry_price=2655.0,
        stop_loss=2645.0,
        take_profits=[
            {"price": 2665.0, "percent": 50},   # Close 50% at TP1
            {"price": 2675.0, "percent": 100},  # Close remaining at TP2
        ],
        lot_size=0.02,
        trailing_config=TrailingStopConfig(
            enabled=True,
            activation_profit_pips=500,  # $5 for gold (pip = $0.01)
            trail_distance_pips=300,     # $3 trail
            step_pips=100                # $1 minimum step
        )
    )
    
    print(f"\n✅ Pending order created: Ticket #{trade.ticket}")
    print(f"   Waiting for price to reach ${trade.entry_price}...")
    
    # Simulate price movements
    print("\n" + "-" * 50)
    print("SIMULATING PRICE MOVEMENTS...")
    print("-" * 50)
    
    # This would be called in your main loop
    # manager.monitor_and_manage_trades()
    
    print("\n📝 In real usage, call manager.monitor_and_manage_trades() continuously")
    print("   to handle pending activation, TP hits, and trailing stop updates.")


if __name__ == "__main__":
    example_usage()
