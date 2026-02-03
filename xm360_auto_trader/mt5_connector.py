"""
MetaTrader 5 Connector for XM360

Handles connection to MetaTrader 5 terminal and provides
interface for trade operations.
"""

import logging
from datetime import datetime
from typing import Optional, Dict, Any, Tuple

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    print("⚠️ MetaTrader5 library not installed. Run: pip install MetaTrader5")

from . import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('MT5Connector')


class MT5Connector:
    """
    MetaTrader 5 connection handler for XM360 broker.
    """
    
    def __init__(self):
        self.connected = False
        self.account_info = None
        self.last_error = None
        
        if not MT5_AVAILABLE:
            logger.error("MetaTrader5 library not available")
    
    def connect(self) -> bool:
        """
        Initialize connection to MetaTrader 5 terminal.
        
        Returns:
            bool: True if connection successful
        """
        if not MT5_AVAILABLE:
            self.last_error = "MetaTrader5 library not installed"
            return False
        
        # Check configuration
        if not config.ACCOUNT or not config.PASSWORD:
            self.last_error = "Account credentials not configured in config.py"
            logger.error(self.last_error)
            return False
        
        try:
            # Initialize MT5
            init_params = {}
            if config.MT5_PATH:
                init_params['path'] = config.MT5_PATH
            
            if not mt5.initialize(**init_params):
                self.last_error = f"MT5 initialization failed: {mt5.last_error()}"
                logger.error(self.last_error)
                return False
            
            logger.info("MT5 initialized successfully")
            
            # Login to account
            login_result = mt5.login(
                login=config.ACCOUNT,
                password=config.PASSWORD,
                server=config.SERVER
            )
            
            if not login_result:
                self.last_error = f"Login failed: {mt5.last_error()}"
                logger.error(self.last_error)
                mt5.shutdown()
                return False
            
            # Get account info
            self.account_info = mt5.account_info()
            if self.account_info is None:
                self.last_error = "Failed to get account info"
                logger.error(self.last_error)
                mt5.shutdown()
                return False
            
            self.connected = True
            logger.info(f"✅ Connected to MT5 Account: {self.account_info.login}")
            logger.info(f"   Server: {self.account_info.server}")
            logger.info(f"   Balance: ${self.account_info.balance:.2f}")
            logger.info(f"   Account Type: {'DEMO' if self.account_info.trade_mode == 0 else 'LIVE'}")
            
            # Safety check for demo mode
            if config.USE_DEMO and self.account_info.trade_mode != 0:
                logger.warning("⚠️ USE_DEMO is True but connected to LIVE account!")
                logger.warning("⚠️ Auto-trading disabled for safety. Set USE_DEMO=False to enable LIVE trading.")
                self.connected = False
                mt5.shutdown()
                return False
            
            return True
            
        except Exception as e:
            self.last_error = f"Connection error: {str(e)}"
            logger.error(self.last_error)
            return False
    
    def disconnect(self):
        """Disconnect from MetaTrader 5."""
        if MT5_AVAILABLE:
            mt5.shutdown()
        self.connected = False
        logger.info("Disconnected from MT5")
    
    def is_connected(self) -> bool:
        """Check if still connected to MT5."""
        if not MT5_AVAILABLE or not self.connected:
            return False
        
        # Verify connection by getting account info
        info = mt5.account_info()
        if info is None:
            self.connected = False
            return False
        
        return True
    
    def get_account_info(self) -> Optional[Dict[str, Any]]:
        """
        Get current account information.
        
        Returns:
            dict: Account info or None if not connected
        """
        if not self.is_connected():
            return None
        
        info = mt5.account_info()
        if info is None:
            return None
        
        return {
            'login': info.login,
            'server': info.server,
            'balance': info.balance,
            'equity': info.equity,
            'margin': info.margin,
            'free_margin': info.margin_free,
            'profit': info.profit,
            'leverage': info.leverage,
            'currency': info.currency,
            'trade_mode': 'DEMO' if info.trade_mode == 0 else 'LIVE'
        }
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a trading symbol.
        
        Args:
            symbol: Symbol name (e.g., 'GOLD', 'EURUSD')
            
        Returns:
            dict: Symbol info or None if not found
        """
        if not self.is_connected():
            return None
        
        # Map symbol if needed
        mt5_symbol = config.SYMBOL_MAPPING.get(symbol, symbol)
        
        info = mt5.symbol_info(mt5_symbol)
        if info is None:
            logger.warning(f"Symbol not found: {mt5_symbol}")
            return None
        
        # Enable symbol for trading if not visible
        if not info.visible:
            if not mt5.symbol_select(mt5_symbol, True):
                logger.warning(f"Failed to enable symbol: {mt5_symbol}")
                return None
        
        return {
            'name': info.name,
            'bid': info.bid,
            'ask': info.ask,
            'spread': info.spread,
            'digits': info.digits,
            'volume_min': info.volume_min,
            'volume_max': info.volume_max,
            'volume_step': info.volume_step,
            'trade_mode': info.trade_mode,
            'point': info.point
        }
    
    def get_current_price(self, symbol: str) -> Optional[Tuple[float, float]]:
        """
        Get current bid/ask price for a symbol.
        
        Args:
            symbol: Symbol name
            
        Returns:
            tuple: (bid, ask) or None if not available
        """
        info = self.get_symbol_info(symbol)
        if info:
            return (info['bid'], info['ask'])
        return None
    
    def place_order(
        self,
        symbol: str,
        order_type: str,  # 'BUY' or 'SELL'
        lot_size: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        comment: str = "TelegramBot"
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Place a market order.
        
        Args:
            symbol: Trading symbol
            order_type: 'BUY' or 'SELL'
            lot_size: Position size in lots
            stop_loss: Stop loss price (optional)
            take_profit: Take profit price (optional)
            comment: Order comment
            
        Returns:
            tuple: (success, result_dict)
        """
        if not self.is_connected():
            return False, {'error': 'Not connected to MT5'}
        
        # Map symbol
        mt5_symbol = config.SYMBOL_MAPPING.get(symbol, symbol)
        
        # Get symbol info
        symbol_info = mt5.symbol_info(mt5_symbol)
        if symbol_info is None:
            return False, {'error': f'Symbol not found: {mt5_symbol}'}
        
        # Enable symbol
        if not symbol_info.visible:
            if not mt5.symbol_select(mt5_symbol, True):
                return False, {'error': f'Failed to enable symbol: {mt5_symbol}'}
        
        # Determine order type
        if order_type.upper() == 'BUY':
            mt5_order_type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(mt5_symbol).ask
        elif order_type.upper() == 'SELL':
            mt5_order_type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(mt5_symbol).bid
        else:
            return False, {'error': f'Invalid order type: {order_type}'}
        
        # Validate lot size
        lot_size = max(symbol_info.volume_min, min(lot_size, symbol_info.volume_max))
        lot_size = round(lot_size / symbol_info.volume_step) * symbol_info.volume_step
        
        # Prepare order request
        request = {
            'action': mt5.TRADE_ACTION_DEAL,
            'symbol': mt5_symbol,
            'volume': lot_size,
            'type': mt5_order_type,
            'price': price,
            'deviation': config.MAX_SLIPPAGE,
            'magic': 123456,  # Unique identifier for our bot orders
            'comment': comment,
            'type_time': mt5.ORDER_TIME_GTC,
            'type_filling': mt5.ORDER_FILLING_IOC,
        }
        
        # Add SL/TP if provided
        if stop_loss is not None:
            request['sl'] = stop_loss
        if take_profit is not None:
            request['tp'] = take_profit
        
        # Send order
        result = mt5.order_send(request)
        
        if result is None:
            return False, {'error': f'Order send failed: {mt5.last_error()}'}
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return False, {
                'error': f'Order rejected',
                'retcode': result.retcode,
                'comment': result.comment
            }
        
        # Success
        return True, {
            'ticket': result.order,
            'symbol': mt5_symbol,
            'type': order_type,
            'volume': result.volume,
            'price': result.price,
            'sl': stop_loss,
            'tp': take_profit,
            'comment': comment,
            'time': datetime.now().isoformat()
        }
    
    def close_position(self, ticket: int) -> Tuple[bool, Dict[str, Any]]:
        """
        Close an open position by ticket number.
        
        Args:
            ticket: Position ticket number
            
        Returns:
            tuple: (success, result_dict)
        """
        if not self.is_connected():
            return False, {'error': 'Not connected to MT5'}
        
        # Get position info
        position = mt5.positions_get(ticket=ticket)
        if not position:
            return False, {'error': f'Position not found: {ticket}'}
        
        position = position[0]
        
        # Determine close order type
        if position.type == mt5.ORDER_TYPE_BUY:
            close_type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(position.symbol).bid
        else:
            close_type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(position.symbol).ask
        
        # Prepare close request
        request = {
            'action': mt5.TRADE_ACTION_DEAL,
            'symbol': position.symbol,
            'volume': position.volume,
            'type': close_type,
            'position': ticket,
            'price': price,
            'deviation': config.MAX_SLIPPAGE,
            'magic': 123456,
            'comment': 'Close by TelegramBot',
            'type_time': mt5.ORDER_TIME_GTC,
            'type_filling': mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        
        if result is None:
            return False, {'error': f'Close order failed: {mt5.last_error()}'}
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return False, {
                'error': 'Close rejected',
                'retcode': result.retcode,
                'comment': result.comment
            }
        
        return True, {
            'ticket': ticket,
            'closed_at': result.price,
            'profit': position.profit,
            'time': datetime.now().isoformat()
        }
    
    def get_open_positions(self) -> list:
        """
        Get all open positions.
        
        Returns:
            list: List of open positions
        """
        if not self.is_connected():
            return []
        
        positions = mt5.positions_get()
        if positions is None:
            return []
        
        return [{
            'ticket': p.ticket,
            'symbol': p.symbol,
            'type': 'BUY' if p.type == mt5.ORDER_TYPE_BUY else 'SELL',
            'volume': p.volume,
            'open_price': p.price_open,
            'current_price': p.price_current,
            'sl': p.sl,
            'tp': p.tp,
            'profit': p.profit,
            'swap': p.swap,
            'time': datetime.fromtimestamp(p.time).isoformat()
        } for p in positions]
    
    def get_todays_trades(self) -> list:
        """
        Get all trades executed today.
        
        Returns:
            list: List of today's trades
        """
        if not self.is_connected():
            return []
        
        from datetime import datetime, timedelta
        
        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        deals = mt5.history_deals_get(today_start, datetime.now())
        if deals is None:
            return []
        
        return [{
            'ticket': d.ticket,
            'symbol': d.symbol,
            'type': 'BUY' if d.type == 0 else 'SELL',
            'volume': d.volume,
            'price': d.price,
            'profit': d.profit,
            'commission': d.commission,
            'swap': d.swap,
            'time': datetime.fromtimestamp(d.time).isoformat()
        } for d in deals]
