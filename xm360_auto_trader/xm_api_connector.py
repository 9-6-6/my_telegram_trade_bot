"""
XM360 Web API Connector (Alternative to MT5 Desktop)

This module attempts to connect to XM360 via web-based methods.
Note: XM doesn't officially offer a public REST API, so this uses
alternative approaches.

Options:
1. MT5 WebSocket API (if available)
2. Screen scraping (not recommended)
3. Third-party services

For now, this provides a simulation mode for testing without MT5.
"""

import logging
import json
import os
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List
import random

from . import config

logger = logging.getLogger('XMAPIConnector')


class XMWebConnector:
    """
    Alternative connector for XM360 that doesn't require MT5 desktop.
    
    Currently operates in SIMULATION mode for testing.
    Can be extended to use:
    - XM's WebSocket API (if they provide one)
    - Third-party trading APIs
    - cTrader API (if XM supports it)
    """
    
    def __init__(self, simulation_mode: bool = True):
        """
        Initialize the XM Web Connector.
        
        Args:
            simulation_mode: If True, simulates trades without real execution
        """
        self.simulation_mode = simulation_mode
        self.connected = False
        self.account_info = None
        self.simulated_positions = []
        self.simulated_balance = 10000.0  # Starting balance for simulation
        self.position_counter = 1000
        
        # Simulated prices (will be updated with real prices if possible)
        self.current_prices = {
            'XAUUSD': {'bid': 2650.00, 'ask': 2650.50},
            'XAGUSD': {'bid': 31.50, 'ask': 31.55},
            'EURUSD': {'bid': 1.0850, 'ask': 1.0852},
            'GBPUSD': {'bid': 1.2700, 'ask': 1.2702},
            'USDJPY': {'bid': 156.50, 'ask': 156.52},
            'BTCUSD': {'bid': 89000.00, 'ask': 89050.00},
        }
        
        logger.info(f"XM Web Connector initialized (Simulation Mode: {simulation_mode})")
    
    def connect(self) -> bool:
        """
        Connect to XM360.
        
        In simulation mode, this always succeeds.
        """
        if self.simulation_mode:
            self.connected = True
            self.account_info = {
                'login': config.ACCOUNT,
                'server': config.SERVER,
                'balance': self.simulated_balance,
                'equity': self.simulated_balance,
                'margin': 0.0,
                'margin_free': self.simulated_balance,
                'profit': 0.0,
                'leverage': 500,
                'currency': 'USD',
                'trade_mode': 'DEMO (Simulated)'
            }
            
            logger.info("=" * 50)
            logger.info("✅ Connected to XM360 (SIMULATION MODE)")
            logger.info(f"   Account: {config.ACCOUNT}")
            logger.info(f"   Balance: ${self.simulated_balance:.2f}")
            logger.info("   ⚠️ Trades are SIMULATED - not real!")
            logger.info("=" * 50)
            
            return True
        else:
            logger.error("Real XM API connection not yet implemented")
            logger.error("Please install MT5 desktop or use simulation mode")
            return False
    
    def disconnect(self):
        """Disconnect from XM360."""
        self.connected = False
        logger.info("Disconnected from XM360")
    
    def is_connected(self) -> bool:
        """Check if connected."""
        return self.connected
    
    def get_account_info(self) -> Optional[Dict[str, Any]]:
        """Get account information."""
        if not self.connected:
            return None
        
        # Update equity based on open positions
        total_profit = sum(p.get('profit', 0) for p in self.simulated_positions)
        self.account_info['equity'] = self.simulated_balance + total_profit
        self.account_info['profit'] = total_profit
        
        return self.account_info
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get symbol information."""
        if not self.connected:
            return None
        
        mt5_symbol = config.SYMBOL_MAPPING.get(symbol, symbol)
        
        # Return simulated symbol info
        return {
            'name': mt5_symbol,
            'bid': self.current_prices.get(symbol, {}).get('bid', 0),
            'ask': self.current_prices.get(symbol, {}).get('ask', 0),
            'spread': 5,
            'digits': 5 if 'JPY' not in symbol else 3,
            'volume_min': 0.01,
            'volume_max': 100.0,
            'volume_step': 0.01,
            'trade_mode': 1,  # Trading allowed
            'point': 0.00001 if 'JPY' not in symbol else 0.001
        }
    
    def get_current_price(self, symbol: str) -> Optional[Tuple[float, float]]:
        """Get current bid/ask price."""
        if symbol in self.current_prices:
            return (
                self.current_prices[symbol]['bid'],
                self.current_prices[symbol]['ask']
            )
        return None
    
    def update_price(self, symbol: str, bid: float, ask: float):
        """Update price for a symbol (used to sync with real prices)."""
        self.current_prices[symbol] = {'bid': bid, 'ask': ask}
    
    def place_order(
        self,
        symbol: str,
        order_type: str,
        lot_size: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        comment: str = "TelegramBot"
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Place a market order (SIMULATED).
        
        Args:
            symbol: Trading symbol
            order_type: 'BUY' or 'SELL'
            lot_size: Position size
            stop_loss: Stop loss price
            take_profit: Take profit price
            comment: Order comment
            
        Returns:
            tuple: (success, result)
        """
        if not self.connected:
            return False, {'error': 'Not connected'}
        
        if not self.simulation_mode:
            return False, {'error': 'Real trading not implemented'}
        
        # Get current price
        prices = self.get_current_price(symbol)
        if not prices:
            return False, {'error': f'No price for {symbol}'}
        
        bid, ask = prices
        
        if order_type.upper() == 'BUY':
            price = ask
        else:
            price = bid
        
        # Create simulated position
        self.position_counter += 1
        position = {
            'ticket': self.position_counter,
            'symbol': config.SYMBOL_MAPPING.get(symbol, symbol),
            'original_symbol': symbol,
            'type': order_type.upper(),
            'volume': lot_size,
            'open_price': price,
            'current_price': price,
            'sl': stop_loss,
            'tp': take_profit,
            'profit': 0.0,
            'swap': 0.0,
            'time': datetime.now().isoformat(),
            'comment': comment
        }
        
        self.simulated_positions.append(position)
        
        # Update margin
        margin_required = lot_size * 1000  # Simplified margin calculation
        self.account_info['margin'] += margin_required
        self.account_info['margin_free'] = self.simulated_balance - self.account_info['margin']
        
        logger.info(f"📈 SIMULATED ORDER PLACED:")
        logger.info(f"   Ticket: #{position['ticket']}")
        logger.info(f"   {symbol} {order_type} {lot_size} lots @ {price}")
        logger.info(f"   SL: {stop_loss}, TP: {take_profit}")
        
        return True, {
            'ticket': position['ticket'],
            'symbol': symbol,
            'type': order_type,
            'volume': lot_size,
            'price': price,
            'sl': stop_loss,
            'tp': take_profit,
            'comment': comment,
            'time': datetime.now().isoformat(),
            'simulated': True
        }
    
    def close_position(self, ticket: int) -> Tuple[bool, Dict[str, Any]]:
        """Close a simulated position."""
        for i, pos in enumerate(self.simulated_positions):
            if pos['ticket'] == ticket:
                # Get closing price
                prices = self.get_current_price(pos['original_symbol'])
                if prices:
                    bid, ask = prices
                    close_price = bid if pos['type'] == 'BUY' else ask
                    
                    # Calculate profit (simplified)
                    if pos['type'] == 'BUY':
                        profit = (close_price - pos['open_price']) * pos['volume'] * 100
                    else:
                        profit = (pos['open_price'] - close_price) * pos['volume'] * 100
                    
                    # Update balance
                    self.simulated_balance += profit
                    
                    # Remove position
                    closed = self.simulated_positions.pop(i)
                    
                    logger.info(f"📉 SIMULATED POSITION CLOSED:")
                    logger.info(f"   Ticket: #{ticket}")
                    logger.info(f"   Profit: ${profit:.2f}")
                    
                    return True, {
                        'ticket': ticket,
                        'closed_at': close_price,
                        'profit': profit,
                        'time': datetime.now().isoformat()
                    }
        
        return False, {'error': f'Position not found: {ticket}'}
    
    def get_open_positions(self) -> List[Dict]:
        """Get all open simulated positions."""
        # Update current prices and profits
        for pos in self.simulated_positions:
            prices = self.get_current_price(pos['original_symbol'])
            if prices:
                bid, ask = prices
                pos['current_price'] = bid if pos['type'] == 'BUY' else ask
                
                # Calculate profit
                if pos['type'] == 'BUY':
                    pos['profit'] = (pos['current_price'] - pos['open_price']) * pos['volume'] * 100
                else:
                    pos['profit'] = (pos['open_price'] - pos['current_price']) * pos['volume'] * 100
        
        return self.simulated_positions
    
    def get_todays_trades(self) -> List[Dict]:
        """Get today's simulated trades."""
        return []  # Would need trade history tracking


# Export a function to get the appropriate connector
def get_connector(prefer_simulation: bool = False):
    """
    Get the appropriate connector based on availability.
    
    Args:
        prefer_simulation: If True, use simulation even if MT5 is available
        
    Returns:
        Connector instance (MT5Connector or XMWebConnector)
    """
    if prefer_simulation:
        return XMWebConnector(simulation_mode=True)
    
    # Try to import and use MT5 first
    try:
        import MetaTrader5 as mt5
        if mt5.initialize():
            mt5.shutdown()
            from .mt5_connector import MT5Connector
            logger.info("Using MT5 Desktop Connector")
            return MT5Connector()
    except:
        pass
    
    # Fall back to simulation
    logger.warning("MT5 not available, using Simulation Mode")
    return XMWebConnector(simulation_mode=True)
