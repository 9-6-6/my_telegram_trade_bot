"""
Trade Manager Module

Handles trade execution and management:
- Execute trades from signals
- Manage open positions
- Track trade history
"""

import json
import os
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

from . import config
from .mt5_connector import MT5Connector
from .risk_manager import RiskManager

logger = logging.getLogger('TradeManager')


class TradeManager:
    """
    Manages trade execution and tracking.
    """
    
    def __init__(self, mt5_connector: MT5Connector):
        """
        Initialize trade manager.
        
        Args:
            mt5_connector: MT5Connector instance
        """
        self.mt5 = mt5_connector
        self.risk_manager = RiskManager(mt5_connector)
        self.trade_history = []
        self._load_history()
    
    def _load_history(self):
        """Load trade history from file."""
        try:
            if os.path.exists(config.TRADE_HISTORY_FILE):
                with open(config.TRADE_HISTORY_FILE, 'r') as f:
                    self.trade_history = json.load(f)
        except Exception as e:
            logger.error(f"Error loading trade history: {e}")
            self.trade_history = []
    
    def _save_history(self):
        """Save trade history to file."""
        try:
            os.makedirs(os.path.dirname(config.TRADE_HISTORY_FILE), exist_ok=True)
            with open(config.TRADE_HISTORY_FILE, 'w') as f:
                json.dump(self.trade_history, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving trade history: {e}")
    
    def execute_signal(self, signal: Dict) -> Tuple[bool, Dict]:
        """
        Execute a trading signal.
        
        Args:
            signal: Trading signal dictionary
            
        Returns:
            tuple: (success, result)
        """
        logger.info(f"📈 Executing signal: {signal['symbol']} {signal['direction']}")
        
        # Validate trade
        is_valid, message, adjusted_signal = self.risk_manager.validate_trade(signal)
        
        if not is_valid:
            logger.warning(f"❌ Trade rejected: {message}")
            return False, {'error': message, 'signal': signal}
        
        # Execute trade
        success, result = self.mt5.place_order(
            symbol=adjusted_signal['symbol'],
            order_type=adjusted_signal['direction'],
            lot_size=adjusted_signal['lot_size'],
            stop_loss=adjusted_signal.get('stop_loss'),
            take_profit=adjusted_signal.get('take_profit'),
            comment=f"Signal: {adjusted_signal.get('source', 'TelegramBot')}"
        )
        
        if success:
            logger.info(f"✅ Trade executed: Ticket #{result['ticket']}")
            
            # Record in history
            trade_record = {
                'timestamp': datetime.now().isoformat(),
                'signal': adjusted_signal,
                'result': result,
                'status': 'OPEN'
            }
            self.trade_history.append(trade_record)
            self._save_history()
            
            # Update risk manager
            self.risk_manager.daily_trades += 1
            
        else:
            logger.error(f"❌ Trade failed: {result.get('error')}")
        
        return success, result
    
    def close_all_positions(self) -> List[Dict]:
        """
        Close all open positions.
        
        Returns:
            list: Results for each closed position
        """
        results = []
        positions = self.mt5.get_open_positions()
        
        for pos in positions:
            success, result = self.mt5.close_position(pos['ticket'])
            results.append({
                'ticket': pos['ticket'],
                'symbol': pos['symbol'],
                'success': success,
                'result': result
            })
            
            if success:
                self.risk_manager.record_trade(pos['profit'])
        
        return results
    
    def get_open_positions(self) -> List[Dict]:
        """
        Get all open positions.
        
        Returns:
            list: Open positions
        """
        return self.mt5.get_open_positions()
    
    def get_trade_history(self, limit: int = 50) -> List[Dict]:
        """
        Get recent trade history.
        
        Args:
            limit: Maximum number of trades to return
            
        Returns:
            list: Recent trades
        """
        return self.trade_history[-limit:]
    
    def get_daily_summary(self) -> Dict:
        """
        Get daily trading summary.
        
        Returns:
            dict: Daily summary
        """
        risk_stats = self.risk_manager.get_daily_stats()
        open_positions = self.get_open_positions()
        
        total_profit = sum(p['profit'] for p in open_positions)
        
        return {
            **risk_stats,
            'open_positions': len(open_positions),
            'unrealized_profit': total_profit,
            'account_info': self.mt5.get_account_info()
        }
