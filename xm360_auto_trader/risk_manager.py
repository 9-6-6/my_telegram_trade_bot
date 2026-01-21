"""
Risk Manager Module

Handles risk management for auto trading:
- Position sizing based on risk percentage
- Daily loss limits
- Maximum position limits
- Trade validation
- Balance usage limits (10% strategy)
- Minimum balance protection
- Price validation (ensure signal price matches market)
"""

import logging
from datetime import datetime
from typing import Optional, Dict, Any, Tuple

from . import config

logger = logging.getLogger('RiskManager')


class RiskManager:
    """
    Risk management for the auto trader.
    
    Strategy:
    - Only use 10% of total balance for trading
    - Stop trading if balance falls below minimum
    - Enforce daily loss limits
    - Position size limits
    """
    
    def __init__(self, mt5_connector):
        """
        Initialize risk manager.
        
        Args:
            mt5_connector: MT5Connector instance
        """
        self.mt5 = mt5_connector
        self.daily_loss = 0.0
        self.daily_trades = 0
        self.last_reset_date = datetime.now().date()
        self.initial_balance = None  # Track starting balance of the day
    
    def _reset_daily_counters(self):
        """Reset daily counters if it's a new day."""
        today = datetime.now().date()
        if today != self.last_reset_date:
            self.daily_loss = 0.0
            self.daily_trades = 0
            self.last_reset_date = today
            # Update initial balance for the new day
            account_info = self.mt5.get_account_info()
            if account_info:
                self.initial_balance = account_info['balance']
            logger.info("📅 Daily counters reset")
    
    def check_balance_requirements(self) -> Tuple[bool, str]:
        """
        Check if account balance meets trading requirements.
        
        Returns:
            tuple: (can_trade, reason)
        """
        account_info = self.mt5.get_account_info()
        if not account_info:
            return False, "Unable to get account info"
        
        balance = account_info['balance']
        equity = account_info['equity']
        free_margin = account_info['free_margin']
        
        # Check minimum balance
        if balance < config.MIN_BALANCE_TO_TRADE:
            return False, f"Balance (${balance:.2f}) below minimum (${config.MIN_BALANCE_TO_TRADE:.2f})"
        
        # Check minimum free margin
        if free_margin < config.MIN_FREE_MARGIN:
            return False, f"Free margin (${free_margin:.2f}) below minimum (${config.MIN_FREE_MARGIN:.2f})"
        
        # Check if we're using more than 10% of balance
        margin_used = account_info['margin']
        max_margin_allowed = balance * (config.MAX_BALANCE_USAGE_PERCENT / 100)
        
        if margin_used >= max_margin_allowed:
            return False, f"Margin usage (${margin_used:.2f}) exceeds {config.MAX_BALANCE_USAGE_PERCENT}% limit (${max_margin_allowed:.2f})"
        
        return True, "OK"
    
    def get_available_trading_balance(self) -> float:
        """
        Get the available balance for trading (10% of total).
        
        Returns:
            float: Available trading balance
        """
        account_info = self.mt5.get_account_info()
        if not account_info:
            return 0.0
        
        balance = account_info['balance']
        margin_used = account_info['margin']
        
        # Maximum margin we can use (10% of balance)
        max_margin = balance * (config.MAX_BALANCE_USAGE_PERCENT / 100)
        
        # Available margin for new trades
        available = max_margin - margin_used
        
        return max(0.0, available)
    
    def _get_max_price_deviation(self, symbol: str) -> Tuple[float, str]:
        """
        Get the maximum allowed price deviation for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            tuple: (max_deviation, deviation_type) - type is 'absolute' or 'percent'
        """
        symbol_upper = symbol.upper()
        
        # Check for specific asset types
        if 'XAU' in symbol_upper or 'GOLD' in symbol_upper:
            if config.MAX_PRICE_DEVIATION_GOLD > 0:
                return config.MAX_PRICE_DEVIATION_GOLD, 'absolute'
        
        elif 'XAG' in symbol_upper or 'SILVER' in symbol_upper:
            if config.MAX_PRICE_DEVIATION_SILVER > 0:
                return config.MAX_PRICE_DEVIATION_SILVER, 'absolute'
        
        elif 'BTC' in symbol_upper or 'ETH' in symbol_upper:
            if config.MAX_PRICE_DEVIATION_CRYPTO > 0:
                return config.MAX_PRICE_DEVIATION_CRYPTO, 'absolute'
        
        elif any(idx in symbol_upper for idx in ['US30', 'US500', 'US100', 'NAS', 'SPX', 'DOW']):
            if config.MAX_PRICE_DEVIATION_INDICES > 0:
                return config.MAX_PRICE_DEVIATION_INDICES, 'absolute'
        
        elif config.MAX_PRICE_DEVIATION_FOREX > 0:
            return config.MAX_PRICE_DEVIATION_FOREX, 'absolute'
        
        # Default to percentage
        return config.MAX_PRICE_DEVIATION_PERCENT, 'percent'
    
    def validate_price(self, symbol: str, signal_price: float, direction: str) -> Tuple[bool, str, float]:
        """
        Validate that the signal price matches the current market price.
        
        Args:
            symbol: Trading symbol
            signal_price: Entry price from the signal
            direction: 'BUY' or 'SELL'
            
        Returns:
            tuple: (is_valid, message, current_price)
        """
        if not config.ENABLE_PRICE_VALIDATION:
            return True, "Price validation disabled", signal_price
        
        # Get current market price
        price_data = self.mt5.get_current_price(symbol)
        if not price_data:
            return False, f"Cannot get current price for {symbol}", 0.0
        
        bid, ask = price_data
        
        # Use appropriate price based on direction
        if direction.upper() == 'BUY':
            current_price = ask  # Buy at ask price
        else:
            current_price = bid  # Sell at bid price
        
        # Calculate deviation
        price_diff = abs(current_price - signal_price)
        
        # Get max allowed deviation
        max_deviation, deviation_type = self._get_max_price_deviation(symbol)
        
        if deviation_type == 'percent':
            # Calculate percentage deviation
            deviation_percent = (price_diff / signal_price) * 100 if signal_price > 0 else 100
            is_valid = deviation_percent <= max_deviation
            
            if config.LOG_PRICE_VALIDATION:
                logger.info(f"📊 Price Validation for {symbol}:")
                logger.info(f"   Signal Price: {signal_price:.5f}")
                logger.info(f"   Current {'Ask' if direction.upper() == 'BUY' else 'Bid'}: {current_price:.5f}")
                logger.info(f"   Deviation: {deviation_percent:.3f}% (Max: {max_deviation}%)")
                logger.info(f"   Status: {'✅ VALID' if is_valid else '❌ REJECTED'}")
            
            if is_valid:
                return True, f"Price valid ({deviation_percent:.2f}% deviation)", current_price
            else:
                return False, f"Price deviation too high: {deviation_percent:.2f}% (max {max_deviation}%)", current_price
        
        else:
            # Absolute deviation
            is_valid = price_diff <= max_deviation
            
            if config.LOG_PRICE_VALIDATION:
                logger.info(f"📊 Price Validation for {symbol}:")
                logger.info(f"   Signal Price: {signal_price:.5f}")
                logger.info(f"   Current {'Ask' if direction.upper() == 'BUY' else 'Bid'}: {current_price:.5f}")
                logger.info(f"   Deviation: {price_diff:.5f} (Max: {max_deviation})")
                logger.info(f"   Status: {'✅ VALID' if is_valid else '❌ REJECTED'}")
            
            if is_valid:
                return True, f"Price valid ({price_diff:.2f} deviation)", current_price
            else:
                return False, f"Price deviation too high: {price_diff:.2f} (max {max_deviation})", current_price
    
    def can_trade(self) -> Tuple[bool, str]:
        """
        Check if trading is allowed based on risk rules.
        
        Returns:
            tuple: (can_trade, reason)
        """
        self._reset_daily_counters()
        
        if not self.mt5.is_connected():
            return False, "Not connected to MT5"
        
        account_info = self.mt5.get_account_info()
        if not account_info:
            return False, "Unable to get account info"
        
        # Check balance requirements (10% strategy + minimum balance)
        can_trade, reason = self.check_balance_requirements()
        if not can_trade:
            return False, reason
        
        # Check daily trade limit
        if self.daily_trades >= config.MAX_DAILY_TRADES:
            return False, f"Daily trade limit reached ({config.MAX_DAILY_TRADES})"
        
        # Check daily loss limit
        max_daily_loss = account_info['balance'] * (config.MAX_DAILY_LOSS_PERCENT / 100)
        if self.daily_loss >= max_daily_loss:
            return False, f"Daily loss limit reached (${self.daily_loss:.2f})"
        
        # Check open positions limit
        open_positions = self.mt5.get_open_positions()
        if len(open_positions) >= config.MAX_OPEN_POSITIONS:
            return False, f"Max open positions reached ({config.MAX_OPEN_POSITIONS})"
        
        return True, "OK"
    
    def calculate_lot_size(
        self,
        symbol: str,
        stop_loss_pips: float,
        risk_percent: float = None
    ) -> float:
        """
        Calculate position size based on risk percentage and 10% balance limit.
        
        Args:
            symbol: Trading symbol
            stop_loss_pips: Distance to stop loss in pips
            risk_percent: Risk percentage (uses config default if None)
            
        Returns:
            float: Calculated lot size
        """
        if risk_percent is None:
            risk_percent = config.MAX_RISK_PERCENT
        
        account_info = self.mt5.get_account_info()
        if not account_info:
            return config.DEFAULT_LOT_SIZE
        
        symbol_info = self.mt5.get_symbol_info(symbol)
        if not symbol_info:
            return config.DEFAULT_LOT_SIZE
        
        balance = account_info['balance']
        
        # Check minimum balance
        if balance < config.MIN_BALANCE_TO_TRADE:
            logger.warning(f"⚠️ Balance (${balance:.2f}) below minimum. Cannot calculate lot size.")
            return 0.0
        
        # Calculate risk amount (based on risk percentage)
        risk_amount = balance * (risk_percent / 100)
        
        # Also ensure we don't exceed 10% balance usage
        available_trading_balance = self.get_available_trading_balance()
        if available_trading_balance <= 0:
            logger.warning(f"⚠️ No available trading balance (10% limit reached)")
            return 0.0
        
        # Cap risk amount to available trading balance
        risk_amount = min(risk_amount, available_trading_balance)
        
        # Calculate pip value (simplified)
        # For most forex pairs, 1 standard lot = $10 per pip
        # For gold, 1 standard lot = $10 per pip (0.1 price change)
        pip_value = 10.0  # Approximate, actual depends on account currency and symbol
        
        # Calculate lot size
        if stop_loss_pips > 0:
            lot_size = risk_amount / (stop_loss_pips * pip_value)
        else:
            lot_size = config.DEFAULT_LOT_SIZE
        
        # Apply constraints
        lot_size = max(config.MIN_LOT_SIZE, min(lot_size, config.MAX_LOT_SIZE))
        
        # Round to valid step
        lot_size = round(lot_size / symbol_info['volume_step']) * symbol_info['volume_step']
        
        logger.info(f"📊 Lot size calculated: {lot_size} (Balance: ${balance:.2f}, Available: ${available_trading_balance:.2f})")
        
        return lot_size
    
    def validate_trade(self, signal: Dict) -> Tuple[bool, str, Dict]:
        """
        Validate a trade before execution.
        
        Validation includes:
        1. Check if trading is allowed (balance, limits)
        2. Check if symbol is available
        3. Validate price matches current market (price trending check)
        4. Validate stop loss position
        5. Calculate appropriate lot size
        
        Args:
            signal: Trading signal
            
        Returns:
            tuple: (is_valid, message, adjusted_signal)
        """
        # Check if trading is allowed (includes 10% balance check)
        can_trade, reason = self.can_trade()
        if not can_trade:
            return False, reason, signal
        
        symbol = signal['symbol']
        direction = signal['direction']
        entry_price = signal['entry_price']
        
        # Get symbol info
        symbol_info = self.mt5.get_symbol_info(symbol)
        if not symbol_info:
            return False, f"Symbol not available: {symbol}", signal
        
        # Check if market is open
        if symbol_info.get('trade_mode', 0) == 0:
            return False, f"Market closed for {symbol}", signal
        
        # ============================================================
        # PRICE VALIDATION - Check if signal price matches market price
        # ============================================================
        price_valid, price_message, current_price = self.validate_price(symbol, entry_price, direction)
        if not price_valid:
            logger.warning(f"❌ Price validation failed for {symbol}: {price_message}")
            return False, f"Price validation failed: {price_message}", signal
        
        # Check available trading balance (10% strategy)
        available_balance = self.get_available_trading_balance()
        if available_balance <= 0:
            return False, f"10% balance limit reached. No margin available for new trades.", signal
        
        # Check for existing position in same symbol
        open_positions = self.mt5.get_open_positions()
        for pos in open_positions:
            if pos['symbol'] == config.SYMBOL_MAPPING.get(symbol, symbol):
                # Already have a position
                if pos['type'] == direction:
                    return False, f"Already have {direction} position in {symbol}", signal
                else:
                    logger.warning(f"⚠️ Opening opposite position in {symbol}")
        
        # Validate stop loss
        if signal.get('stop_loss'):
            entry = signal['entry_price']
            sl = signal['stop_loss']
            
            # Check SL is on correct side
            if direction == 'BUY' and sl >= entry:
                return False, "Stop loss must be below entry for BUY", signal
            if direction == 'SELL' and sl <= entry:
                return False, "Stop loss must be above entry for SELL", signal
        
        # Calculate lot size if not specified
        adjusted_signal = signal.copy()
        
        # Update entry price to current market price (use validated current price)
        adjusted_signal['original_signal_price'] = entry_price
        adjusted_signal['executed_price'] = current_price
        
        if not signal.get('lot_size') or signal['lot_size'] == config.DEFAULT_LOT_SIZE:
            if signal.get('stop_loss'):
                sl_distance = abs(signal['entry_price'] - signal['stop_loss'])
                # Convert to pips (approximate)
                sl_pips = sl_distance / symbol_info.get('point', 0.0001)
                calculated_lot = self.calculate_lot_size(symbol, sl_pips)
                
                # If lot size is 0, we can't trade
                if calculated_lot <= 0:
                    return False, "Cannot calculate valid lot size (balance limits)", signal
                
                adjusted_signal['lot_size'] = calculated_lot
            else:
                adjusted_signal['lot_size'] = config.DEFAULT_LOT_SIZE
        
        # Enforce lot size limits
        adjusted_signal['lot_size'] = max(
            config.MIN_LOT_SIZE,
            min(adjusted_signal['lot_size'], config.MAX_LOT_SIZE)
        )
        
        # Log the trade validation
        account_info = self.mt5.get_account_info()
        if account_info:
            logger.info(f"✅ Trade validated: {symbol} {direction}")
            logger.info(f"   Signal Price: {entry_price:.5f} → Current: {current_price:.5f}")
            logger.info(f"   Price Check: {price_message}")
            logger.info(f"   Balance: ${account_info['balance']:.2f}, Available (10%): ${available_balance:.2f}")
            logger.info(f"   Lot Size: {adjusted_signal['lot_size']}")
        
        return True, "OK", adjusted_signal
    
    def record_trade(self, profit: float):
        """
        Record a completed trade for risk tracking.
        
        Args:
            profit: Trade profit/loss
        """
        self._reset_daily_counters()
        
        self.daily_trades += 1
        if profit < 0:
            self.daily_loss += abs(profit)
        
        logger.info(f"📊 Daily stats: {self.daily_trades} trades, ${self.daily_loss:.2f} loss")
    
    def get_daily_stats(self) -> Dict:
        """
        Get daily trading statistics.
        
        Returns:
            dict: Daily stats
        """
        self._reset_daily_counters()
        
        account_info = self.mt5.get_account_info() or {}
        balance = account_info.get('balance', 0)
        equity = account_info.get('equity', 0)
        margin_used = account_info.get('margin', 0)
        free_margin = account_info.get('free_margin', 0)
        
        max_loss = balance * (config.MAX_DAILY_LOSS_PERCENT / 100)
        max_margin = balance * (config.MAX_BALANCE_USAGE_PERCENT / 100)
        available_trading = max(0, max_margin - margin_used)
        
        # Check trading status
        can_trade = True
        status_reasons = []
        
        if balance < config.MIN_BALANCE_TO_TRADE:
            can_trade = False
            status_reasons.append(f"Balance below ${config.MIN_BALANCE_TO_TRADE}")
        
        if margin_used >= max_margin:
            can_trade = False
            status_reasons.append(f"10% balance limit reached")
        
        if self.daily_trades >= config.MAX_DAILY_TRADES:
            can_trade = False
            status_reasons.append(f"Daily trade limit reached")
        
        if self.daily_loss >= max_loss:
            can_trade = False
            status_reasons.append(f"Daily loss limit reached")
        
        return {
            'date': self.last_reset_date.isoformat(),
            'balance': balance,
            'equity': equity,
            'margin_used': margin_used,
            'free_margin': free_margin,
            'trades_executed': self.daily_trades,
            'trades_remaining': config.MAX_DAILY_TRADES - self.daily_trades,
            'daily_loss': self.daily_loss,
            'max_daily_loss': max_loss,
            'loss_remaining': max_loss - self.daily_loss,
            'max_trading_margin': max_margin,
            'available_trading_balance': available_trading,
            'balance_usage_percent': (margin_used / max_margin * 100) if max_margin > 0 else 0,
            'can_trade': can_trade and self.daily_trades < config.MAX_DAILY_TRADES and self.daily_loss < max_loss,
            'status_reasons': status_reasons if not can_trade else ['Ready to trade']
        }
