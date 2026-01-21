"""
Signal Bridge Module

This module provides a simple interface to send signals from
the main trading bot to the XM360 auto trader.

Usage in your trading bot:
    from xm360_auto_trader.signal_bridge import forward_signal_to_auto_trader
    
    # When a signal is generated
    forward_signal_to_auto_trader(
        symbol='XAUUSD',
        direction='BUY',
        entry_price=1920.50,
        stop_loss=1915.00,
        take_profit=1930.00,
        confidence=85
    )
"""

import os
import json
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger('SignalBridge')

# Signal queue file path
SIGNAL_QUEUE_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'signal_queue.json'
)


def forward_signal_to_auto_trader(
    symbol: str,
    direction: str,
    entry_price: float,
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None,
    lot_size: Optional[float] = None,
    confidence: Optional[float] = None,
    source: str = 'TelegramBot'
) -> bool:
    """
    Forward a trading signal to the XM360 auto trader.
    
    This function writes the signal to a shared queue file that
    the auto trader monitors. This is a non-blocking operation.
    
    Args:
        symbol: Trading symbol (e.g., 'XAUUSD', 'EURUSD')
        direction: Trade direction ('BUY' or 'SELL')
        entry_price: Entry price
        stop_loss: Stop loss price (optional)
        take_profit: Take profit price (optional)
        lot_size: Lot size (optional, uses default if not set)
        confidence: Signal confidence percentage (optional)
        source: Signal source identifier
        
    Returns:
        bool: True if signal was queued successfully
        
    Example:
        >>> forward_signal_to_auto_trader(
        ...     symbol='XAUUSD',
        ...     direction='BUY', 
        ...     entry_price=1920.50,
        ...     stop_loss=1915.00,
        ...     take_profit=1930.00
        ... )
        True
    """
    try:
        # Create signal object
        signal = {
            'symbol': symbol.upper(),
            'direction': direction.upper(),
            'entry_price': float(entry_price),
            'timestamp': datetime.now().isoformat(),
            'source': source,
            'processed': False
        }
        
        # Add optional fields
        if stop_loss is not None:
            signal['stop_loss'] = float(stop_loss)
        if take_profit is not None:
            signal['take_profit'] = float(take_profit)
        if lot_size is not None:
            signal['lot_size'] = float(lot_size)
        if confidence is not None:
            signal['confidence'] = float(confidence)
        
        # Read existing queue
        signals = []
        if os.path.exists(SIGNAL_QUEUE_FILE):
            try:
                with open(SIGNAL_QUEUE_FILE, 'r') as f:
                    signals = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                signals = []
        
        # Add new signal
        signals.append(signal)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(SIGNAL_QUEUE_FILE), exist_ok=True)
        
        # Write queue
        with open(SIGNAL_QUEUE_FILE, 'w') as f:
            json.dump(signals, f, indent=2)
        
        logger.info(f"📤 Signal forwarded to auto trader: {symbol} {direction}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to forward signal: {e}")
        return False


def parse_signal_message_and_forward(message: str, source: str = 'TelegramBot') -> bool:
    """
    Parse a formatted signal message and forward it to the auto trader.
    
    This function extracts signal details from a formatted message
    (like those sent by the Telegram bot) and forwards them.
    
    Args:
        message: Formatted signal message
        source: Signal source identifier
        
    Returns:
        bool: True if signal was parsed and forwarded successfully
        
    Example message format:
        🚀 SCALP SIGNAL - XAUUSD
        
        Direction: 🟢 BUY
        Entry Price: $1920.50
        Stop Loss: $1915.00
        Take Profit: $1930.00
        Confidence: 85%
    """
    import re
    
    try:
        # Extract symbol
        symbol_match = re.search(r'SIGNAL.*?[-–]\s*([A-Z0-9]+)', message, re.IGNORECASE)
        if not symbol_match:
            # Try alternative format
            symbol_match = re.search(r'Symbol[:\s]+`?([A-Z0-9]+)`?', message, re.IGNORECASE)
        
        if not symbol_match:
            logger.warning("Could not extract symbol from message")
            return False
        
        symbol = symbol_match.group(1).upper()
        
        # Extract direction
        direction = None
        if 'BUY' in message.upper() or '🟢' in message:
            direction = 'BUY'
        elif 'SELL' in message.upper() or '🔴' in message:
            direction = 'SELL'
        
        if not direction:
            logger.warning("Could not extract direction from message")
            return False
        
        # Extract prices
        entry_match = re.search(r'Entry[:\s]+[$€£]?([\d,.]+)', message, re.IGNORECASE)
        sl_match = re.search(r'Stop Loss[:\s]+[$€£]?([\d,.]+)', message, re.IGNORECASE)
        tp_match = re.search(r'Take Profit[:\s]+[$€£]?([\d,.]+)', message, re.IGNORECASE)
        conf_match = re.search(r'Confidence[:\s]+([\d.]+)', message, re.IGNORECASE)
        
        if not entry_match:
            logger.warning("Could not extract entry price from message")
            return False
        
        entry_price = float(entry_match.group(1).replace(',', ''))
        stop_loss = float(sl_match.group(1).replace(',', '')) if sl_match else None
        take_profit = float(tp_match.group(1).replace(',', '')) if tp_match else None
        confidence = float(conf_match.group(1)) if conf_match else None
        
        # Forward the signal
        return forward_signal_to_auto_trader(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
            source=source
        )
        
    except Exception as e:
        logger.error(f"Failed to parse and forward signal: {e}")
        return False


# For testing
if __name__ == '__main__':
    # Test signal forwarding
    print("Testing signal bridge...")
    
    success = forward_signal_to_auto_trader(
        symbol='XAUUSD',
        direction='BUY',
        entry_price=1920.50,
        stop_loss=1915.00,
        take_profit=1930.00,
        confidence=85,
        source='Test'
    )
    
    print(f"Signal forwarded: {success}")
    
    # Check queue file
    if os.path.exists(SIGNAL_QUEUE_FILE):
        with open(SIGNAL_QUEUE_FILE, 'r') as f:
            print(f"Queue contents:\n{f.read()}")
