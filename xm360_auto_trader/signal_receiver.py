"""
Signal Receiver Module

Receives trading signals from the main Telegram bot.
Supports multiple communication methods:
1. File-based queue (default)
2. In-memory queue for same-process usage
"""

import json
import os
import time
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List, Callable
from queue import Queue
import threading

from . import config

logger = logging.getLogger('SignalReceiver')


class SignalQueue:
    """Thread-safe signal queue for inter-process communication."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.queue_file = config.SIGNAL_QUEUE_FILE
        self.memory_queue = Queue()
        self._initialized = True
        
        # Ensure queue file directory exists
        os.makedirs(os.path.dirname(self.queue_file), exist_ok=True)
        
        # Initialize empty queue file if not exists
        if not os.path.exists(self.queue_file):
            self._write_queue([])
    
    def _read_queue(self) -> List[Dict]:
        """Read signals from file queue."""
        try:
            with open(self.queue_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []
    
    def _write_queue(self, signals: List[Dict]):
        """Write signals to file queue."""
        with open(self.queue_file, 'w') as f:
            json.dump(signals, f, indent=2)
    
    def push_signal(self, signal: Dict[str, Any]):
        """
        Add a new signal to the queue.
        
        Args:
            signal: Trading signal dictionary
        """
        # Add timestamp if not present
        if 'timestamp' not in signal:
            signal['timestamp'] = datetime.now().isoformat()
        
        if 'processed' not in signal:
            signal['processed'] = False
        
        # Add to file queue
        signals = self._read_queue()
        signals.append(signal)
        self._write_queue(signals)
        
        # Also add to memory queue
        self.memory_queue.put(signal)
        
        logger.info(f"📥 Signal queued: {signal.get('symbol')} {signal.get('direction')}")
    
    def get_pending_signals(self) -> List[Dict]:
        """
        Get all unprocessed signals.
        
        Returns:
            list: List of pending signals
        """
        signals = self._read_queue()
        return [s for s in signals if not s.get('processed', True)]
    
    def mark_processed(self, signal_timestamp: str, result: Dict = None):
        """
        Mark a signal as processed.
        
        Args:
            signal_timestamp: Timestamp of the signal to mark
            result: Optional result of processing
        """
        signals = self._read_queue()
        for s in signals:
            if s.get('timestamp') == signal_timestamp:
                s['processed'] = True
                s['processed_at'] = datetime.now().isoformat()
                if result:
                    s['result'] = result
                break
        self._write_queue(signals)
    
    def clear_old_signals(self, hours: int = 24):
        """
        Remove signals older than specified hours.
        
        Args:
            hours: Remove signals older than this many hours
        """
        signals = self._read_queue()
        cutoff = datetime.now().timestamp() - (hours * 3600)
        
        filtered = []
        for s in signals:
            try:
                signal_time = datetime.fromisoformat(s['timestamp']).timestamp()
                if signal_time > cutoff:
                    filtered.append(s)
            except:
                pass
        
        self._write_queue(filtered)
        logger.info(f"🗑️ Cleared {len(signals) - len(filtered)} old signals")


class SignalReceiver:
    """
    Receives and validates trading signals from the main bot.
    """
    
    def __init__(self, on_signal_callback: Callable = None):
        """
        Initialize the signal receiver.
        
        Args:
            on_signal_callback: Function to call when a new signal is received
        """
        self.queue = SignalQueue()
        self.on_signal_callback = on_signal_callback
        self.running = False
        self._thread = None
        self.last_signal_times = {}  # Track last signal per symbol
    
    def validate_signal(self, signal: Dict) -> tuple:
        """
        Validate a trading signal.
        
        Args:
            signal: Signal dictionary to validate
            
        Returns:
            tuple: (is_valid, error_message)
        """
        required_fields = ['symbol', 'direction', 'entry_price']
        
        # Check required fields
        for field in required_fields:
            if field not in signal:
                return False, f"Missing required field: {field}"
        
        # Validate direction
        if signal['direction'].upper() not in ['BUY', 'SELL']:
            return False, f"Invalid direction: {signal['direction']}"
        
        # Check symbol is allowed
        symbol = signal['symbol'].upper()
        
        if config.BLOCKED_SYMBOLS and symbol in config.BLOCKED_SYMBOLS:
            return False, f"Symbol blocked: {symbol}"
        
        if config.ALLOWED_SYMBOLS and symbol not in config.ALLOWED_SYMBOLS:
            return False, f"Symbol not in allowed list: {symbol}"
        
        # Check if symbol is mapped
        if symbol not in config.SYMBOL_MAPPING:
            logger.warning(f"Symbol not in mapping, using as-is: {symbol}")
        
        # Check signal age
        if 'timestamp' in signal:
            try:
                signal_time = datetime.fromisoformat(signal['timestamp'])
                age_seconds = (datetime.now() - signal_time).total_seconds()
                
                if age_seconds > config.SIGNAL_EXPIRY_SECONDS:
                    return False, f"Signal expired (age: {age_seconds:.0f}s)"
            except:
                pass
        
        # Check minimum interval between signals for same symbol
        last_time = self.last_signal_times.get(symbol)
        if last_time:
            elapsed = (datetime.now() - last_time).total_seconds()
            if elapsed < config.MIN_SIGNAL_INTERVAL:
                return False, f"Signal too soon for {symbol} (wait {config.MIN_SIGNAL_INTERVAL - elapsed:.0f}s)"
        
        return True, None
    
    def process_signal(self, signal: Dict) -> Dict:
        """
        Process and normalize a signal.
        
        Args:
            signal: Raw signal dictionary
            
        Returns:
            dict: Processed signal
        """
        processed = {
            'symbol': signal['symbol'].upper(),
            'direction': signal['direction'].upper(),
            'entry_price': float(signal['entry_price']),
            'stop_loss': float(signal.get('stop_loss', 0)) if signal.get('stop_loss') else None,
            'take_profit': float(signal.get('take_profit', 0)) if signal.get('take_profit') else None,
            'lot_size': float(signal.get('lot_size', config.DEFAULT_LOT_SIZE)),
            'timestamp': signal.get('timestamp', datetime.now().isoformat()),
            'source': signal.get('source', 'TelegramBot'),
            'confidence': signal.get('confidence', 0)
        }
        
        # Record signal time for this symbol
        self.last_signal_times[processed['symbol']] = datetime.now()
        
        return processed
    
    def start_listening(self):
        """Start listening for signals in a background thread."""
        if self.running:
            logger.warning("Signal receiver already running")
            return
        
        self.running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        logger.info("📡 Signal receiver started")
    
    def stop_listening(self):
        """Stop listening for signals."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Signal receiver stopped")
    
    def _listen_loop(self):
        """Main listening loop."""
        while self.running:
            try:
                # Check for pending signals
                pending = self.queue.get_pending_signals()
                
                for signal in pending:
                    # Validate signal
                    is_valid, error = self.validate_signal(signal)
                    
                    if not is_valid:
                        logger.warning(f"❌ Invalid signal: {error}")
                        self.queue.mark_processed(signal['timestamp'], {'error': error})
                        continue
                    
                    # Process signal
                    processed = self.process_signal(signal)
                    
                    # Call callback if set
                    if self.on_signal_callback:
                        try:
                            result = self.on_signal_callback(processed)
                            self.queue.mark_processed(signal['timestamp'], result)
                        except Exception as e:
                            logger.error(f"Callback error: {e}")
                            self.queue.mark_processed(signal['timestamp'], {'error': str(e)})
                    else:
                        self.queue.mark_processed(signal['timestamp'])
                
                # Clear old signals periodically
                self.queue.clear_old_signals(24)
                
            except Exception as e:
                logger.error(f"Error in signal listener: {e}")
            
            time.sleep(config.SIGNAL_CHECK_INTERVAL)


# Utility function for the main trading bot to send signals
def send_signal_to_auto_trader(signal: Dict):
    """
    Send a trading signal to the auto trader.
    
    Call this from the main trading bot when a signal is generated.
    
    Args:
        signal: Signal dictionary with at least:
            - symbol: Trading symbol (e.g., 'XAUUSD')
            - direction: 'BUY' or 'SELL'
            - entry_price: Entry price
            - stop_loss: Stop loss price (optional)
            - take_profit: Take profit price (optional)
    
    Example:
        from xm360_auto_trader.signal_receiver import send_signal_to_auto_trader
        
        send_signal_to_auto_trader({
            'symbol': 'XAUUSD',
            'direction': 'BUY',
            'entry_price': 1920.50,
            'stop_loss': 1915.00,
            'take_profit': 1930.00,
            'confidence': 85
        })
    """
    queue = SignalQueue()
    queue.push_signal(signal)
