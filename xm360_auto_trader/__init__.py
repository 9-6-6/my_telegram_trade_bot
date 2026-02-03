"""
XM360 Auto Trader Package

Automatic trade execution for XM360 broker based on
signals from the Telegram Trading Bot.
"""

from . import config
from .mt5_connector import MT5Connector
from .signal_receiver import SignalReceiver, SignalQueue, send_signal_to_auto_trader
from .risk_manager import RiskManager
from .trade_manager import TradeManager

__version__ = '1.0.0'
__all__ = [
    'config',
    'MT5Connector',
    'SignalReceiver',
    'SignalQueue',
    'send_signal_to_auto_trader',
    'RiskManager',
    'TradeManager'
]
