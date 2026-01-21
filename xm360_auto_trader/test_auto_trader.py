"""
Test Script for XM360 Auto Trader

This script tests the auto trader functionality without executing real trades.
Run this to verify your setup before going live.

Usage:
    python test_auto_trader.py
"""

import os
import sys
import time
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xm360_auto_trader import config
from xm360_auto_trader.signal_bridge import forward_signal_to_auto_trader, SIGNAL_QUEUE_FILE


def test_signal_queue():
    """Test signal queue functionality."""
    print("\n" + "="*60)
    print("TEST 1: Signal Queue")
    print("="*60)
    
    # Clear existing queue
    if os.path.exists(SIGNAL_QUEUE_FILE):
        os.remove(SIGNAL_QUEUE_FILE)
        print("✓ Cleared existing queue")
    
    # Send test signal
    success = forward_signal_to_auto_trader(
        symbol='XAUUSD',
        direction='BUY',
        entry_price=1920.50,
        stop_loss=1915.00,
        take_profit=1930.00,
        confidence=85,
        source='Test'
    )
    
    if success:
        print("✓ Test signal queued successfully")
    else:
        print("✗ Failed to queue signal")
        return False
    
    # Verify queue file
    if os.path.exists(SIGNAL_QUEUE_FILE):
        with open(SIGNAL_QUEUE_FILE, 'r') as f:
            signals = json.load(f)
            if len(signals) == 1:
                print("✓ Queue file contains 1 signal")
                print(f"  Signal: {signals[0]['symbol']} {signals[0]['direction']}")
            else:
                print(f"✗ Unexpected queue length: {len(signals)}")
                return False
    else:
        print("✗ Queue file not created")
        return False
    
    return True


def test_configuration():
    """Test configuration settings."""
    print("\n" + "="*60)
    print("TEST 2: Configuration")
    print("="*60)
    
    issues = []
    
    # Check account settings
    if not config.ACCOUNT:
        issues.append("ACCOUNT not set")
    else:
        print(f"✓ Account: {config.ACCOUNT}")
    
    if not config.PASSWORD:
        issues.append("PASSWORD not set")
    else:
        print("✓ Password: ***configured***")
    
    if config.SERVER:
        print(f"✓ Server: {config.SERVER}")
    else:
        issues.append("SERVER not set")
    
    # Check risk settings
    print(f"✓ Use Demo: {config.USE_DEMO}")
    print(f"✓ Default Lot: {config.DEFAULT_LOT_SIZE}")
    print(f"✓ Max Risk: {config.MAX_RISK_PERCENT}%")
    print(f"✓ Max Positions: {config.MAX_OPEN_POSITIONS}")
    print(f"✓ Max Daily Loss: {config.MAX_DAILY_LOSS_PERCENT}%")
    
    # Check symbol mapping
    print(f"✓ Symbol mappings: {len(config.SYMBOL_MAPPING)}")
    
    if issues:
        print("\n⚠️ Configuration Issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    
    return True


def test_mt5_connection():
    """Test MetaTrader 5 connection."""
    print("\n" + "="*60)
    print("TEST 3: MT5 Connection")
    print("="*60)
    
    try:
        import MetaTrader5 as mt5
        print("✓ MetaTrader5 library imported")
    except ImportError:
        print("✗ MetaTrader5 library not installed")
        print("  Run: pip install MetaTrader5")
        return False
    
    # Try to initialize MT5
    if not mt5.initialize():
        error = mt5.last_error()
        print(f"✗ MT5 initialization failed: {error}")
        print("\n  Make sure:")
        print("  1. MetaTrader 5 is installed")
        print("  2. MT5 terminal is running")
        print("  3. You're on Windows")
        return False
    
    print("✓ MT5 initialized")
    
    # Try to login
    if config.ACCOUNT and config.PASSWORD:
        login_result = mt5.login(
            login=config.ACCOUNT,
            password=config.PASSWORD,
            server=config.SERVER
        )
        
        if login_result:
            account_info = mt5.account_info()
            print(f"✓ Logged in successfully")
            print(f"  Account: {account_info.login}")
            print(f"  Server: {account_info.server}")
            print(f"  Balance: ${account_info.balance:.2f}")
            print(f"  Type: {'DEMO' if account_info.trade_mode == 0 else 'LIVE'}")
            
            # Verify demo mode safety
            if config.USE_DEMO and account_info.trade_mode != 0:
                print("\n⚠️ WARNING: USE_DEMO is True but connected to LIVE account!")
                print("   Auto trading will be disabled for safety.")
        else:
            error = mt5.last_error()
            print(f"✗ Login failed: {error}")
            mt5.shutdown()
            return False
    else:
        print("⚠️ Account credentials not configured, skipping login test")
    
    mt5.shutdown()
    print("✓ MT5 disconnected cleanly")
    
    return True


def test_symbol_availability():
    """Test if trading symbols are available."""
    print("\n" + "="*60)
    print("TEST 4: Symbol Availability")
    print("="*60)
    
    try:
        import MetaTrader5 as mt5
    except ImportError:
        print("⚠️ Skipping (MT5 not available)")
        return True
    
    if not mt5.initialize():
        print("⚠️ Skipping (MT5 not initialized)")
        return True
    
    # Login required for symbol info
    if config.ACCOUNT and config.PASSWORD:
        if not mt5.login(login=config.ACCOUNT, password=config.PASSWORD, server=config.SERVER):
            print("⚠️ Skipping (login failed)")
            mt5.shutdown()
            return True
    
    # Test each symbol
    print("\nChecking symbol availability:")
    available = []
    unavailable = []
    
    for telegram_symbol, mt5_symbol in config.SYMBOL_MAPPING.items():
        info = mt5.symbol_info(mt5_symbol)
        if info:
            available.append((telegram_symbol, mt5_symbol, info.bid))
            # Enable symbol
            mt5.symbol_select(mt5_symbol, True)
        else:
            unavailable.append((telegram_symbol, mt5_symbol))
    
    # Print results
    print("\n✓ Available symbols:")
    for ts, ms, price in available:
        print(f"  {ts} -> {ms}: {price:.5f}")
    
    if unavailable:
        print("\n✗ Unavailable symbols:")
        for ts, ms in unavailable:
            print(f"  {ts} -> {ms}")
    
    mt5.shutdown()
    return len(unavailable) == 0


def run_all_tests():
    """Run all tests."""
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║         XM360 AUTO TRADER - TEST SUITE                     ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    results = {
        'Signal Queue': test_signal_queue(),
        'Configuration': test_configuration(),
        'MT5 Connection': test_mt5_connection(),
        'Symbol Availability': test_symbol_availability()
    }
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL TESTS PASSED - Ready to run auto trader!")
    else:
        print("⚠️ SOME TESTS FAILED - Please fix issues before running")
    print("="*60)
    
    return all_passed


if __name__ == '__main__':
    run_all_tests()
