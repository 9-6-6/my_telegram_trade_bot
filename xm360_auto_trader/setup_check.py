"""
Quick Setup Script for XM360 Auto Trader on Local PC
Run this first to verify everything is ready!
"""

import os
import sys

def check_setup():
    print("=" * 50)
    print("  XM360 Auto Trader - Setup Checker")
    print("=" * 50)
    print()
    
    all_good = True
    
    # Check 1: Python version
    print("1. Checking Python version...")
    print(f"   Python {sys.version}")
    print("   ✅ Python OK")
    print()
    
    # Check 2: MetaTrader5 library
    print("2. Checking MetaTrader5 library...")
    try:
        import MetaTrader5 as mt5
        print("   ✅ MetaTrader5 library installed")
    except ImportError:
        print("   ❌ MetaTrader5 library NOT installed")
        print("   Run: pip install MetaTrader5")
        all_good = False
    print()
    
    # Check 3: MT5 Terminal
    print("3. Checking MT5 Desktop Application...")
    mt5_paths = [
        r"C:\Program Files\MetaTrader 5\terminal64.exe",
        r"C:\Program Files (x86)\MetaTrader 5\terminal64.exe",
        r"C:\Program Files\XM MT5\terminal64.exe",
        r"C:\Program Files (x86)\XM MT5\terminal64.exe",
        os.path.expanduser(r"~\AppData\Roaming\MetaQuotes\Terminal")
    ]
    
    mt5_found = False
    for path in mt5_paths:
        if os.path.exists(path):
            print(f"   ✅ Found MT5 at: {path}")
            mt5_found = True
            break
    
    if not mt5_found:
        print("   ❌ MT5 Desktop NOT found")
        print()
        print("   📥 DOWNLOAD MT5 from XM:")
        print("   https://www.xm.com/mt5")
        print()
        print("   After installing MT5:")
        print("   1. Open MT5")
        print("   2. Login with your account:")
        print("      - Account: 315982803")
        print("      - Password: Gadhiya@098")
        print("      - Server: XMGlobal-MT5 7")
        print("   3. Keep MT5 OPEN and run this script again")
        all_good = False
    print()
    
    # Check 4: Try to connect to MT5
    print("4. Trying to connect to MT5...")
    try:
        import MetaTrader5 as mt5
        if mt5.initialize():
            account_info = mt5.account_info()
            if account_info:
                print(f"   ✅ Connected to MT5!")
                print(f"   Account: {account_info.login}")
                print(f"   Balance: ${account_info.balance:.2f}")
                print(f"   Server: {account_info.server}")
            else:
                print("   ⚠️ MT5 running but not logged in")
                print("   Please login to your XM account in MT5")
                all_good = False
            mt5.shutdown()
        else:
            print("   ❌ Could not connect to MT5")
            print("   Make sure MT5 is OPEN and logged in")
            all_good = False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        all_good = False
    print()
    
    # Summary
    print("=" * 50)
    if all_good:
        print("  ✅ ALL CHECKS PASSED!")
        print("  You can now run: start_auto_trader.bat")
    else:
        print("  ❌ Some checks failed. Fix issues above first.")
    print("=" * 50)
    
    return all_good

if __name__ == "__main__":
    check_setup()
    input("\nPress Enter to exit...")
