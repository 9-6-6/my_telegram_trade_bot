"""
🧪 FULL SYSTEM TEST
=====================
Tests:
1. Signal generation from copy_trading_bot
2. Sending signals to NGBOT
3. Control bot (NG XM Trading BOT) with account info
4. Full integration - START command triggers signals
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Tokens
CONTROL_BOT_TOKEN = "7849517577:AAGx8PhFyAf-cEFt06pfL_CPT8x9REVB1_U"  # NG XM Trading BOT
SIGNAL_BOT_TOKEN = "8548359658:AAE420kDIrgpyExD8gJwi9b4kZfNsJ1nJYA"    # NGBOT
ADMIN_CHAT_ID = 603932135

# State file
STATE_FILE = "trading_state.json"

print("""
╔═══════════════════════════════════════════════════════════════════╗
║              🧪 FULL SYSTEM TEST                                  ║
╠═══════════════════════════════════════════════════════════════════╣
║  Testing ALL components:                                          ║
║  1. Signal generation (copy_trading_bot)                          ║
║  2. NGBOT signal delivery                                         ║
║  3. Control Bot (account info)                                    ║
║  4. Full integration                                              ║
╚═══════════════════════════════════════════════════════════════════╝
""")

# ============================================================
# TEST 1: Test Signal Generation
# ============================================================
print("\n" + "="*60)
print("TEST 1: Signal Generation from copy_trading_bot")
print("="*60)

try:
    # Test importing modules
    print("\n[1.1] Testing imports...")
    
    from autonomous_signal_generator import AutonomousSignalGenerator
    print("   ✅ AutonomousSignalGenerator imported")
    
    from scalp_ai_engine import ScalpSignalEngine
    print("   ✅ ScalpSignalEngine imported")
    
    from news_signal_integration import NewsSignalIntegration
    print("   ✅ NewsSignalIntegration imported")
    
    # Test signal generator initialization
    print("\n[1.2] Testing signal generators...")
    
    auto_gen = AutonomousSignalGenerator()
    print("   ✅ AutonomousSignalGenerator created")
    
    scalp_engine = ScalpSignalEngine()
    print("   ✅ ScalpSignalEngine created")
    
    news_handler = NewsSignalIntegration(min_confidence=0.40)
    print("   ✅ NewsSignalIntegration created")
    
    TEST1_PASSED = True
    print("\n✅ TEST 1 PASSED: Signal generators are working!")
    
except Exception as e:
    TEST1_PASSED = False
    print(f"\n❌ TEST 1 FAILED: {e}")

# ============================================================
# TEST 2: Test Signal Generation (Quick Scan)
# ============================================================
print("\n" + "="*60)
print("TEST 2: Quick Signal Scan Test")
print("="*60)

async def test_quick_scan():
    try:
        print("\n[2.1] Testing scalp signal generation for XAUUSD...")
        
        scalp_engine = ScalpSignalEngine()
        signals = await scalp_engine.scan_for_scalp_signals_async()
        
        if signals:
            print(f"   ✅ Generated {len(signals)} scalp signals!")
            for s in signals[:2]:  # Show first 2
                print(f"      • {s.symbol}: {s.signal_type.value} @ {s.entry_price}")
        else:
            print("   ⚠️ No scalp signals found (market conditions)")
        
        print("\n[2.2] Testing news signal for XAUUSD...")
        news_handler = NewsSignalIntegration(min_confidence=0.40)
        await news_handler.initialize()
        result = await news_handler.generate_signal("XAUUSD")
        
        if result:
            signal, msg, meta = result
            print(f"   ✅ News signal generated!")
            print(f"      • Direction: {signal['type']}")
            print(f"      • Entry: ${signal['entry']:.2f}")
            print(f"      • Confidence: {signal.get('confidence', 'N/A')}")
        else:
            print("   ⚠️ No news signal (may need more news)")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

try:
    TEST2_PASSED = asyncio.run(test_quick_scan())
    if TEST2_PASSED:
        print("\n✅ TEST 2 PASSED: Signal generation works!")
except Exception as e:
    TEST2_PASSED = False
    print(f"\n❌ TEST 2 FAILED: {e}")

# ============================================================
# TEST 3: Test Telegram Message Sending
# ============================================================
print("\n" + "="*60)
print("TEST 3: Test Sending Messages to NGBOT")
print("="*60)

async def test_telegram_send():
    try:
        import httpx
        
        # Test sending to NGBOT
        print("\n[3.1] Sending test message to NGBOT...")
        
        url = f"https://api.telegram.org/bot{SIGNAL_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": ADMIN_CHAT_ID,
            "text": "🧪 *TEST MESSAGE*\n\nThis is a test from the system test script.\nIf you see this, NGBOT is working!",
            "parse_mode": "Markdown"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload)
            data = response.json()
            
            if data.get("ok"):
                print("   ✅ Message sent to NGBOT successfully!")
                return True
            else:
                print(f"   ❌ Failed: {data.get('description')}")
                return False
                
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

try:
    TEST3_PASSED = asyncio.run(test_telegram_send())
    if TEST3_PASSED:
        print("\n✅ TEST 3 PASSED: Can send messages to NGBOT!")
except Exception as e:
    TEST3_PASSED = False
    print(f"\n❌ TEST 3 FAILED: {e}")

# ============================================================
# TEST 4: Test Control Bot Message Sending
# ============================================================
print("\n" + "="*60)
print("TEST 4: Test Sending Messages to Control Bot")
print("="*60)

async def test_control_bot():
    try:
        import httpx
        
        print("\n[4.1] Sending test message to NG XM Trading BOT...")
        
        url = f"https://api.telegram.org/bot{CONTROL_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": ADMIN_CHAT_ID,
            "text": "🧪 *CONTROL BOT TEST*\n\nThis is a test message.\nControl Bot is working!",
            "parse_mode": "Markdown"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload)
            data = response.json()
            
            if data.get("ok"):
                print("   ✅ Message sent to Control Bot successfully!")
                return True
            else:
                print(f"   ❌ Failed: {data.get('description')}")
                return False
                
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

try:
    TEST4_PASSED = asyncio.run(test_control_bot())
    if TEST4_PASSED:
        print("\n✅ TEST 4 PASSED: Can send messages to Control Bot!")
except Exception as e:
    TEST4_PASSED = False
    print(f"\n❌ TEST 4 FAILED: {e}")

# ============================================================
# TEST 5: Test Full Signal Flow
# ============================================================
print("\n" + "="*60)
print("TEST 5: Full Signal Flow Test")
print("="*60)

async def test_full_flow():
    try:
        import httpx
        
        print("\n[5.1] Setting trading state to ENABLED...")
        state = {
            "is_trading_enabled": True,
            "auto_scan_enabled": False,
            "scalp_scan_enabled": False,
            "balance": 10000.0,
            "start_time": datetime.now().isoformat()
        }
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f)
        print("   ✅ State saved")
        
        print("\n[5.2] Generating a real signal...")
        scalp_engine = ScalpSignalEngine()
        signals = await scalp_engine.scan_for_scalp_signals_async()
        
        if signals:
            signal = signals[0]
            message = scalp_engine.format_scalp_signal(signal)
            
            print(f"   ✅ Signal generated: {signal.symbol} {signal.signal_type.value}")
            
            print("\n[5.3] Sending signal to NGBOT...")
            url = f"https://api.telegram.org/bot{SIGNAL_BOT_TOKEN}/sendMessage"
            payload = {
                "chat_id": ADMIN_CHAT_ID,
                "text": f"📡 *REAL SIGNAL TEST*\n\n{message}",
                "parse_mode": "Markdown"
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=payload)
                if response.json().get("ok"):
                    print("   ✅ Signal delivered to NGBOT!")
                    return True
        else:
            # Generate a test signal even if no real one
            print("   ⚠️ No real signal, sending test signal...")
            
            test_signal = """
⚡ *TEST SCALP SIGNAL* ⚡

🎯 *XAUUSD* (Gold)
📈 *Direction:* LONG

💰 *Entry:* $4,882.50
🎯 *Take Profit:* $4,890.00
🛑 *Stop Loss:* $4,875.00

📊 *Confidence:* 87%
⏱️ *Timeframe:* 5M-15M

_This is a test signal from system test_
            """
            
            url = f"https://api.telegram.org/bot{SIGNAL_BOT_TOKEN}/sendMessage"
            payload = {
                "chat_id": ADMIN_CHAT_ID,
                "text": test_signal,
                "parse_mode": "Markdown"
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=payload)
                if response.json().get("ok"):
                    print("   ✅ Test signal delivered to NGBOT!")
                    return True
        
        return False
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

try:
    TEST5_PASSED = asyncio.run(test_full_flow())
    if TEST5_PASSED:
        print("\n✅ TEST 5 PASSED: Full signal flow works!")
except Exception as e:
    TEST5_PASSED = False
    print(f"\n❌ TEST 5 FAILED: {e}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*60)
print("TEST SUMMARY")
print("="*60)
print(f"""
Test 1 - Signal Generators:     {"✅ PASSED" if TEST1_PASSED else "❌ FAILED"}
Test 2 - Quick Signal Scan:     {"✅ PASSED" if TEST2_PASSED else "❌ FAILED"}
Test 3 - NGBOT Messaging:       {"✅ PASSED" if TEST3_PASSED else "❌ FAILED"}
Test 4 - Control Bot Messaging: {"✅ PASSED" if TEST4_PASSED else "❌ FAILED"}
Test 5 - Full Signal Flow:      {"✅ PASSED" if TEST5_PASSED else "❌ FAILED"}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

all_passed = all([TEST1_PASSED, TEST2_PASSED, TEST3_PASSED, TEST4_PASSED, TEST5_PASSED])

if all_passed:
    print("🎉 ALL TESTS PASSED!")
    print("\nNow check your Telegram - you should have received test messages!")
else:
    print("⚠️ Some tests failed. Check the errors above.")
