"""
Quick test to verify the news signal integration works
"""
import asyncio
import sys
import io

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from news_signal_integration import NewsSignalIntegration

async def main():
    print("=" * 60)
    print("TESTING NEWS SIGNAL INTEGRATION")
    print("=" * 60)
    
    handler = NewsSignalIntegration(min_confidence=0.35)
    
    symbols = ['XAUUSD', 'EURUSD', 'GBPUSD', 'BTCUSD']
    
    for symbol in symbols:
        print(f"\n{'='*50}")
        print(f"Testing {symbol}...")
        result = await handler.get_signal_for_symbol(symbol)
        
        if result:
            print(f"[OK] {symbol}: {result.signal_type.value}")
            print(f"   Confidence: {result.confidence:.1%}")
            print(f"   Entry: {result.entry_price}")
            print(f"   SL: {result.stop_loss}")
        else:
            print(f"[--] {symbol}: No signal generated")
    
    await handler.close()
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
