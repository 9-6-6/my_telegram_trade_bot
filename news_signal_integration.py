"""
News Signal Integration for Trading Bot
========================================
This module integrates the NewsSignalEngine with the main Telegram bot.
Import and use this in your main bot file.
"""

import asyncio
import logging
from typing import Optional, Dict, List
from datetime import datetime
import pytz

from news_signal_engine import (
    NewsSignalEngine, 
    EnhancedSignal, 
    SignalType,
    NewsImpact
)

logger = logging.getLogger(__name__)


class NewsSignalIntegration:
    """
    Integration class to connect NewsSignalEngine with the main trading bot.
    """
    
    def __init__(self, min_confidence: float = 0.55):
        """
        Initialize the integration.
        
        Args:
            min_confidence: Minimum confidence threshold for signals (0.0 to 1.0)
        """
        self.engine = NewsSignalEngine(min_confidence=min_confidence)
        self.last_signals: Dict[str, EnhancedSignal] = {}
        self.signal_cooldown: Dict[str, datetime] = {}
        self.cooldown_minutes = 30  # Don't send same signal within 30 minutes

    async def close(self):
        """Cleanup resources"""
        await self.engine.close()

    def _is_on_cooldown(self, symbol: str) -> bool:
        """Check if a symbol is on signal cooldown"""
        if symbol not in self.signal_cooldown:
            return False
        
        last_time = self.signal_cooldown[symbol]
        now = datetime.now(pytz.UTC)
        minutes_passed = (now - last_time).total_seconds() / 60
        
        return minutes_passed < self.cooldown_minutes

    def _set_cooldown(self, symbol: str):
        """Set cooldown for a symbol"""
        self.signal_cooldown[symbol] = datetime.now(pytz.UTC)

    async def get_signal_for_symbol(self, symbol: str) -> Optional[EnhancedSignal]:
        """
        Get a trading signal for a specific symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'XAUUSD', 'EURUSD')
            
        Returns:
            EnhancedSignal if a valid signal is found, None otherwise
        """
        # Check cooldown
        if self._is_on_cooldown(symbol):
            logger.info(f"Symbol {symbol} is on cooldown")
            return None
        
        try:
            signal = await self.engine.generate_signal(symbol)
            
            if signal:
                self.last_signals[symbol] = signal
                self._set_cooldown(symbol)
                logger.info(f"Generated signal for {symbol}: {signal.signal_type.value}")
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return None

    def format_telegram_message(self, signal: EnhancedSignal) -> str:
        """
        Format signal for Telegram with proper formatting.
        
        Args:
            signal: The enhanced signal to format
            
        Returns:
            Formatted message string for Telegram
        """
        # Emoji based on signal type
        if signal.signal_type in [SignalType.STRONG_BUY, SignalType.BUY]:
            emoji = "🟢"
            direction = "📈"
        else:
            emoji = "🔴"
            direction = "📉"
        
        # Build message
        msg = f"""
{emoji} *{signal.signal_type.value} SIGNAL* {emoji}
━━━━━━━━━━━━━━━━━━━━━━

{direction} *Symbol:* `{signal.symbol}`
⏰ *Time:* {signal.timestamp.strftime('%Y-%m-%d %H:%M')} UTC

💹 *Entry Price:* `{signal.entry_price:.5f}`
🛑 *Stop Loss:* `{signal.stop_loss:.5f}`
🎯 *Take Profit Levels:*
"""
        for tp in signal.take_profit_levels:
            msg += f"   • {tp['level']}: `{tp['price']:.5f}`\n"
        
        msg += f"""
📊 *Analysis Metrics:*
   • Confidence: `{signal.confidence:.1%}`
   • Risk/Reward: `{signal.risk_reward_ratio:.2f}`
   • News Sentiment: `{signal.news_sentiment:+.2f}`
   • Technical Score: `{signal.technical_score:.2f}`

📰 *News Analysis:*
   • Total News: {signal.news_analysis.get('total_news', 0)}
   • Bullish: {signal.news_analysis.get('bullish_news', 0)}
   • Bearish: {signal.news_analysis.get('bearish_news', 0)}
   • High Impact: {signal.news_analysis.get('high_impact', 0)}
"""
        
        # Add headlines if available
        headlines = signal.news_analysis.get('headlines', [])
        if headlines:
            msg += "\n📰 *Recent Headlines:*\n"
            for h in headlines[:3]:
                sent_emoji = "🟢" if h['sentiment'] > 0 else "🔴" if h['sentiment'] < 0 else "⚪"
                title = h['title'][:50] + "..." if len(h['title']) > 50 else h['title']
                msg += f"   {sent_emoji} {title}\n"
        
        msg += "\n⚠️ _Always use proper risk management!_"
        
        return msg

    def get_signal_summary(self, signal: EnhancedSignal) -> Dict:
        """
        Get a dictionary summary of the signal for logging/storage.
        
        Args:
            signal: The enhanced signal
            
        Returns:
            Dictionary with signal details
        """
        return {
            'symbol': signal.symbol,
            'type': signal.signal_type.value,
            'entry': signal.entry_price,
            'stop_loss': signal.stop_loss,
            'take_profits': signal.take_profit_levels,
            'confidence': signal.confidence,
            'news_sentiment': signal.news_sentiment,
            'technical_score': signal.technical_score,
            'risk_reward': signal.risk_reward_ratio,
            'timestamp': signal.timestamp.isoformat(),
            'reasoning': signal.reasoning
        }


# Example usage for the main bot
async def example_bot_integration():
    """
    Example of how to integrate with the main bot.
    Copy this pattern into your main bot file.
    """
    integration = NewsSignalIntegration(min_confidence=0.55)
    
    try:
        # This would be called from your /signal command handler
        symbols = ['XAUUSD', 'EURUSD', 'BTCUSD']
        
        for symbol in symbols:
            print(f"\n{'='*50}")
            print(f"Getting signal for {symbol}...")
            
            signal = await integration.get_signal_for_symbol(symbol)
            
            if signal:
                message = integration.format_telegram_message(signal)
                print(message)
            else:
                print(f"No signal available for {symbol}")
                
    finally:
        await integration.close()


# Integration function for copy_trading_bot.py
def create_enhanced_signal_handler():
    """
    Creates an enhanced signal handler that can be imported into the main bot.
    
    Usage in copy_trading_bot.py:
    
    from news_signal_integration import create_enhanced_signal_handler
    
    # In your TradingBot class __init__:
    self.news_signal_handler = create_enhanced_signal_handler()
    
    # In your signal command:
    async def signal(self, update, context):
        symbol = context.args[0] if context.args else 'XAUUSD'
        signal = await self.news_signal_handler.get_signal_for_symbol(symbol)
        if signal:
            message = self.news_signal_handler.format_telegram_message(signal)
            await update.message.reply_text(message, parse_mode='Markdown')
    """
    return NewsSignalIntegration(min_confidence=0.55)


if __name__ == "__main__":
    print("\n" + "="*60)
    print("📊 NEWS SIGNAL INTEGRATION TEST")
    print("="*60)
    
    asyncio.run(example_bot_integration())
