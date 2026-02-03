import asyncio
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode as TelegramParseMode
from telegram.ext import Application, CommandHandler, ContextTypes
import logging
import os
from datetime import datetime
import pytz

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def is_market_open(symbol: str) -> bool:
    # Forex market is open from Sunday 5pm EST to Friday 5pm EST
    # Crypto (BTCUSD, ETHUSD, etc.) is open 24/7
    symbol = symbol.upper()
    if symbol.endswith("USD") and symbol.startswith("BTC"):
        return True  # Crypto is always open

    # Check for Forex market hours (example: New York time)
    now_utc = datetime.utcnow()
    now_ny = now_utc.replace(tzinfo=pytz.utc).astimezone(pytz.timezone("America/New_York"))
    weekday = now_ny.weekday()
    hour = now_ny.hour

    # Forex opens Sunday 5pm (17:00) and closes Friday 5pm (17:00)
    if weekday == 6 and hour >= 17:  # Sunday after 5pm
        return True
    if 0 <= weekday <= 3:  # Monday to Thursday
        return True
    if weekday == 4 and hour < 17:  # Friday before 5pm
        return True
    return False

class TestTradingBot:
    def get_current_signals(self, symbol="EURUSD") -> list:
        if not is_market_open(symbol):
            return []  # Or return a message saying market is closed

        import time
        time.sleep(2)  # Simulate a short delay
        return [{
            'symbol': 'EURUSD',
            'type': 'BUY',
            'confidence': 0.85,
            'confirmations': 8,
            'pattern_confirmations': 3,
            'indicator_combinations': [
                {'type': 'Moving Averages', 'confidence': 0.9},
                {'type': 'RSI', 'confidence': 0.85},
                {'type': 'MACD', 'confidence': 0.8}
            ],
            'entry_price': 1.0850,
            'targets': {
                'take_profit_levels': [
                    {'price': 1.0900, 'level': '1.5R'},
                    {'price': 1.0950, 'level': '2.0R'},
                    {'price': 1.1000, 'level': '3.0R'}
                ],
                'stop_loss': 1.0800
            },
            'resistance_levels': {
                'r1': 1.0900,
                'r2': 1.0950,
                'r3': 1.1000
            },
            'risk_reward': 2.5,
            'potential_profit': 150.0,
            'max_loss': 50.0,
            'technical_summary': {
                'trend': 'Bullish',
                'volatility': 'Medium',
                'volume': 'High'
            }
        }]

    async def handle_signal(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            message = await update.message.reply_text("🔔 Getting latest trading signal...")
            try:
                symbol = "EURUSD"
                if context.args:
                    symbol = context.args[0].upper()
                signals = await asyncio.wait_for(
                    asyncio.to_thread(self.get_current_signals, symbol),
                    timeout=30
                )
            except asyncio.TimeoutError:
                await message.edit_text("❌ Signal request timed out. Please try again later.")
                return
            except Exception as e:
                logger.error(f"Error getting signals: {e}")
                await message.edit_text("❌ Error retrieving signals. Please try again later.")
                return
            if not signals:
                await message.edit_text("ℹ️ No high-probability signals available at the moment.")
                return
            for signal in signals:
                signal_message = (
                    f"🎯 ULTRA HIGH-PROBABILITY TRADING SIGNAL for {signal['symbol']} 🎯\n\n"
                    f"Signal Type: {signal['type']}\n"
                    f"Confidence: {signal['confidence']*100:.1f}%\n"
                    f"Indicator Confirmations: {signal['confirmations']}\n"
                    f"Pattern Confirmations: {signal['pattern_confirmations']}\n\n"
                    f"Indicator Combinations:\n" + "\n".join(
                        f"- {combo['type']} ({combo['confidence']*100:.1f}%)"
                        for combo in signal['indicator_combinations']
                    ) + "\n\n"
                    f"Price Targets:\n"
                    f"- Entry: {signal['entry_price']:.5f}\n"
                    f"- Take Profit 1: {signal['targets']['take_profit_levels'][0]['price']:.5f} ({signal['targets']['take_profit_levels'][0]['level']}R)\n"
                    f"- Take Profit 2: {signal['targets']['take_profit_levels'][1]['price']:.5f} ({signal['targets']['take_profit_levels'][1]['level']}R)\n"
                    f"- Take Profit 3: {signal['targets']['take_profit_levels'][2]['price']:.5f} ({signal['targets']['take_profit_levels'][2]['level']}R)\n"
                    f"- Stop Loss: {signal['targets']['stop_loss']:.5f}\n\n"
                    f"Resistance Levels:\n"
                    f"- R1: {signal['resistance_levels']['r1']:.5f}\n"
                    f"- R2: {signal['resistance_levels']['r2']:.5f}\n"
                    f"- R3: {signal['resistance_levels']['r3']:.5f}\n\n"
                    f"Profit Summary:\n"
                    f"- Risk:Reward Ratio: {signal['risk_reward']:.1f}\n"
                    f"- Potential Profit: {signal['potential_profit']:.1f} pips\n"
                    f"- Maximum Loss: {signal['max_loss']:.1f} pips\n\n"
                    f"Technical Summary:\n"
                    f"- Trend: {signal['technical_summary']['trend']}\n"
                    f"- Volatility: {signal['technical_summary']['volatility']}\n"
                    f"- Volume: {signal['technical_summary']['volume']}"
                )
                await message.edit_text(signal_message)
        except Exception as e:
            logger.error(f"Error handling signal command: {e}")
            try:
                await message.edit_text("❌ Error retrieving signals. Please try again later.")
            except:
                await update.message.reply_text("❌ Error retrieving signals. Please try again later.")

def main():
    application = Application.builder().token(os.getenv('TELEGRAM_BOT_TOKEN')).build()
    bot = TestTradingBot()
    application.add_handler(CommandHandler("signal", bot.handle_signal))
    print("Test bot started. Use /signal to test timeout and response.")
    application.run_polling()

if __name__ == "__main__":
    main() 