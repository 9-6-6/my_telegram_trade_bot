import os
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from datetime import datetime
import pytz

class TradingBot:
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        print("/start command received")
        await update.message.reply_text("✅ Main bot is alive! Send /signal BTCUSD to test.")

    def is_market_open(self, symbol: str) -> bool:
        symbol = symbol.upper()
        if symbol.endswith("USD") and symbol.startswith("BTC"):
            return True
        now_utc = datetime.utcnow()
        now_ny = now_utc.replace(tzinfo=pytz.utc).astimezone(pytz.timezone("America/New_York"))
        weekday = now_ny.weekday()
        hour = now_ny.hour
        if weekday == 6 and hour >= 17:
            return True
        if 0 <= weekday <= 3:
            return True
        if weekday == 4 and hour < 17:
            return True
        return False

    async def signal(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        print("/signal command received")
        symbol = "XAUUSD"
        if context.args:
            symbol = context.args[0].upper()
        if not self.is_market_open(symbol):
            await update.message.reply_text(f"ℹ️ The market for {symbol} is currently closed. Try BTCUSD or another crypto for 24/7 signals.")
            return
        await update.message.reply_text(f"Test signal for {symbol}: BUY @ 70000, SL: 69000, TP: 72000")

def main():
    os.environ['TZ'] = 'UTC'
    application = Application.builder().token(os.getenv('TELEGRAM_BOT_TOKEN')).build()
    bot = TradingBot()
    application.add_handler(CommandHandler("start", bot.start))
    application.add_handler(CommandHandler("signal", bot.signal))
    print("Main bot started. Use /start and /signal BTCUSD in Telegram.")
    application.run_polling()

if __name__ == "__main__":
    main() 