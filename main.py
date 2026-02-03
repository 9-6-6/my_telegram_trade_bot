import os
from dotenv import load_dotenv
from trading_agent import TradingAgent
import logging
import time
from typing import Dict, List

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_user_confirmation(message: str) -> bool:
    """Get user confirmation for trading decisions."""
    while True:
        response = input(f"{message} (yes/no): ").lower()
        if response in ['yes', 'y']:
            return True
        elif response in ['no', 'n']:
            return False
        print("Please enter 'yes' or 'no'")

def main():
    # Initialize trading agent
    api_key = os.getenv('EXNESS_API_KEY')
    api_secret = os.getenv('EXNESS_API_SECRET')
    
    if not api_key or not api_secret:
        logger.error("API credentials not found in environment variables")
        return
    
    # Get default lot size from user
    while True:
        try:
            default_lot_size = float(input("Enter default lot size (minimum 0.01): "))
            if default_lot_size >= 0.01:
                break
            print("Lot size must be at least 0.01")
        except ValueError:
            print("Please enter a valid number")
    
    agent = TradingAgent(api_key, api_secret, default_lot_size)
    
    # Get trading pairs from user
    print("\nEnter trading pairs to monitor (e.g., EUR/USD, GBP/USD)")
    print("Press Enter twice when done")
    trading_pairs = []
    while True:
        pair = input("Enter trading pair: ").strip().upper()
        if not pair:
            break
        if '/' not in pair:
            print("Invalid format. Please use format like EUR/USD")
            continue
        trading_pairs.append(pair)
    
    if not trading_pairs:
        logger.error("No trading pairs specified")
        return
    
    print("\nTrading agent started. Monitoring pairs:", trading_pairs)
    print("Press Ctrl+C to stop")
    
    try:
        while True:
            for symbol in trading_pairs:
                logger.info(f"Analyzing {symbol}...")
                
                # Get trading signal
                signal = agent.get_trading_signal(symbol)
                
                if signal['action'] != 'hold':
                    # Format the analysis for user
                    analysis = f"""
                    Trading Signal for {symbol}:
                    Action: {signal['action'].upper()}
                    Confidence: {signal['confidence']:.2f}
                    Reason: {signal['reason']}
                    
                    Market Trend:
                    - Direction: {signal['market_trend']['direction']}
                    - Confidence: {signal['market_trend']['confidence']:.2f}
                    
                    News Sentiment:
                    - Score: {signal['news_sentiment']['sentiment_score']}
                    - Confidence: {signal['news_sentiment']['confidence']:.2f}
                    """
                    
                    print(analysis)
                    
                    # Ask for user confirmation
                    if get_user_confirmation(f"Do you want to {signal['action']} {symbol}?"):
                        # Ask for lot size
                        while True:
                            try:
                                lot_size = float(input(f"Enter lot size (default: {agent.default_lot_size}): ") or agent.default_lot_size)
                                if lot_size >= 0.01:
                                    break
                                print("Lot size must be at least 0.01")
                            except ValueError:
                                print("Please enter a valid number")
                        
                        # Execute trade
                        result = agent.execute_trade(symbol, signal['action'], lot_size)
                        if result['status'] == 'success':
                            logger.info(f"Trade executed successfully: {result['details']}")
                        else:
                            logger.error(f"Trade execution failed: {result['error']}")
                
                # Wait before next analysis
                time.sleep(5)
            
            # Wait before next round of analysis
            time.sleep(60)
            
    except KeyboardInterrupt:
        print("\nTrading agent stopped by user")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 