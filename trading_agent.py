import os
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from newsapi import NewsApiClient
import logging
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TradingAgent:
    def __init__(self, api_key: str, api_secret: str, default_lot_size: float = 0.01):
        """
        Initialize the trading agent with API credentials and default settings.
        
        Args:
            api_key (str): Exness API key
            api_secret (str): Exness API secret
            default_lot_size (float): Default lot size for trades (default: 0.01)
        """
        self.exchange = ccxt.exness({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True
        })
        
        self.default_lot_size = default_lot_size
        self.newsapi = NewsApiClient(api_key=os.getenv('NEWS_API_KEY'))
        self.active_trades = {}
        
    def analyze_market_trend(self, symbol: str, timeframe: str = '1h') -> Dict:
        """
        Analyze market trend using technical indicators.
        
        Args:
            symbol (str): Trading pair symbol (e.g., 'EUR/USD')
            timeframe (str): Timeframe for analysis
            
        Returns:
            Dict: Analysis results including trend direction and confidence
        """
        try:
            # Fetch OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=100)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Calculate technical indicators
            df['SMA20'] = df['close'].rolling(window=20).mean()
            df['SMA50'] = df['close'].rolling(window=50).mean()
            df['RSI'] = self._calculate_rsi(df['close'])
            
            # Determine trend
            current_price = df['close'].iloc[-1]
            sma20 = df['SMA20'].iloc[-1]
            sma50 = df['SMA50'].iloc[-1]
            rsi = df['RSI'].iloc[-1]
            
            trend = {
                'direction': 'neutral',
                'confidence': 0.0,
                'indicators': {
                    'price': current_price,
                    'sma20': sma20,
                    'sma50': sma50,
                    'rsi': rsi
                }
            }
            
            # Simple trend analysis
            if current_price > sma20 and sma20 > sma50:
                trend['direction'] = 'bullish'
                trend['confidence'] = 0.7
            elif current_price < sma20 and sma20 < sma50:
                trend['direction'] = 'bearish'
                trend['confidence'] = 0.7
                
            return trend
            
        except Exception as e:
            logger.error(f"Error analyzing market trend: {str(e)}")
            return {'direction': 'neutral', 'confidence': 0.0, 'error': str(e)}
    
    def analyze_news_sentiment(self, symbol: str) -> Dict:
        """
        Analyze news sentiment for the given trading pair.
        
        Args:
            symbol (str): Trading pair symbol
            
        Returns:
            Dict: News sentiment analysis results
        """
        try:
            # Extract base and quote currencies
            base, quote = symbol.split('/')
            
            # Fetch relevant news
            news = self.newsapi.get_everything(
                q=f"{base} {quote} forex",
                language='en',
                sort_by='relevancy',
                page_size=10
            )
            
            # Simple sentiment analysis (to be enhanced with proper NLP)
            sentiment_score = 0
            for article in news['articles']:
                # Basic keyword-based sentiment
                positive_words = ['up', 'rise', 'gain', 'positive', 'growth']
                negative_words = ['down', 'fall', 'loss', 'negative', 'decline']
                
                title = article['title'].lower()
                description = article['description'].lower() if article['description'] else ''
                
                for word in positive_words:
                    if word in title or word in description:
                        sentiment_score += 1
                for word in negative_words:
                    if word in title or word in description:
                        sentiment_score -= 1
            
            return {
                'sentiment_score': sentiment_score,
                'articles_analyzed': len(news['articles']),
                'confidence': min(abs(sentiment_score) / 10, 1.0)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing news sentiment: {str(e)}")
            return {'sentiment_score': 0, 'confidence': 0.0, 'error': str(e)}
    
    def execute_trade(self, symbol: str, side: str, lot_size: Optional[float] = None) -> Dict:
        """
        Execute a trade with the specified parameters.
        
        Args:
            symbol (str): Trading pair symbol
            side (str): 'buy' or 'sell'
            lot_size (float, optional): Lot size for the trade
            
        Returns:
            Dict: Trade execution results
        """
        try:
            if lot_size is None:
                lot_size = self.default_lot_size
                
            # Validate lot size
            if lot_size < 0.01:
                logger.warning("Lot size too small, using minimum lot size of 0.01")
                lot_size = 0.01
                
            # Execute the trade
            order = self.exchange.create_order(
                symbol=symbol,
                type='market',
                side=side,
                amount=lot_size
            )
            
            self.active_trades[order['id']] = {
                'symbol': symbol,
                'side': side,
                'lot_size': lot_size,
                'entry_price': order['price'],
                'timestamp': datetime.now().isoformat()
            }
            
            return {
                'status': 'success',
                'order_id': order['id'],
                'details': self.active_trades[order['id']]
            }
            
        except Exception as e:
            logger.error(f"Error executing trade: {str(e)}")
            return {'status': 'error', 'error': str(e)}
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            prices (pd.Series): Price series
            period (int): RSI period
            
        Returns:
            pd.Series: RSI values
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def get_trading_signal(self, symbol: str) -> Dict:
        """
        Get comprehensive trading signal combining technical and fundamental analysis.
        
        Args:
            symbol (str): Trading pair symbol
            
        Returns:
            Dict: Trading signal with recommendations
        """
        market_trend = self.analyze_market_trend(symbol)
        news_sentiment = self.analyze_news_sentiment(symbol)
        
        # Combine signals
        signal = {
            'action': 'hold',
            'confidence': 0.0,
            'reason': '',
            'market_trend': market_trend,
            'news_sentiment': news_sentiment
        }
        
        # Decision making logic
        if market_trend['direction'] == 'bullish' and news_sentiment['sentiment_score'] > 0:
            signal['action'] = 'buy'
            signal['confidence'] = (market_trend['confidence'] + news_sentiment['confidence']) / 2
            signal['reason'] = 'Positive market trend and news sentiment'
        elif market_trend['direction'] == 'bearish' and news_sentiment['sentiment_score'] < 0:
            signal['action'] = 'sell'
            signal['confidence'] = (market_trend['confidence'] + news_sentiment['confidence']) / 2
            signal['reason'] = 'Negative market trend and news sentiment'
            
        return signal 