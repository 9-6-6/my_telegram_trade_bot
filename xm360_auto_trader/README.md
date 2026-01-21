# XM360 Auto Trader

Automatic trade execution system that receives signals from the Telegram Trading Bot and executes them on XM360 broker.

## Features
- Receives signals from the trading bot via shared signal queue
- Executes trades automatically on XM360 (MetaTrader 5)
- Supports both DEMO and LIVE accounts
- Risk management with configurable lot sizes
- Trade logging and history

## Requirements
- MetaTrader 5 installed and connected to XM360
- Python 3.10+
- MT5 Python library

## Setup

### 1. Install MetaTrader 5
Download and install MT5 from XM360's website.

### 2. Install Dependencies
```bash
pip install MetaTrader5
```

### 3. Configure Your Account
Edit `config.py` with your XM360 credentials:
- Account number
- Password
- Server name (e.g., "XMGlobal-MT5" for demo)

### 4. Run the Auto Trader
```bash
python auto_trader.py
```

## Configuration

| Setting | Description | Default |
|---------|-------------|---------|
| `ACCOUNT` | Your XM360 account number | Required |
| `PASSWORD` | Your MT5 password | Required |
| `SERVER` | MT5 server name | "XMGlobal-MT5" |
| `LOT_SIZE` | Default lot size per trade | 0.01 |
| `MAX_RISK_PERCENT` | Max risk per trade (%) | 1.0 |
| `USE_DEMO` | Use demo account | True |

## Signal Format
The auto trader expects signals in this format:
```python
{
    'symbol': 'XAUUSD',
    'direction': 'BUY' or 'SELL',
    'entry_price': 1920.50,
    'stop_loss': 1915.00,
    'take_profit': 1930.00,
    'lot_size': 0.01  # optional
}
```

## Safety Features
- Demo mode by default
- Maximum daily loss limit
- Position size limits
- Duplicate signal prevention
- Connection monitoring

## Files
- `auto_trader.py` - Main auto trading script
- `mt5_connector.py` - MetaTrader 5 connection handler
- `signal_receiver.py` - Receives signals from trading bot
- `trade_manager.py` - Trade execution and management
- `config.py` - Configuration settings
- `risk_manager.py` - Risk management logic

## Warning
⚠️ **Trading involves significant risk of loss. Use at your own risk.**
⚠️ **Always test on DEMO account first.**
