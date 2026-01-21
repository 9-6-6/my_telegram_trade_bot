@echo off
title XM360 Auto Trader
echo =========================================
echo    XM360 Auto Trader Starting...
echo =========================================
echo.

cd /d "%~dp0"

echo Checking MT5 connection...
echo.

py -3.12 auto_trader.py

echo.
echo Auto Trader stopped. Press any key to exit...
pause >nul
