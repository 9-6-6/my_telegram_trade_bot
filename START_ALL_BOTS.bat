@echo off
title Trading Bot System - Launcher
color 0A

echo.
echo  ╔═══════════════════════════════════════════════════════════╗
echo  ║           TRADING BOT SYSTEM - LAUNCHER                   ║
echo  ╠═══════════════════════════════════════════════════════════╣
echo  ║                                                           ║
echo  ║   This will start:                                        ║
echo  ║   1. Telegram Bot (sends signals to your phone)           ║
echo  ║   2. XM360 Auto Trader (places trades automatically)      ║
echo  ║                                                           ║
echo  ║   PREREQUISITES:                                          ║
echo  ║   - MT5 Desktop must be OPEN and logged in                ║
echo  ║   - Internet connection required                          ║
echo  ║                                                           ║
echo  ╚═══════════════════════════════════════════════════════════╝
echo.

echo Checking if MT5 is running...
tasklist /FI "IMAGENAME eq terminal64.exe" 2>NUL | find /I /N "terminal64.exe">NUL
if "%ERRORLEVEL%"=="0" (
    echo [OK] MT5 is running
) else (
    echo [WARNING] MT5 Desktop is NOT running!
    echo Please open MT5 and login to your XM360 account first.
    echo.
    pause
    exit /b
)

echo.
echo Starting bots...
echo.

cd /d "%~dp0"

echo [1/2] Starting Telegram Bot...
start "Telegram Bot" cmd /k "color 0B && echo TELEGRAM BOT && echo ============ && py -3.12 trading_bot.py"

timeout /t 3 /nobreak >nul

echo [2/2] Starting XM360 Auto Trader (v2.0 Enhanced)...
start "XM360 Auto Trader" cmd /k "color 0E && echo XM360 AUTO TRADER v2.0 && echo ================== && echo Features: Trailing Stop, Multiple TPs, Pending Orders && py -3.12 xm360_auto_trader\auto_trader_v2.py"

echo.
echo  ╔═══════════════════════════════════════════════════════════╗
echo  ║   ✅ BOTH BOTS STARTED SUCCESSFULLY!                      ║
echo  ╠═══════════════════════════════════════════════════════════╣
echo  ║                                                           ║
echo  ║   Two new windows opened:                                 ║
echo  ║   - Blue window: Telegram Bot                             ║
echo  ║   - Yellow window: XM360 Auto Trader                      ║
echo  ║                                                           ║
echo  ║   TO STOP: Press Ctrl+C in each window                    ║
echo  ║   OR close the windows                                    ║
echo  ║                                                           ║
echo  ╚═══════════════════════════════════════════════════════════╝
echo.

pause
