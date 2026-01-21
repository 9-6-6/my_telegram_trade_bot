@echo off
title Stop All Trading Bots
color 0C

echo.
echo  ╔═══════════════════════════════════════════════════════════╗
echo  ║           STOPPING ALL TRADING BOTS                       ║
echo  ╚═══════════════════════════════════════════════════════════╝
echo.

echo Stopping Python processes...

taskkill /F /IM python.exe /T 2>nul
taskkill /F /IM py.exe /T 2>nul

echo.
echo  ╔═══════════════════════════════════════════════════════════╗
echo  ║   ✅ ALL BOTS STOPPED                                     ║
echo  ║                                                           ║
echo  ║   Note: MT5 Desktop is still running (close manually      ║
echo  ║   if needed)                                              ║
echo  ╚═══════════════════════════════════════════════════════════╝
echo.

pause
