@echo off
rem ─────────────────────────────────────────────────────────────
rem  Binance Trader - quick start
rem    start.cmd                 start in sim mode on 127.0.0.1:8899
rem    start.cmd live            start in live mode (testnet in this env)
rem    start.cmd status          is it running?
rem    start.cmd stop            stop the instance on the port
rem    start.cmd --port 8900     use another port
rem  If it is already running this only opens the browser.
rem  Logs: run_logs\start.log
rem ─────────────────────────────────────────────────────────────
setlocal EnableExtensions
set "ROOT=%~dp0"
cd /d "%ROOT%"

set "PORT=8899"
set "MODE=sim"
set "OPEN_BROWSER=1"

:parse
if "%~1"=="" goto :parsed
if /I "%~1"=="status" goto :status
if /I "%~1"=="stop"   goto :stop
if /I "%~1"=="sim"      ( set "MODE=sim"      & shift & goto :parse )
if /I "%~1"=="live"     ( set "MODE=live"     & shift & goto :parse )
if /I "%~1"=="backtest" ( set "MODE=backtest" & shift & goto :parse )
if /I "%~1"=="--port"   ( set "PORT=%~2"      & shift & shift & goto :parse )
if /I "%~1"=="--no-browser" ( set "OPEN_BROWSER=0" & shift & goto :parse )
echo [x] Unknown argument: %~1
echo     usage: start.cmd [sim^|live^|backtest] [--port N^] [--no-browser^] ^| status ^| stop
exit /b 2
:parsed

call :find_pid
if defined PID (
  echo [i] Already running on port %PORT% ^(PID %PID%^) - opening browser.
  if "%OPEN_BROWSER%"=="1" start "" "http://127.0.0.1:%PORT%"
  exit /b 0
)

set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
if not exist "run_logs" mkdir "run_logs"

echo [i] Starting %MODE% mode on http://127.0.0.1:%PORT% ...
start "Binance Trader" /min cmd /c "%PY% -m app.main --mode %MODE% --port %PORT% >> run_logs\start.log 2>&1"

rem with --no-browser there is nothing to open, so return immediately
if "%OPEN_BROWSER%"=="0" (
  echo [i] Spawned; log: run_logs\start.log
  exit /b 0
)

rem wait up to ~45s for /health, then open the browser
for /L %%i in (1,1,45) do (
  curl -s -m 2 -o nul "http://127.0.0.1:%PORT%/health" >nul 2>&1
  if not errorlevel 1 goto :up
  ping -n 2 127.0.0.1 >nul
)
echo [x] No healthy response after ~45s. See run_logs\start.log
exit /b 1

:up
echo [ok] Up: http://127.0.0.1:%PORT%
if "%OPEN_BROWSER%"=="1" start "" "http://127.0.0.1:%PORT%"
exit /b 0

:status
call :find_pid
if defined PID (
  echo [ok] running - PID %PID% - http://127.0.0.1:%PORT%
  exit /b 0
)
echo [i] not running
exit /b 1

:stop
call :find_pid
if not defined PID (
  echo [i] not running
  exit /b 0
)
taskkill /PID %PID% /T /F >nul 2>&1
echo [ok] stopped PID %PID%
exit /b 0

:find_pid
set "PID="
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":%PORT%" ^| findstr "LISTENING"') do set "PID=%%p"
exit /b 0
