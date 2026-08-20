@echo off
setlocal
REM ============================================================
REM  Personal Knowledge Agent - start backend
REM  v2: auto-detect port 3001 to prevent Qdrant lock conflicts.
REM
REM  IMPORTANT: Keep this file in ASCII encoding. cmd parses
REM  the file before any chcp command takes effect, so non-ASCII
REM  bytes would be split into bogus tokens like "detects".
REM ============================================================
set PYTHONUTF8=1
cd /d %~dp0

set PORT_BUSY=0
netstat -ano | findstr ":3001" | findstr "LISTENING" >nul 2>&1
if %errorlevel%==0 set PORT_BUSY=1

if %PORT_BUSY%==1 (
  echo [WARN] Port 3001 is already in use - another instance is running.
  echo        Starting again would trigger Qdrant lock conflict.
  echo.
  echo   [1] Kill old instance and restart  (recommended)
  echo   [2] Start anyway  (may fail with lock error)
  choice /c 12 /n /m "Choose (1=clean restart, 2=start anyway): "
  if errorlevel 2 goto start
  echo Killing old instance on port 3001 ...
  for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":3001" ^| findstr "LISTENING"') do taskkill /f /t /pid %%p >nul 2>&1
  REM Also kill any leftover langgraph launcher process (and its child workers)
  for /f "tokens=2" %%p in ('tasklist ^| findstr /i "langgraph.exe"') do taskkill /f /t /pid %%p >nul 2>&1
  timeout /t 2 /nobreak >nul
  echo Cleaned. Starting fresh instance ...
)

:start
.venv\Scripts\python.exe -m langgraph_cli dev --port 3001
endlocal
