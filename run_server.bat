@echo off
rem PromptMaker HTML test server launcher (Windows)
rem Usage: double-click this file. Optional: run_server.bat 8800  (port)
setlocal
cd /d "%~dp0"

set PORT=%1
if "%PORT%"=="" set PORT=8765

where python >nul 2>nul
if errorlevel 1 (
  where py >nul 2>nul
  if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.9+ from https://www.python.org/
    pause
    exit /b 1
  )
  set PY=py -3
) else (
  set PY=python
)

set USERDATA=
if exist "%~dp0user_data" set USERDATA=--user-data "%~dp0user_data"

echo PromptMaker test server: http://127.0.0.1:%PORT%
echo Close this window or press Ctrl+C to stop.
start "" "http://127.0.0.1:%PORT%"
%PY% -m promptmaker.server --port %PORT% %USERDATA%
pause
