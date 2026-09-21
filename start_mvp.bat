@echo off
REM VAJRA MVP one-command launcher: venv -> deps -> tests -> server -> browser
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [1/4] creating venv ...
  python -m venv .venv
  if errorlevel 1 ( echo ERROR: need Python 3.10+ on PATH & exit /b 1 )
)
set PY=.venv\Scripts\python.exe

echo [2/4] installing requirements ...
%PY% -m pip install --quiet -r requirements.txt
if errorlevel 1 ( echo ERROR: pip install failed & exit /b 1 )

echo [3/4] running test suite ...
%PY% -m pytest tests\test_vajra.py -q
if errorlevel 1 ( echo ERROR: tests failed - fix before serving & exit /b 1 )

echo [4/4] starting VAJRA on http://127.0.0.1:8000 ...
start "" "http://127.0.0.1:8000"
%PY% -m vajra.api.server --port 8000
