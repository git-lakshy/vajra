@echo off
REM VAJRA MVP one-command launcher: venv -> deps -> tests -> server -> browser
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [1/5] creating venv ...
  python -m venv .venv
  if errorlevel 1 ( echo ERROR: need Python 3.10+ on PATH & exit /b 1 )
)
set PY=.venv\Scripts\python.exe

echo [2/5] installing requirements ...
%PY% -m pip install --quiet -r requirements.txt
if errorlevel 1 ( echo ERROR: pip install failed & exit /b 1 )

echo [3/5] running test suite ...
%PY% -m pytest tests\ -q
if errorlevel 1 ( echo ERROR: tests failed - fix before serving & exit /b 1 )

echo [4/5] starting VAJRA server in a separate window (KEEP IT OPEN) ...
start "VAJRA server - DO NOT CLOSE" %PY% -m vajra.api.server --port 8000

echo [5/5] waiting for http://127.0.0.1:8000 to respond ...
powershell -NoProfile -Command "$t=0; while ($t -lt 150) { try { $r=Invoke-WebRequest 'http://127.0.0.1:8000/api/state' -TimeoutSec 3 -UseBasicParsing; if ($r.StatusCode -eq 200) { Write-Output 'server is up'; exit 0 } } catch {}; Start-Sleep 3; $t+=3 }; Write-Output 'TIMEOUT'; exit 1"
if errorlevel 1 ( echo ERROR: server did not respond - check the "VAJRA server" window for errors & exit /b 1 )

echo Opening dashboard ...
start "" "http://127.0.0.1:8000"
echo Done. VAJRA is live. Keep the "VAJRA server" window open - closing it stops the demo.
