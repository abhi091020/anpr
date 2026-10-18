@echo off
cd /d "%~dp0"
echo Starting PPE ANPR Monitor on http://127.0.0.1:8010
start "" http://127.0.0.1:8010
.venv\Scripts\python.exe -m uvicorn app.server:app --host 127.0.0.1 --port 8010
pause
