@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Ambiente nao encontrado. Execute setup_windows.bat primeiro.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m streamlit run app.py --server.port 8502
pause
