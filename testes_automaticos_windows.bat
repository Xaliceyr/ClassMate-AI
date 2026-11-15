@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Ambiente nao encontrado. Execute setup_windows.bat primeiro.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m pytest -q
set "RESULT=%ERRORLEVEL%"
echo.
if "%RESULT%"=="0" (echo Testes automatizados aprovados.) else (echo Alguns testes falharam. Envie a mensagem de erro ao grupo.)
pause
exit /b %RESULT%
