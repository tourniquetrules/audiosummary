@echo off
SET VENV_PATH=%~dp0.venv

if not exist "%VENV_PATH%" (
    echo Virtual environment not found at %VENV_PATH%
    echo Please make sure you have created the environment using:
    echo py -3.12 -m venv .venv
    pause
    exit /b
)

echo Activating virtual environment...
call "%VENV_PATH%\Scripts\activate.bat"

echo Starting Audio Summary Pipeline...
python -m src.main

pause
