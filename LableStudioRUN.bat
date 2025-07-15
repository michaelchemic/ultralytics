@echo off
REM Script：Open Anaconda run Label Studio
REM Date：2025-07-15

set CONDA_ROOT=D:\ProgramData\anaconda3
set ENV_NAME=label_sudio
set LABEL_STUDIO_PORT=8080

REM Check Anaconda installation path 
if not exist %CONDA_ROOT% (
    echo Error: Anaconda installation path not found.
    echo Please modify the CONDA_ROOT variable in the script to your Anaconda installation path.
    pause
    exit /b 1
)

REM Activate the Anaconda environment.
call "%CONDA_ROOT%\Scripts\activate.bat" %ENV_NAME%

if %errorlevel% neq 0 (
    echo Error: Unable to activate virtual environment "%ENV_NAME%"
    echo Please confirm if the environment name is correct, or use the following command to view all environments:
    echo conda env list
    pause
    exit /b 1
)

REM Check if Label Studio is installed
where label-studio >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Label Studio installation not detected
    echo Please run pip install label-studio
    pause
    exit /b 1
)

REM Start Label Studio
echo Label Studio is starting up...
echo Browser access:http://localhost%LABEL_STUDIO_PORT%
start "" "label-studio" --port %LABEL_STUDIO_PORT%

REM Keep the window open
echo Label Studio has been launched in the background. Press any key to close this window...
pause >nul