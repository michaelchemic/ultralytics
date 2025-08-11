@echo off
:: 自动以管理员身份运行本脚本
:: 检查是否有管理员权限
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo Requesting administrator privileges...
    powershell -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
    exit /b
)

@echo off
chcp 65001 > nul
SETLOCAL ENABLEDELAYEDEXPANSION

:: Ensure script runs from its own directory
cd /d "%~dp0"

:: Configuration
set PYTHON_INSTALLER=python-3.13.5-amd64.exe
set OFFLINE_DIR=offline_packages
set REQUIREMENTS_FILE=requirements.txt
set MAIN_SCRIPT=main.py

:: Check for administrator privileges
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo Error: Please run this script as Administrator!
    pause
    exit /b
)

:: Check if required files exist
if not exist "%PYTHON_INSTALLER%" (
    echo Error: %PYTHON_INSTALLER% not found!
    pause
    exit /b
)
if not exist "%REQUIREMENTS_FILE%" (
    echo Error: %REQUIREMENTS_FILE% not found!
    pause
    exit /b
)

:: Install Python
echo Installing Python...
start /wait "" "%PYTHON_INSTALLER%"


:: 设置Python路径（不检查是否成功）
set PYTHON_PATH=python




:: Check if install_offline.py exists
if not exist "install_offline.py" (
    echo Error: install_offline.py not found!
    pause
    exit /b
)

:: Install dependencies
echo Installing dependencies from offline packages...
"%PYTHON_PATH%" "%~dp0install_offline.py" "%OFFLINE_DIR%" "%REQUIREMENTS_FILE%" || (
    echo Error: Failed to run install_offline.py
    pause
    exit /b
)

:: Launch main program
if not exist "%MAIN_SCRIPT%" (
    echo Error: %MAIN_SCRIPT% not found!
    pause
    exit /b
)

echo Launching main program...
%PYTHON_PATH% "%MAIN_SCRIPT%"
if %errorlevel% neq 0 (
    echo Warning: The main program exited with code %errorlevel%.
)
pause
exit /b

:: Create shortcut of start.exe on Desktop
:: 检查目标EXE是否存在
if not exist "%~dp0数智安AI病害检测标注平台.exe" (
    echo 错误: 无法找到 "%~dp0数智安AI病害检测标注平台.exe"
    pause
    exit /b
)

:: 创建快捷方式（单行命令，避免换行转义问题）
powershell -ExecutionPolicy Bypass -command "$desktop = [Environment]::GetFolderPath('Desktop'); $exePath = '%~dp0数智安AI病害检测标注平台.exe'; $WshShell = New-Object -ComObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut(\"$desktop\\数智安AI病害检测标注平台.lnk\"); $Shortcut.TargetPath = $exePath; $Shortcut.WorkingDirectory = '%~dp0'; $Shortcut.Save(); Write-Output '快捷方式已创建到桌面'"

pause
