@echo off
SET SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%image-recognition"
call "%SCRIPT_DIR%image-recognition\.venv\Scripts\activate.bat"
python "%SCRIPT_DIR%image-recognition\train_gui.py"
