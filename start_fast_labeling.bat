@echo off
SET SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%fast-labeling"
call "%SCRIPT_DIR%fast-labeling\venv\Scripts\activate.bat"
python "%SCRIPT_DIR%fast-labeling\fast-labeling.py"
