@echo off
echo Starting District RAG Chatbot Backend...
cd /d %~dp0
if exist venv (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo Warning: Virtual environment not found.
)
cd /d %~dp0\backend
echo Running server on http://localhost:8000
python -m uvicorn app.main:app --reload
pause
