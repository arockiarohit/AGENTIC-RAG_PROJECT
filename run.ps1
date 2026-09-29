# Activate the virtual environment and start the FastAPI server
Write-Host "Starting District RAG Chatbot Backend..." -ForegroundColor Cyan

# Check if venv exists
if (Test-Path ".\venv") {
    Write-Host "Activating virtual environment..." -ForegroundColor Gray
    # Import the activation script
    . .\venv\Scripts\Activate.ps1
} else {
    Write-Host "Warning: Virtual environment (venv) not found. Running with system python." -ForegroundColor Yellow
}

Write-Host "Running server on http://localhost:8000" -ForegroundColor Green
python -m uvicorn app.main:app --reload
