# Start the Frontend
Write-Host "Starting District RAG Chatbot Frontend..." -ForegroundColor Cyan

if (Test-Path "node_modules") {
    Write-Host "Running frontend dev server..." -ForegroundColor Green
    npm run dev
} else {
    Write-Host "node_modules not found. Installing dependencies first..." -ForegroundColor Yellow
    npm install
    npm run dev
}
