@echo off
echo ===================================================
echo   Starting Face Anti-Spoofing System
echo ===================================================

echo 1. Starting Backend API (FastAPI)...
start "Face Anti-Spoofing Backend" cmd /k "python -m uvicorn api.main:app --host 127.0.0.1 --port 8000"

echo Waiting 5 seconds for backend to start...
timeout /t 5 /nobreak >nul

echo 2. Starting Frontend UI (Streamlit)...
start "Face Anti-Spoofing Frontend" cmd /k "python -m streamlit run app/streamlit_app.py"

echo ===================================================
echo   System Running!
echo   Frontend: http://localhost:8501
echo   Backend:  http://127.0.0.1:8000/docs
echo ===================================================
