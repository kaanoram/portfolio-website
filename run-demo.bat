@echo off
echo Starting E-commerce Analytics Demo...
echo =======================================

echo Starting backend server on port 8000...
start /B cmd /c "cd src\backend && python server.py"
timeout /t 3 /nobreak >nul

echo.
echo Starting frontend on port 5173...
start /B cmd /c "npm run dev"

echo.
echo =======================================
echo Demo is running!
echo Frontend: http://localhost:5173
echo Backend WebSocket: ws://localhost:8000/ws
echo.
echo Navigate to: http://localhost:5173/projects/ecommerce-analytics
echo.
echo Press Ctrl+C in both windows to stop the servers
pause