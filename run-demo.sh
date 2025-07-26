#!/bin/bash

echo "Starting E-commerce Analytics Demo..."
echo "======================================="

# Start backend server
echo "Starting backend server on port 8000..."
cd src/backend
python3 server.py &
BACKEND_PID=$!
echo "Backend server started with PID: $BACKEND_PID"

# Wait for backend to start
sleep 3

# Start frontend
echo ""
echo "Starting frontend on port 5173..."
cd ../..
npm run dev &
FRONTEND_PID=$!
echo "Frontend started with PID: $FRONTEND_PID"

echo ""
echo "======================================="
echo "Demo is running!"
echo "Frontend: http://localhost:5173"
echo "Backend WebSocket: ws://localhost:8000/ws"
echo ""
echo "Navigate to: http://localhost:5173/projects/ecommerce-analytics"
echo ""
echo "Press Ctrl+C to stop both servers"

# Wait for user to press Ctrl+C
trap "echo 'Stopping servers...'; kill $BACKEND_PID $FRONTEND_PID; exit" INT
wait