# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

### Frontend (React + Vite)
- `npm run dev` - Start the development server on http://localhost:5173
- `npm run build` - Build for production
- `npm run lint` - Run ESLint
- `npm run preview` - Preview production build locally

### Backend (Python/FastAPI)
- `cd src/backend && pip install -r requirements.txt` - Install Python dependencies
- `cd src/backend && python server.py` - Start the FastAPI WebSocket server on port 8000
- `cd src/backend && python train_model.py` - Train the ML models

## Architecture Overview

This is a portfolio website with a React frontend and Python backend demonstrating real-time analytics capabilities.

### Frontend Architecture
- **React + Vite** application using Tailwind CSS for styling
- **Main Components**:
  - `Hero.jsx` - Landing section
  - `Skills.jsx` - Technical skills showcase
  - `Projects.jsx` - Project portfolio
  - `projects/ecommerce/` - E-commerce analytics dashboard demo with real-time WebSocket connection

### Backend Architecture
- **FastAPI** server (`server.py`) with WebSocket support for real-time data streaming
- **Machine Learning Pipeline**:
  - `analytics_pipeline.py` - Transaction processing with Apache Beam
  - `train_model.py` - Model training using TensorFlow/Keras and scikit-learn
  - Pre-trained models stored in `models/` directory
- **WebSocket Connection**: Frontend connects to `ws://localhost:8000/ws` for real-time analytics updates

### Key Integration Points
- WebSocket connection managed by `useAnalytics.js` hook
- CORS configured for http://localhost:5173 (Vite dev server)
- Transaction processing includes ML predictions for customer behavior