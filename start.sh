#!/bin/bash
PORT=${PORT:-8080}
echo "Starting server on port $PORT"
uvicorn backend.main:app --host 0.0.0.0 --port $PORT 