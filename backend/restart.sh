#!/bin/bash
# Restart script for DTF Color Extractor Backend

echo "🔄 Restarting DTF Color Extractor Backend..."

# Kill any existing server on port 8000
echo "🛑 Stopping existing server..."
lsof -ti:8000 | xargs kill -9 2>/dev/null || echo "No existing server found"

# Wait a moment for cleanup
sleep 2

# Start the server
echo "🚀 Starting new server..."
python3 start.py
