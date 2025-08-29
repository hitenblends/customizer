#!/usr/bin/env python3
"""
Production startup script for Render hosting
"""

import uvicorn
from main import app

if __name__ == "__main__":
    # Production settings for Render
    uvicorn.run(
        app,
        host="0.0.0.0",  # Bind to all interfaces
        port=int(8000),   # Render will set PORT environment variable
        log_level="info"
    )
