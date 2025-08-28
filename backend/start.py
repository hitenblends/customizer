#!/usr/bin/env python3
"""
Startup script for DTF Color Extractor Backend
"""

import uvicorn
import os
from main import app

if __name__ == "__main__":
    # Get port from environment or use default
    port = int(os.getenv("PORT", 8000))
    
    # Get host from environment or use default
    host = os.getenv("HOST", "0.0.0.0")
    
    print(f"🚀 Starting DTF Color Extractor Backend...")
    print(f"📍 Server will run on: http://{host}:{port}")
    print(f"📚 API Documentation: http://{host}:{port}/docs")
    print(f"🔍 Health Check: http://{host}:{port}/health")
    print(f"⏹️  Press Ctrl+C to stop the server")
    print("-" * 50)
    
    # Start the server
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,  # Auto-reload on code changes
        log_level="info"
    )
