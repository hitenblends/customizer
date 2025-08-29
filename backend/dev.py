#!/usr/bin/env python3
"""
Development server with auto-reload enabled
Use this when actively developing and want automatic restarts
"""

import uvicorn
import os

if __name__ == "__main__":
    print("🚀 Starting DTF Color Extractor Backend (DEVELOPMENT MODE)")
    print("=" * 60)
    print("⚠️  Auto-reload ENABLED - server will restart on file changes")
    print("💡 Use 'python3 start.py' for stable operation")
    print("=" * 60)
    
    # Start the server with auto-reload
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=True,  # Auto-reload on code changes
        log_level="info"
    )
