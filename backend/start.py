#!/usr/bin/env python3
"""
Startup script for DTF Color Extractor Backend
"""

import uvicorn
import os
import sys
from main import app

def print_banner():
    print("🚀 DTF Color Extractor Backend")
    print("=" * 50)
    print("📚 API Documentation: /docs")
    print("🔍 Health Check: /health")
    print("🎨 Color Extraction: /extract-colors")
    print("🖼️  Background Removal: /remove-background")
    print("🔍 Background Detection: /detect-background")
    print("🎨 Palette Info: /palette")
    print("=" * 50)

def get_server_config():
    """Get server configuration from environment variables"""
    config = {
        'host': os.getenv("HOST", "0.0.0.0"),
        'port': int(os.getenv("PORT", 8000)),
        'reload': os.getenv("RELOAD", "false").lower() == "true",
        'log_level': os.getenv("LOG_LEVEL", "info")
    }
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] in ['--reload', '-r']:
            config['reload'] = True
        elif sys.argv[1] in ['--no-reload', '-n']:
            config['reload'] = False
        elif sys.argv[1] in ['--help', '-h']:
            print_help()
            sys.exit(0)
    
    return config

def print_help():
    print("Usage: python3 start.py [OPTIONS]")
    print("")
    print("Options:")
    print("  --reload, -r     Enable auto-reload (default: false)")
    print("  --no-reload, -n  Disable auto-reload")
    print("  --help, -h       Show this help message")
    print("")
    print("Environment Variables:")
    print("  RELOAD=true      Enable auto-reload")
    print("  HOST=0.0.0.0    Server host (default: 0.0.0.0)")
    print("  PORT=8000       Server port (default: 8000)")
    print("  LOG_LEVEL=info  Log level (default: info)")
    print("")
    print("Examples:")
    print("  python3 start.py              # No auto-reload (stable)")
    print("  python3 start.py --reload     # With auto-reload (development)")
    print("  RELOAD=true python3 start.py  # Environment variable method")

if __name__ == "__main__":
    config = get_server_config()
    
    print_banner()
    print(f"📍 Server: http://{config['host']}:{config['port']}")
    print(f"🔄 Auto-reload: {'✅ Enabled' if config['reload'] else '❌ Disabled'}")
    print(f"📝 Log Level: {config['log_level']}")
    print(f"⏹️  Press Ctrl+C to stop")
    print("-" * 50)
    
    if config['reload']:
        print("⚠️  Auto-reload enabled - server will restart on file changes")
        print("💡 Use 'python3 start.py --no-reload' for stable operation")
    else:
        print("✅ Stable mode - no auto-restarts")
        print("💡 Use 'python3 start.py --reload' for development mode")
    
    print("-" * 50)
    
    # Start the server
    uvicorn.run(
        "main:app",
        host=config['host'],
        port=config['port'],
        reload=config['reload'],
        log_level=config['log_level']
    )
