#!/usr/bin/env python3
"""
Test script to debug the import issue in the server context.
"""

import sys
import os
from pathlib import Path

print("🔍 Testing server import context...")
print(f"Current working directory: {os.getcwd()}")
print(f"Python path: {sys.path[:3]}...")  # Show first 3 entries

# Try to import the same way the server does
try:
    print("\n1. Testing direct import...")
    from background_remover import BackgroundRemover
    print("   ✅ Direct import successful")
except ImportError as e:
    print(f"   ❌ Direct import failed: {e}")
    
    # Try with absolute path
    try:
        print("\n2. Testing with absolute path...")
        current_dir = Path(__file__).parent.absolute()
        sys.path.insert(0, str(current_dir))
        from background_remover import BackgroundRemover
        print("   ✅ Absolute path import successful")
    except ImportError as e2:
        print(f"   ❌ Absolute path import failed: {e2}")
        
        # Try with relative path
        try:
            print("\n3. Testing with relative path...")
            sys.path.insert(0, ".")
            from background_remover import BackgroundRemover
            print("   ✅ Relative path import successful")
        except ImportError as e3:
            print(f"   ❌ Relative path import failed: {e3}")
            
            # List files in current directory
            print(f"\n4. Files in current directory:")
            for file in os.listdir("."):
                if file.endswith(".py"):
                    print(f"   - {file}")
            
            sys.exit(1)

# If we get here, import was successful
print("\n✅ BackgroundRemover import successful in server context!")
print("The issue must be elsewhere in the code.")
