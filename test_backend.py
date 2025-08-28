#!/usr/bin/env python3
"""
Test script for DTF Color Extractor Backend
"""

import requests
import json
import os
from PIL import Image
import numpy as np

def create_test_image():
    """Create a simple test image with known colors"""
    # Create a 100x100 image with 4 distinct colors
    img = Image.new('RGB', (100, 100), color='white')
    pixels = img.load()
    
    # Add some colored regions
    for x in range(100):
        for y in range(100):
            if x < 50 and y < 50:
                pixels[x, y] = (255, 0, 0)  # Red
            elif x >= 50 and y < 50:
                pixels[x, y] = (0, 255, 0)  # Green
            elif x < 50 and y >= 50:
                pixels[x, y] = (0, 0, 255)  # Blue
            else:
                pixels[x, y] = (255, 255, 0)  # Yellow
    
    # Save test image
    test_image_path = "test_image.png"
    img.save(test_image_path)
    return test_image_path

def test_backend():
    """Test the backend API"""
    base_url = "http://localhost:8000"
    
    print("🧪 Testing DTF Color Extractor Backend...")
    print("=" * 50)
    
    # Test 1: Health check
    print("1️⃣ Testing health check...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to backend. Is it running?")
        print("   Start with: python start.py")
        return False
    
    # Test 2: Root endpoint
    print("\n2️⃣ Testing root endpoint...")
    try:
        response = requests.get(base_url)
        if response.status_code == 200:
            print("✅ Root endpoint passed")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Root endpoint error: {e}")
    
    # Test 3: Color extraction
    print("\n3️⃣ Testing color extraction...")
    try:
        # Create test image
        test_image_path = create_test_image()
        
        # Prepare file upload
        with open(test_image_path, 'rb') as f:
            files = {'file': ('test_image.png', f, 'image/png')}
            data = {
                'num_colors': 4,
                'min_percentage': 0.1,
                'merge_threshold': 30.0
            }
            
            response = requests.post(f"{base_url}/extract-colors", files=files, data=data)
        
        # Clean up test image
        os.remove(test_image_path)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Color extraction passed")
            print(f"   Colors found: {result['total_colors']}")
            print(f"   Algorithm: {result['algorithm']}")
            
            # Show extracted colors
            print("   Extracted colors:")
            for i, color in enumerate(result['colors']):
                print(f"     {i+1}. RGB: {color['rgb']}, HEX: {color['hex']}, %: {color['percentage']}")
                
        else:
            print(f"❌ Color extraction failed: {response.status_code}")
            print(f"   Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Color extraction error: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Backend testing completed!")
    return True

if __name__ == "__main__":
    test_backend()
