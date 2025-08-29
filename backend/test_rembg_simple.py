#!/usr/bin/env python3
"""
Simple test script to test rembg directly
"""

import rembg
from PIL import Image
import numpy as np

def test_rembg():
    print("🔍 Testing rembg directly...")
    
    # Create a simple test image (white background with colored text)
    width, height = 300, 200
    
    # Create white background
    test_image = Image.new('RGB', (width, height), 'white')
    
    # Add some colored elements (simulating BP logo)
    from PIL import ImageDraw
    draw = ImageDraw.Draw(test_image)
    
    # Add green text (simulating BP text)
    draw.text((50, 80), "BP", fill=(0, 165, 80))  # Green color
    
    # Add yellow circle (simulating logo element)
    draw.ellipse([100, 50, 200, 150], fill=(255, 220, 0))  # Yellow
    
    # Save test image
    test_path = "test_bp_logo.png"
    test_image.save(test_path)
    print(f"✅ Created test image: {test_path}")
    
    try:
        # Test rembg with default model
        print("🔍 Testing rembg with default model...")
        result_default = rembg.remove(test_image)
        print(f"✅ Default model result: {result_default.size}, mode: {result_default.mode}")
        
        # Test rembg with u2net model
        print("🔍 Testing rembg with u2net model...")
        session = rembg.new_session('u2net')
        result_u2net = rembg.remove(test_image, session=session)
        print(f"✅ u2net model result: {result_u2net.size}, mode: {result_u2net.mode}")
        
        # Save results
        result_default.save("result_default.png")
        result_u2net.save("result_u2net.png")
        print("✅ Saved both results for comparison")
        
        # Check transparency
        default_array = np.array(result_default)
        u2net_array = np.array(result_u2net)
        
        if len(default_array.shape) == 3 and default_array.shape[2] == 4:
            default_transparency = np.sum(default_array[:, :, 3] < 128) / (default_array.shape[0] * default_array.shape[1]) * 100
            print(f"📊 Default model transparency: {default_transparency:.1f}%")
        
        if len(u2net_array.shape) == 3 and u2net_array.shape[2] == 4:
            u2net_transparency = np.sum(u2net_array[:, :, 3] < 128) / (u2net_array.shape[0] * u2net_array.shape[1]) * 100
            print(f"📊 u2net model transparency: {u2net_transparency:.1f}%")
        
        print("\n🎯 rembg test completed successfully!")
        
    except Exception as e:
        print(f"❌ rembg test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_rembg()
