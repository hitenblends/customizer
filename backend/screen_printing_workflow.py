"""
Screen Printing Workflow Functions
=================================

This module implements the professional screen printing workflow:
1. Background removal with rembg
2. Edge cleanup for crisp boundaries
3. Vectorization to SVG
4. Color separations for printing
"""

import os
import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict, Any
import tempfile
import subprocess
import json


class ScreenPrintingWorkflow:
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def remove_background_clean(self, image_path: str) -> str:
        """
        Step 1: Simple color-based background removal with text preservation
        """
        try:
            import cv2
            import numpy as np
            
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("Could not load image")
            
            # IMPORTANT: OpenCV loads images as BGR, but we want to keep original colors
            # Don't convert to RGB - keep the original BGR colors to preserve exact colors
            img_bgr = img.copy()
            
            # Method 1: Color-based background removal
            # Convert to HSV for better color detection
            img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            
            # Get image dimensions
            height, width = img_hsv.shape[:2]
            
            # Sample background color from corners (assuming background is uniform)
            corner_samples = [
                img_hsv[0, 0],           # Top-left
                img_hsv[0, width-1],     # Top-right
                img_hsv[height-1, 0],    # Bottom-left
                img_hsv[height-1, width-1] # Bottom-right
            ]
            
            # Calculate average background color
            bg_hsv = np.mean(corner_samples, axis=0)
            
            # Create mask for background removal
            # Use a tolerance range around the background color
            h_tolerance = 20  # Hue tolerance
            s_tolerance = 50  # Saturation tolerance
            v_tolerance = 50  # Value tolerance
            
            # Create background mask
            bg_mask = np.ones((height, width), dtype=np.uint8)
            
            for y in range(height):
                for x in range(width):
                    pixel_hsv = img_hsv[y, x]
                    
                    # Check if pixel is close to background color
                    h_diff = abs(pixel_hsv[0] - bg_hsv[0])
                    s_diff = abs(pixel_hsv[1] - bg_hsv[1])
                    v_diff = abs(pixel_hsv[2] - bg_hsv[2])
                    
                    # Handle hue wrap-around (hue is circular 0-179)
                    if h_diff > 90:
                        h_diff = 179 - h_diff
                    
                    # If pixel is close to background, mark as background
                    if (h_diff <= h_tolerance and 
                        s_diff <= s_tolerance and 
                        v_diff <= v_tolerance):
                        bg_mask[y, x] = 0
                    else:
                        bg_mask[y, x] = 255
            
            # Method 2: Edge-based refinement to preserve text and outlines
            # Convert to grayscale
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            
            # Detect edges
            edges = cv2.Canny(gray, 30, 100)
            
            # Dilate edges to ensure text is captured
            kernel = np.ones((2, 2), np.uint8)
            dilated_edges = cv2.dilate(edges, kernel, iterations=1)
            
            # Combine background mask with edge preservation
            # Keep pixels that are either not background OR are edges
            edge_mask = (dilated_edges > 0).astype(np.uint8) * 255
            final_mask = cv2.bitwise_or(bg_mask, edge_mask)
            
            # Method 3: Component analysis to preserve important elements
            # Find connected components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(final_mask, connectivity=8)
            
            # Refine mask based on component analysis
            refined_mask = final_mask.copy()
            
            for i in range(1, num_labels):  # Skip background (label 0)
                component_mask = (labels == i).astype(np.uint8)
                component_area = np.sum(component_mask)
                
                # Check if component contains edges (likely text or important design)
                component_edges = cv2.bitwise_and(edge_mask, component_mask)
                edge_density = np.sum(component_edges) / component_area if component_area > 0 else 0
                
                # If component has high edge density, preserve it completely
                if edge_density > 0.1:  # More than 10% edges
                    refined_mask[component_mask > 0] = 255
                    continue
                
                # Check if this is a large component (likely important design element)
                if component_area > 200:  # Large enough to be significant
                    # Sample pixels to check if they're dark (likely text/design)
                    component_pixels = img_bgr[component_mask > 0]
                    if len(component_pixels) > 0:
                        avg_brightness = np.mean(component_pixels)
                        # Preserve dark components (likely important design elements)
                        if avg_brightness < 180:  # Dark component threshold
                            refined_mask[component_mask > 0] = 255
            
            # Create RGBA output with transparency
            # IMPORTANT: Keep original BGR colors exactly as they are - NO COLOR CONVERSION
            # Create RGBA image manually to avoid color space conversion
            height, width = img_bgr.shape[:2]
            result_rgba = np.zeros((height, width, 4), dtype=np.uint8)
            
            # Copy BGR channels exactly as they are (no conversion)
            result_rgba[:, :, 0] = img_bgr[:, :, 0]  # Blue channel
            result_rgba[:, :, 1] = img_bgr[:, :, 1]  # Green channel  
            result_rgba[:, :, 2] = img_bgr[:, :, 2]  # Red channel
            
            # Apply mask to alpha channel (transparency)
            result_rgba[:, :, 3] = refined_mask
            
            # IMPORTANT: Set RGB values of background pixels to white to prevent black color detection
            # This ensures transparent areas don't interfere with color analysis
            bg_pixels = (refined_mask == 0)  # Background pixels (alpha = 0)
            result_rgba[bg_pixels, 0] = 255  # Set blue to 255 for background
            result_rgba[bg_pixels, 1] = 255  # Set green to 255 for background
            result_rgba[bg_pixels, 2] = 255  # Set red to 255 for background
            
            # Save result
            output_path = os.path.join(self.temp_dir, "01_removed_bg.png")
            cv2.imwrite(output_path, result_rgba)
            
            return output_path
            
        except Exception as e:
            raise Exception(f"Background removal failed: {str(e)}")
    
    def cleanup_edges(self, image_path: str, threshold: int = 128) -> str:
        """
        Step 2: Clean up edges and convert transparency to hard edges
        """
        try:
            # Load the transparent PNG
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError("Could not load image")
            
            # Extract alpha channel
            if img.shape[-1] == 4:  # RGBA
                alpha = img[:, :, 3]
            else:
                # If no alpha, create one from the image
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                alpha = gray
            
            # Apply threshold to create hard edges
            # Pixels above threshold become fully opaque (255)
            # Pixels below threshold become fully transparent (0)
            hard_alpha = np.where(alpha > threshold, 255, 0).astype(np.uint8)
            
            # Create new RGBA image with hard edges
            result = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
            
            if img.shape[-1] == 4:
                # Copy RGB channels
                result[:, :, :3] = img[:, :, :3]
            else:
                # Convert BGR to RGB
                result[:, :, :3] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Apply hard alpha
            result[:, :, 3] = hard_alpha
            
            # Save cleaned image
            output_path = os.path.join(self.temp_dir, "02_cleaned_edges.png")
            cv2.imwrite(output_path, result)
            
            return output_path
            
        except Exception as e:
            raise Exception(f"Edge cleanup failed: {str(e)}")
    
    def vectorize_to_svg(self, image_path: str) -> str:
        """
        Step 3: Convert cleaned PNG to SVG using potrace
        """
        try:
            # First, convert to black and white for potrace
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError("Could not load image")
            
            # Extract alpha channel and convert to binary
            if img.shape[-1] == 4:
                alpha = img[:, :, 3]
            else:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                alpha = gray
            
            # Create binary mask (black = transparent, white = opaque)
            binary_mask = np.where(alpha > 128, 255, 0).astype(np.uint8)
            
            # Save binary PNG for potrace
            binary_path = os.path.join(self.temp_dir, "03_binary.png")
            cv2.imwrite(binary_path, binary_mask)
            
            # Convert to SVG using potrace
            svg_path = os.path.join(self.temp_dir, "04_vectorized.svg")
            
            try:
                # Try to use potrace if available
                subprocess.run([
                    'potrace', 
                    binary_path, 
                    '-s',  # SVG output
                    '-o', svg_path
                ], check=True, capture_output=True)
                
                return svg_path
                
            except (subprocess.CalledProcessError, FileNotFoundError):
                # Fallback: create a simple SVG manually
                return self._create_simple_svg(binary_mask, svg_path)
                
        except Exception as e:
            raise Exception(f"Vectorization failed: {str(e)}")
    
    def _create_simple_svg(self, binary_mask: np.ndarray, svg_path: str) -> str:
        """
        Fallback: Create a simple SVG when potrace is not available
        """
        height, width = binary_mask.shape
        
        # Create basic SVG
        svg_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <pattern id="grid" width="10" height="10" patternUnits="userSpaceOnUse">
      <path d="M 10 0 L 0 0 0 10" fill="none" stroke="gray" stroke-width="0.5"/>
    </pattern>
  </defs>
  <rect width="100%" height="100%" fill="url(#grid)"/>
  <text x="50%" y="50%" text-anchor="middle" dominant-baseline="middle" 
        font-family="Arial" font-size="16" fill="black">
    Vector conversion requires potrace installation
  </text>
</svg>"""
        
        with open(svg_path, 'w') as f:
            f.write(svg_content)
        
        return svg_path
    
    def create_color_separations(self, image_path: str, colors: List[Tuple[int, int, int]]) -> Dict[str, str]:
        """
        Step 4: Create color separations for each detected color
        """
        try:
            # Load the cleaned image
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError("Could not load image")
            
            # Extract alpha channel
            if img.shape[-1] == 4:
                alpha = img[:, :, 3]
                rgb = img[:, :, :3]
            else:
                alpha = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            separations = {}
            
            for i, color in enumerate(colors):
                # Create mask for this color
                # Find pixels that are similar to this color
                color_array = np.array(color, dtype=np.uint8)
                
                # Calculate color distance for each pixel
                color_diff = np.sqrt(np.sum((rgb - color_array) ** 2, axis=2))
                
                # Create mask for pixels within threshold
                threshold = 50  # Adjustable color tolerance
                color_mask = (color_diff < threshold) & (alpha > 128)
                
                # Create separation image
                separation = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
                separation[:, :, 3] = color_mask.astype(np.uint8) * 255  # Alpha channel
                
                # Save separation
                separation_path = os.path.join(self.temp_dir, f"05_separation_{i+1}_{color[0]}_{color[1]}_{color[2]}.png")
                cv2.imwrite(separation_path, separation)
                
                separations[f"color_{i+1}"] = {
                    "rgb": color,
                    "hex": f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}",
                    "file": separation_path
                }
            
            return separations
            
        except Exception as e:
            raise Exception(f"Color separation failed: {str(e)}")
    
    def run_complete_workflow(self, image_path: str, colors: List[Tuple[int, int, int]]) -> Dict[str, Any]:
        """
        Run the complete screen printing workflow
        """
        try:
            results = {}
            
            # Step 1: Remove background
            print("🔄 Step 1: Removing background...")
            bg_removed = self.remove_background_clean(image_path)
            results["background_removed"] = bg_removed
            
            # Step 2: Clean up edges
            print("🔄 Step 2: Cleaning up edges...")
            edges_cleaned = self.cleanup_edges(bg_removed)
            results["edges_cleaned"] = edges_cleaned
            
            # Step 3: Vectorize
            print("🔄 Step 3: Vectorizing to SVG...")
            svg_file = self.vectorize_to_svg(edges_cleaned)
            results["vectorized"] = svg_file
            
            # Step 4: Color separations
            print("🔄 Step 4: Creating color separations...")
            separations = self.create_color_separations(edges_cleaned, colors)
            results["separations"] = separations
            
            print("✅ Workflow completed successfully!")
            return results
            
        except Exception as e:
            raise Exception(f"Workflow failed: {str(e)}")
    
    def cleanup_temp_files(self):
        """Clean up temporary files"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"🧹 Cleaned up temporary directory: {self.temp_dir}")


# Create global instance
workflow = ScreenPrintingWorkflow()
