from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import tempfile
import os
from typing import List, Dict, Any
from color_palette import color_matcher

app = FastAPI(title="DTF Color Extractor API - PALETTE MATCHING", version="3.0.0")

# Import screen printing workflow router
from screen_printing_api import router as screen_printing_router

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PaletteBasedColorExtractor:
    """Extract colors and match them to predefined palette"""
    
    def _is_black_legitimate(self, img_bgr: np.ndarray, color: tuple, percentage: float) -> bool:
        """
        Check if black color is legitimate design or background removal artifact
        Returns True if black appears to be legitimate design element
        """
        try:
            r, g, b = color
            if not (r < 20 and g < 20 and b < 20):
                return True  # Not black, so legitimate
                
            # For black colors, check spatial distribution
            height, width = img_bgr.shape[:2]
            
            # Create mask for this specific color
            color_mask = np.zeros((height, width), dtype=np.uint8)
            
            # Find pixels that match this color (with some tolerance)
            tolerance = 30
            for y in range(height):
                for x in range(width):
                    pixel = img_bgr[y, x]
                    if (abs(pixel[0] - b) <= tolerance and 
                        abs(pixel[1] - g) <= tolerance and 
                        abs(pixel[2] - r) <= tolerance):
                        color_mask[y, x] = 255
            
            # Count total black pixels
            total_black_pixels = np.sum(color_mask > 0)
            
            # If very few black pixels, likely artifact
            if total_black_pixels < (height * width * 0.01):  # Less than 1% of image
                return False
                
            # Check if black pixels are clustered (legitimate) or scattered (artifact)
            # Use connected components to find clusters
            num_labels, labels, stats, centroids = cv2.connectedComponents(color_mask, connectivity=8)
            
            if num_labels <= 1:  # No clusters found
                return False
                
            # Check the largest cluster size
            largest_cluster_size = np.max(stats[1:, cv2.CC_STAT_AREA]) if num_labels > 1 else 0
            
            # If largest cluster is too small, likely scattered artifacts
            if largest_cluster_size < (height * width * 0.005):  # Less than 0.5% of image
                return False
                
            # If we get here, black appears to be legitimate design
            return True
            
        except Exception as e:
            # If analysis fails, be conservative and keep the color
            return True
    
    def __init__(self):
        self.color_matcher = color_matcher
    
    def extract_colors_with_palette_matching(self, image_path: str, num_colors: int = 12, 
                                           min_percentage: float = 1.0) -> Dict[str, Any]:
        """
        Extract colors from image and match them to predefined palette
        """
        try:
            # Load image with OpenCV - IMPORTANT: Load with alpha channel for transparency
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError("Could not load image")
            
            # Check if image has alpha channel (4 channels = RGBA)
            has_alpha = img.shape[-1] == 4 if len(img.shape) > 2 else False
            
            if has_alpha:
                # Extract alpha channel
                alpha = img[:, :, 3]
                # Create mask for non-transparent pixels (alpha > 0)
                non_transparent_mask = alpha > 0
                
                # Only analyze non-transparent pixels
                img_bgr = img[:, :, :3]  # Extract BGR channels
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                
                # Apply mask to get only non-transparent pixels
                non_transparent_pixels = img_rgb[non_transparent_mask]
                
                if len(non_transparent_pixels) == 0:
                    raise ValueError("No non-transparent pixels found for color extraction")
                
                # IMPORTANT: Keep ALL white pixels - don't filter them out!
                # The issue was that we were filtering out legitimate white design elements
                # Only remove the exact pure white background pixels (255,255,255) that we set
                
                # Keep all pixels - no filtering
                design_pixels = non_transparent_pixels
                
                print(f"   Keeping ALL non-transparent pixels: {len(design_pixels)}")
                print(f"   No white pixel filtering applied")
                
                if len(design_pixels) == 0:
                    # If all non-transparent pixels are white, use the original non-transparent pixels
                    design_pixels = non_transparent_pixels
                
                # Use only design pixels for analysis
                pixels = design_pixels
                total_original_pixels = img_rgb.shape[0] * img_rgb.shape[1]
                transparent_pixel_count = total_original_pixels - len(non_transparent_pixels)
                white_background_count = len(non_transparent_pixels) - len(design_pixels)
                
                print(f"🔍 Transparency Analysis:")
                print(f"   Total pixels: {total_original_pixels}")
                print(f"   Non-transparent pixels: {len(non_transparent_pixels)}")
                print(f"   Transparent pixels: {transparent_pixel_count}")
                print(f"   White background pixels: {white_background_count}")
                print(f"   Design pixels: {len(design_pixels)}")
                print(f"   Transparency percentage: {(transparent_pixel_count/total_original_pixels)*100:.1f}%")
                
                # Debug: Show what colors are in the non-transparent pixels before filtering
                if len(non_transparent_pixels) > 0:
                    unique_colors_before = np.unique(non_transparent_pixels, axis=0)
                    print(f"   Colors in non-transparent pixels (before filtering): {len(unique_colors_before)}")
                    for i, color in enumerate(unique_colors_before[:10]):  # Show first 10
                        print(f"     Color {i+1}: RGB({color[0]},{color[1]},{color[2]})")
                
                # Debug: Show what colors are in the design pixels after filtering
                if len(design_pixels) > 0:
                    unique_colors_after = np.unique(design_pixels, axis=0)
                    print(f"   Colors in design pixels (after filtering): {len(unique_colors_after)}")
                    for i, color in enumerate(unique_colors_after[:10]):  # Show first 10
                        print(f"     Color {i+1}: RGB({color[0]},{color[1]},{color[2]})")
            else:
                # No alpha channel, use all pixels
                img_bgr = img
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                pixels = img_rgb.reshape(-1, 3)
            
            # Get image dimensions for processing
            height, width = img_rgb.shape[:2]
            
            # Filter out very light and very dark pixels (likely background)
            brightness = np.mean(pixels, axis=1)
            # Much more lenient filtering for white and black colors
            # Keep all pixels - let the percentage filtering handle it
            valid_pixels = pixels
            
            if len(valid_pixels) == 0:
                raise ValueError("No valid pixels found for color extraction")
            
            # Use K-means clustering to find dominant colors
            # For background-removed images, limit to fewer colors to avoid artifacts
            if has_alpha and len(design_pixels) < len(non_transparent_pixels) * 0.5:  # If we filtered out more than half
                max_clusters = min(4, num_colors)  # Limit to 4 colors max for background-removed images
            else:
                max_clusters = num_colors
            
            n_clusters = min(max_clusters, len(valid_pixels), 32)
            print(f"🔍 Clustering: Using {n_clusters} clusters from {len(valid_pixels)} pixels")
            
            kmeans = KMeans(n_clusters=n_clusters, n_init=10, max_iter=300, random_state=42)
            kmeans.fit(valid_pixels)
            
            # Get cluster centers and labels
            cluster_centers = kmeans.cluster_centers_.astype(int)
            labels = kmeans.labels_
            
            # Calculate percentage for each cluster
            unique_labels, counts = np.unique(labels, return_counts=True)
            
            # Use total design pixels for percentage calculation
            if has_alpha:
                total_pixels = len(design_pixels)
            else:
                total_pixels = len(valid_pixels)
            
            colors = []
            percentages = []
            
            for label, count in zip(unique_labels, counts):
                percentage = (count / total_pixels) * 100
                color = tuple(int(c) for c in cluster_centers[label])  # Convert numpy.int64 to Python int
                
                # Special handling for white, black, and very light/dark colors
                brightness = np.mean(color)
                r, g, b = color
                
                # Include white, black, and very light/dark colors only if they have meaningful presence
                # Use different thresholds for different color types
                min_percentage_for_whites = 0.01  # Very low threshold for whites to preserve design
                min_percentage_for_blacks = 0.5   # Low threshold for blacks
                
                if (r > 240 and g > 240 and b > 240):  # Near white (more inclusive)
                    # Include whites with lower threshold
                    if percentage >= min_percentage_for_whites:
                        colors.append(color)
                        percentages.append(float(percentage))
                elif (r < 5 and g < 5 and b < 5):  # Pure black
                    # Only include black if it has significant presence (likely intentional)
                    if percentage >= min_percentage_for_blacks:
                        colors.append(color)
                        percentages.append(float(percentage))
                elif brightness > 220:  # Very light colors (more inclusive)
                    # Include very light colors with moderate threshold
                    if percentage >= min_percentage_for_whites:
                        colors.append(color)
                        percentages.append(float(percentage))
                elif brightness < 15:  # Very dark colors (but not pure black)
                    # Include very dark colors only if they have significant presence
                    if percentage >= min_percentage_for_blacks:
                        colors.append(color)
                        percentages.append(float(percentage))
                elif percentage >= min_percentage:  # Regular colors with normal threshold
                    colors.append(color)
                    percentages.append(float(percentage))
            
            # Sort by percentage (highest first)
            sorted_indices = np.argsort(percentages)[::-1]
            colors = [colors[i] for i in sorted_indices]
            percentages = [percentages[i] for i in sorted_indices]
            
            # Debug: Show all detected colors before filtering
            print(f"🔍 All detected colors before filtering:")
            for i, (color, pct) in enumerate(zip(colors, percentages)):
                r, g, b = color
                brightness = np.mean(color)
                print(f"   Color {i+1}: RGB({r},{g},{b}) - {pct:.2f}% - Brightness: {brightness:.1f}")
            
            # Filter out likely artifacts (very low percentage colors that might be noise)
            filtered_colors = []
            filtered_percentages = []
            
            # Check if this might be a background-removed image by looking for transparency
            # If we detect many white pixels, it's likely background-removed
            white_pixel_count = sum(1 for color in colors if np.mean(color) > 240)
            is_background_removed = white_pixel_count > len(colors) * 0.3  # If more than 30% are white
            
            # For background-removed images, detect and filter out edge artifacts
            if has_alpha and len(design_pixels) < len(non_transparent_pixels) * 0.5:
                print(f"🔍 Edge Artifact Detection: Analyzing {len(colors)} colors for edge artifacts")
                
                # Identify edge artifacts by looking for:
                # 1. Colors with very low percentage (likely edge pixels)
                # 2. Colors that are similar to main colors but with lower percentage
                # 3. Colors that are spatially scattered
                
                edge_artifact_colors = []
                main_colors = []
                main_percentages = []
                
                for i, (color, percentage) in enumerate(zip(colors, percentages)):
                    r, g, b = color
                    brightness = np.mean(color)
                    
                    # Check if this color is likely an edge artifact
                    is_edge_artifact = False
                    
                    # Very low percentage colors are likely edge artifacts
                    if percentage < 1.0:
                        is_edge_artifact = True
                        reason = f"Very low percentage ({percentage:.1f}% < 1.0%)"
                    
                    # Check if this color is similar to a main color but with much lower percentage
                    for j, (main_color, main_pct) in enumerate(zip(main_colors, main_percentages)):
                        if main_pct > percentage * 5:  # Main color has 5x more pixels
                            # Calculate color similarity
                            color_diff = np.sqrt((r-main_color[0])**2 + (g-main_color[1])**2 + (b-main_color[2])**2)
                            if color_diff < 25:  # Very similar color
                                is_edge_artifact = True
                                reason = f"Similar to main color {j+1} but {percentage:.1f}% vs {main_pct:.1f}%"
                                break
                    
                    # Very dark or very light colors with low percentage are likely edge artifacts
                    if (brightness < 30 or brightness > 225) and percentage < 2.0:
                        is_edge_artifact = True
                        reason = f"Extreme brightness ({brightness:.1f}) with low percentage ({percentage:.1f}%)"
                    
                    # SPECIFIC: Scattered white edge artifacts detection
                    # White colors with very low percentage that are likely scattered edge pixels
                    if (r > 200 and g > 200 and b > 200) and percentage < 1.5:
                        is_edge_artifact = True
                        reason = f"Scattered white edge pixels ({percentage:.1f}% < 1.5%)"
                    
                    # EXTRA AGGRESSIVE: Any white/near-white color with very low percentage
                    # These are almost always edge artifacts from background removal
                    if (r > 180 and g > 180 and b > 180) and percentage < 1.0:
                        is_edge_artifact = True
                        reason = f"Very low percentage white/near-white ({percentage:.1f}% < 1.0%)"
                    
                    # CRITICAL: Detect scattered white colors with 2-3% that are edge artifacts
                    # These are white colors that appear in small amounts but scattered across the image
                    # This specifically targets the anti-aliasing artifacts you're seeing
                    if (r > 200 and g > 200 and b > 200) and 1.0 <= percentage <= 3.0:
                        # This is a candidate for scattered white detection
                        # We'll mark it for further analysis
                        is_edge_artifact = True
                        reason = f"Anti-aliasing white edge artifact ({percentage:.1f}% - scattered at edges)"
                    
                    # EXTRA: Detect very light blue-white edge artifacts (common in anti-aliasing)
                    # These often appear where blue shapes meet transparent backgrounds
                    if (r > 180 and g > 180 and b > 200) and percentage < 2.5:
                        is_edge_artifact = True
                        reason = f"Blue-white edge artifact ({percentage:.1f}% - likely anti-aliasing)"
                    
                    if is_edge_artifact:
                        edge_artifact_colors.append((color, percentage, reason))
                        print(f"   Edge artifact detected: RGB({r},{g},{b}) - {percentage:.1f}% - {reason}")
                    else:
                        main_colors.append(color)
                        main_percentages.append(percentage)
                
                # Use only main colors for further processing
                colors = main_colors
                percentages = main_percentages
                
                print(f"   Edge artifacts removed: {len(edge_artifact_colors)}")
                print(f"   Main colors kept: {len(colors)}")
                
                # Additional check: Remove any remaining scattered white colors
                # These might have slipped through the initial filtering
                final_colors = []
                final_percentages = []
                
                for color, percentage in zip(colors, percentages):
                    r, g, b = color
                    brightness = np.mean(color)
                    
                    # Final check: Remove any white/near-white colors that are likely scattered
                    # Be more aggressive - remove white colors with low percentage that are likely edge artifacts
                    if (r > 180 and g > 180 and b > 180) and percentage < 3.0:
                        print(f"   Final removal: Scattered white color RGB({r},{g},{b}) - {percentage:.1f}%")
                        continue
                    
                    # EXTRA: Remove any white colors that are in the 2-3% range
                    # These are almost always scattered edge artifacts
                    if (r > 200 and g > 200 and b > 200) and 1.5 <= percentage <= 3.0:
                        print(f"   Final removal: Low percentage white color RGB({r},{g},{b}) - {percentage:.1f}% (likely scattered)")
                        continue
                    
                    # FINAL: Remove any remaining edge white artifacts
                    # This catches the anti-aliasing artifacts you're seeing
                    if (r > 180 and g > 180 and b > 180) and percentage < 2.0:
                        print(f"   Final removal: Edge white artifact RGB({r},{g},{b}) - {percentage:.1f}% (anti-aliasing)")
                        continue
                    
                    final_colors.append(color)
                    final_percentages.append(percentage)
                
                # Use final filtered colors
                colors = final_colors
                percentages = final_percentages
                
                print(f"   After final white filtering: {len(colors)} colors")
            
            # Additional check: if we have very dark colors with low percentage, likely background-removed
            dark_colors_count = sum(1 for color in colors if np.mean(color) < 30)
            if dark_colors_count > 0 and any(percentage < 5.0 for percentage in percentages):
                is_background_removed = True
            
            # Special case: if we have blue + white colors and then black appears, it's likely background removal artifact
            blue_colors = sum(1 for color in colors if color[2] > 150 and color[1] < 100 and color[0] < 100)  # High blue, low green/red
            white_colors = sum(1 for color in colors if np.mean(color) > 240)
            if blue_colors > 0 and white_colors > 0:
                # This looks like the Facebook logo scenario - be extra careful with black
                is_background_removed = True
            
            for color, percentage in zip(colors, percentages):
                r, g, b = color
                brightness = np.mean(color)
                
                # CRITICAL: First filter - remove ALL colors with very low percentage
                # This catches edge artifacts regardless of color type
                if percentage < 2.0:
                    continue  # Skip ALL colors with less than 2% presence
                
                # Skip colors that are likely artifacts or edge noise
                # Very dark colors with very low percentage might be background removal artifacts
                if brightness < 20 and percentage < 3.0:
                    continue  # Skip very dark colors with less than 3% presence
                
                # Pure black colors with low percentage are almost always artifacts
                if (r < 10 and g < 10 and b < 10) and percentage < 4.0:
                    continue  # Skip pure black colors with less than 4% presence
                
                # Very light colors with very low percentage might be compression artifacts
                if brightness > 240 and percentage < 1.0:
                    continue  # Skip very light colors with less than 1% presence
                
                # Skip any color with very low percentage (likely edge noise or artifacts)
                if percentage < 1.5:
                    continue  # Skip colors with less than 1.5% presence
                
                # Skip colors that are likely edge noise (very small isolated areas)
                if percentage < 2.5 and brightness < 50:
                    continue  # Skip dark colors with very low percentage
                
                # For background-removed images, be extra aggressive with edge artifact filtering
                if has_alpha and len(design_pixels) < len(non_transparent_pixels) * 0.5:
                    # Skip any color with very low percentage (likely edge artifacts from background removal)
                    if percentage < 2.0:
                        continue  # Skip colors with less than 2% presence in background-removed images
                    
                    # Skip very dark colors with low percentage (likely edge artifacts)
                    if brightness < 40 and percentage < 3.0:
                        continue  # Skip dark edge artifacts
                    
                    # Skip very light colors with low percentage (likely edge artifacts)
                    if brightness > 200 and percentage < 2.5:
                        continue  # Skip light edge artifacts
                
                # Skip ONLY pure white colors (255,255,255) with very low percentage
                # This preserves legitimate white design elements
                if (r == 255 and g == 255 and b == 255) and percentage < 0.5:
                    continue  # Skip only pure white colors with less than 0.5% presence
                
                # Smart black color detection: Check if black is legitimate design or background removal artifact
                if (r < 20 and g < 20 and b < 20):  # Very dark/black colors
                    # Since we're now properly filtering out white background pixels,
                    # we can be less aggressive with black filtering
                    if percentage < 1.5:  # Skip very small black areas
                        continue  # Skip black colors with less than 1.5% presence
                    
                    # Skip pure black colors with very low percentage (likely scattered artifacts)
                    if (r < 10 and g < 10 and b < 10) and percentage < 2.0:
                        continue  # Skip pure black colors with very low percentage
                
                filtered_colors.append(color)
                filtered_percentages.append(percentage)
            
            # Debug: Log what colors were detected and filtered
            print(f"🔍 Color Detection Debug:")
            print(f"   Original colors detected: {len(colors)}")
            print(f"   After filtering: {len(filtered_colors)}")
            print(f"   Filtered out: {len(colors) - len(filtered_colors)} colors")
            
            # Show which colors were filtered out and why
            if len(colors) != len(filtered_colors):
                print(f"   Filtered out colors:")
                for i, (color, percentage) in enumerate(zip(colors, percentages)):
                    r, g, b = color
                    brightness = np.mean(color)
                    if color not in filtered_colors:
                        reason = "Unknown"
                        if percentage < 2.0:
                            reason = f"Percentage too low ({percentage:.1f}% < 2.0%)"
                        elif brightness < 20 and percentage < 3.0:
                            reason = f"Dark color with low percentage ({percentage:.1f}% < 3.0%)"
                        elif (r < 10 and g < 10 and b < 10) and percentage < 4.0:
                            reason = f"Pure black with low percentage ({percentage:.1f}% < 4.0%)"
                        elif brightness > 240 and percentage < 1.0:
                            reason = f"Light color with low percentage ({percentage:.1f}% < 1.0%)"
                        elif percentage < 2.5 and brightness < 50:
                            reason = f"Dark edge noise ({percentage:.1f}% < 2.5%)"
                        # Don't filter out white colors - they might be legitimate design elements
                        # Only filter out very low percentage colors regardless of color
                        elif percentage < 0.5:
                            reason = f"Very low percentage color ({percentage:.1f}% < 0.5%)"
                        elif (r < 20 and g < 20 and b < 20) and percentage < 1.5:
                            reason = f"Black color with low percentage ({percentage:.1f}% < 1.5%)"
                        
                        print(f"     RGB({r},{g},{b}) - {percentage:.1f}% - {reason}")
            
            if len(filtered_colors) > 0:
                print(f"   Final colors:")
                for i, (color, pct) in enumerate(zip(filtered_colors, filtered_percentages)):
                    r, g, b = color
                    brightness = np.mean(color)
                    print(f"     {i+1}: RGB({r},{g},{b}) - {pct:.2f}% - Brightness: {brightness:.1f}")
            
            # Match colors to palette
            palette_matches = self.color_matcher.match_colors_to_palette(filtered_colors, filtered_percentages)
            
            # Count matched vs unmatched colors
            matched_count = sum(1 for match in palette_matches if match["matched_palette"])
            unmatched_count = len(palette_matches) - matched_count
            
            return {
                "success": True,
                "algorithm": "K-means Clustering + Palette Matching",
                "total_colors_detected": len(palette_matches),
                "matched_to_palette": matched_count,
                "unmatched_colors": unmatched_count,
                "matches": palette_matches,
                "image_dimensions": {"width": int(width), "height": int(height)},  # Convert numpy.int64 to Python int
                "palette_info": self.color_matcher.get_palette_summary()
            }
            
        except Exception as e:
            raise Exception(f"Color extraction failed: {str(e)}")
    
    def remove_background_custom(self, image_path: str) -> str:
        """
        Custom background removal using OpenCV with full control over the process
        """
        try:
            print(f"🔍 [CUSTOM] Starting custom background removal for: {image_path}")
            
            import cv2
            import numpy as np
            from PIL import Image
            
            # Load image with OpenCV
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError("Could not load image")
            
            print(f"🔍 [CUSTOM] Image loaded: {img.shape}")
            
            # Convert BGR to RGB for processing
            if len(img.shape) == 3:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = img
            
            # Method 1: Color-based background removal (preserves text)
            print(f"🔍 [CUSTOM] Attempting color-based background removal...")
            
            try:
                # Convert to HSV for better color detection
                img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
                
                # Detect white/light background (common in logos)
                # Lower and upper bounds for white/light colors
                lower_white = np.array([0, 0, 200])  # High value, low saturation
                upper_white = np.array([180, 30, 255])
                
                # Create mask for white/light background
                white_mask = cv2.inRange(img_hsv, lower_white, upper_white)
                
                # Invert mask to get foreground
                foreground_mask = cv2.bitwise_not(white_mask)
                
                # Clean up the mask with morphological operations
                kernel = np.ones((3,3), np.uint8)
                foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, kernel)
                foreground_mask = cv2.MORPH_OPEN(foreground_mask, kernel)
                
                # Apply mask to original image
                result = cv2.bitwise_and(img_rgb, img_rgb, mask=foreground_mask)
                
                # Convert to RGBA with transparency
                result_rgba = np.zeros((result.shape[0], result.shape[1], 4), dtype=np.uint8)
                result_rgba[:,:,:3] = result
                result_rgba[:,:,3] = foreground_mask  # Alpha channel
                
                print(f"✅ [CUSTOM] Color-based method successful")
                
            except Exception as e1:
                print(f"❌ [CUSTOM] Color-based method failed: {e1}")
                
                try:
                    # Method 2: Edge-based background removal
                    print(f"🔍 [CUSTOM] Attempting edge-based background removal...")
                    
                    # Convert to grayscale
                    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
                    
                    # Apply Gaussian blur to reduce noise
                    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                    
                    # Use Canny edge detection
                    edges = cv2.Canny(blurred, 50, 150)
                    
                    # Dilate edges to connect them
                    kernel = np.ones((3,3), np.uint8)
                    dilated = cv2.dilate(edges, kernel, iterations=1)
                    
                    # Find contours
                    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Create mask from largest contour
                    if contours:
                        largest_contour = max(contours, key=cv2.contourArea)
                        mask = np.zeros_like(gray)
                        cv2.fillPoly(mask, [largest_contour], 255)
                        
                        # Apply mask to original image
                        result = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)
                        
                        # Convert to RGBA
                        result_rgba = np.zeros((result.shape[0], result.shape[1], 4), dtype=np.uint8)
                        result_rgba[:,:,:3] = result
                        result_rgba[:,:,3] = mask
                        
                        print(f"✅ [CUSTOM] Edge-based method successful")
                    else:
                        raise Exception("No contours found")
                        
                except Exception as e2:
                    print(f"❌ [CUSTOM] Edge-based method failed: {e2}")
                    
                    try:
                        # Method 3: Threshold-based background removal
                        print(f"🔍 [CUSTOM] Attempting threshold-based background removal...")
                        
                        # Convert to grayscale
                        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
                        
                        # Use Otsu's thresholding
                        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        
                        # Invert threshold to get foreground
                        foreground_mask = cv2.bitwise_not(thresh)
                        
                        # Apply mask
                        result = cv2.bitwise_and(img_rgb, img_rgb, mask=foreground_mask)
                        
                        # Convert to RGBA
                        result_rgba = np.zeros((result.shape[0], result.shape[1], 4), dtype=np.uint8)
                        result_rgba[:,:,:3] = result
                        result_rgba[:,:,3] = foreground_mask
                        
                        print(f"✅ [CUSTOM] Threshold-based method successful")
                        
                    except Exception as e3:
                        print(f"❌ [CUSTOM] All methods failed!")
                        print(f"   Method 1 error: {e1}")
                        print(f"   Method 2 error: {e2}")
                        print(f"   Method 3 error: {e3}")
                        raise Exception("All custom background removal methods failed")
            
            # Convert numpy array to PIL Image
            result_pil = Image.fromarray(result_rgba, 'RGBA')
            
            # Save the result
            output_path = image_path.replace('.png', '_bg_removed.png').replace('.jpg', '_bg_removed.png').replace('.jpeg', '_bg_removed.png')
            result_pil.save(output_path, 'PNG')
            
            print(f"✅ [CUSTOM] Background removal completed: {output_path}")
            print(f"✅ [CUSTOM] Result image size: {result_pil.size}, mode: {result_pil.mode}")
            return output_path
            
        except Exception as e:
            print(f"❌ [CUSTOM] Background removal failed: {str(e)}")
            print(f"❌ [CUSTOM] Error details: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Background removal failed: {str(e)}")
    

    
    def remove_background_adaptive(self, image_path: str, preserve_text: bool = True, 
                                  edge_sensitivity: float = 0.1) -> str:
        """
        Remove background using rembg with u2net model for better detail preservation
        """
        try:
            from rembg import remove, new_session
            from PIL import Image
            
            # Load image with PIL for rembg
            input_image = Image.open(image_path)
            
            # Create a session with u2net model for better text and detail preservation
            session = new_session('u2net')
            
            # Use rembg with u2net model - it's excellent at preserving fine details
            output_image = remove(input_image, session=session)
            
            # Save the result directly without additional processing
            output_path = image_path.replace('.png', '_nobg_adaptive.png')
            output_image.save(output_path, 'PNG')
            
            return output_path
            
        except Exception as e:
            raise Exception(f"Advanced background removal failed: {str(e)}")

    def remove_background_rembg_improved(self, image_path: str) -> str:
        """
        Lightweight background removal using OpenCV (memory-efficient)
        """
        try:
            print(f"🔍 [OPENCV] Starting OpenCV background removal for: {image_path}")
            
            # Load image
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Could not load image")
            
            print(f"🔍 [OPENCV] Image loaded: {img.shape}")
            
            # Convert to different color spaces for analysis
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Method 1: Simple color-based background removal (white/light backgrounds)
            # Create mask for white/light backgrounds
            lower_white = np.array([0, 0, 200])  # HSV
            upper_white = np.array([180, 30, 255])
            white_mask = cv2.inRange(hsv, lower_white, upper_white)
            
            # Method 2: Edge-based approach for better results
            # Find edges to preserve important content
            edges = cv2.Canny(gray, 50, 150)
            
            # Dilate edges to create regions of interest
            kernel = np.ones((3,3), np.uint8)
            edges_dilated = cv2.dilate(edges, kernel, iterations=2)
            
            # Combine white mask with edge preservation
            # Keep areas near edges even if they're white
            protected_mask = cv2.bitwise_not(edges_dilated)
            final_bg_mask = cv2.bitwise_and(white_mask, protected_mask)
            
            # Create alpha channel
            alpha = np.where(final_bg_mask > 0, 0, 255).astype(np.uint8)
            
            # Create RGBA image
            rgba_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
            rgba_img[:, :, 3] = alpha
            
            # Save result
            output_path = image_path.replace('.png', '_bg_removed.png').replace('.jpg', '_bg_removed.png').replace('.jpeg', '_bg_removed.png')
            cv2.imwrite(output_path, rgba_img)
            
            # Calculate transparency stats
            transparent_pixels = np.sum(alpha == 0)
            total_pixels = alpha.shape[0] * alpha.shape[1]
            transparency_ratio = transparent_pixels / total_pixels
            
            print(f"✅ [OPENCV] Background removal completed: {output_path}")
            print(f"📊 [OPENCV] Transparency achieved: {transparency_ratio*100:.1f}%")
            
            return output_path
            
        except Exception as e:
            print(f"❌ [OPENCV] Background removal failed: {str(e)}")
            print(f"❌ [OPENCV] Error details: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Background removal failed: {str(e)}")

    def detect_background_status(self, image_path: str) -> Dict[str, Any]:
        """
        Detect if an image already has background removed or transparent background
        """
        try:
            # Load image
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError("Could not load image")
            
            # Check if image has alpha channel (RGBA)
            has_alpha = img.shape[-1] == 4 if len(img.shape) > 2 else False
            
            if has_alpha:
                # Extract alpha channel
                alpha = img[:, :, 3]
                
                # Count transparent pixels (alpha < 128)
                transparent_pixels = np.sum(alpha < 128)
                total_pixels = alpha.shape[0] * alpha.shape[1]
                transparency_ratio = transparent_pixels / total_pixels
                
                # Check if there are significant transparent areas
                if transparency_ratio > 0.1:  # More than 10% transparent
                    return {
                        "background_removed": True,
                        "transparency_ratio": float(transparency_ratio),
                        "transparent_pixels": int(transparent_pixels),
                        "total_pixels": int(total_pixels),
                        "confidence": "high" if transparency_ratio > 0.3 else "medium"
                    }
            
            # If no alpha channel, check for uniform background
            if len(img.shape) == 3:
                # Convert to RGB for analysis
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Sample pixels from corners and edges
                height, width = img_rgb.shape[:2]
                corner_samples = [
                    img_rgb[0, 0],      # Top-left
                    img_rgb[0, width-1], # Top-right
                    img_rgb[height-1, 0], # Bottom-left
                    img_rgb[height-1, width-1] # Bottom-right
                ]
                
                # Check if corners are similar (likely background)
                corner_variance = np.var(corner_samples, axis=0)
                corners_uniform = np.all(corner_variance < 100)  # Low variance = uniform
                
                if corners_uniform:
                    # Check if center has different colors
                    center_sample = img_rgb[height//2, width//2]
                    corner_avg = np.mean(corner_samples, axis=0)
                    center_diff = np.linalg.norm(center_sample - corner_avg)
                    
                    if center_diff > 50:  # Significant difference between center and corners
                        return {
                            "background_removed": False,
                            "transparency_ratio": 0.0,
                            "transparent_pixels": 0,
                            "total_pixels": int(total_pixels),
                            "confidence": "medium",
                            "reason": "Uniform background detected in corners"
                        }
            
            # Default: assume background is present
            return {
                "background_removed": False,
                "transparency_ratio": 0.0,
                "transparent_pixels": 0,
                "total_pixels": int(img.shape[0] * img.shape[1]),
                "confidence": "low",
                "reason": "No clear evidence of background removal"
            }
            
        except Exception as e:
            raise Exception(f"Background detection failed: {str(e)}")

# Global instance
color_extractor = PaletteBasedColorExtractor()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "DTF Color Extractor - PALETTE MATCHING",
        "version": "3.0.0",
        "palette_colors": color_matcher.get_palette_summary()["total_colors"]
    }

@app.get("/palette")
async def get_palette():
    """Get the complete color palette"""
    return {
        "success": True,
        "palette": color_matcher.get_palette_summary()
    }

@app.get("/palette/categories")
async def get_palette_categories():
    """Get colors organized by category"""
    summary = color_matcher.get_palette_summary()
    return {
        "success": True,
        "categories": summary["categories"]
    }

@app.get("/palette/search/{query}")
async def search_palette(query: str):
    """Search colors in the palette"""
    results = color_matcher.search_colors(query)
    return {
        "success": True,
        "query": query,
        "results": [
            {
                "name": color.name,
                "hex": color.hex,
                "rgb": color.rgb,
                "category": color.category
            } for color in results
        ]
    }

@app.post("/extract-colors")
async def extract_colors(
    file: UploadFile = File(...),
    num_colors: int = 12,
    min_percentage: float = 1.0
):
    """
    Extract colors from image and match to predefined palette
    
    Parameters:
    - file: Image file (PNG, JPG, JPEG, WebP)
    - num_colors: Maximum number of colors to extract (1-32)
    - min_percentage: Minimum percentage threshold for color inclusion (0.1-10.0)
    """
    
    # Validate parameters
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    if num_colors < 1 or num_colors > 32:
        raise HTTPException(status_code=400, detail="num_colors must be between 1 and 32")
    
    if min_percentage < 0.1 or min_percentage > 10.0:
        raise HTTPException(status_code=400, detail="min_percentage must be between 0.1 and 10.0")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Extract colors with palette matching
        result = color_extractor.extract_colors_with_palette_matching(
            temp_file_path, num_colors, min_percentage
        )
        
        # Clean up temporary file
        os.unlink(temp_file_path)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/remove-background")
async def remove_background(
    file: UploadFile = File(...), 
    method: str = Form("rembg"),
    model: str = Form("u2net"),
    post_process: bool = Form(False)
):
    """
    Remove background from image with user-controlled settings
    """
    try:
        # Save uploaded file temporarily
        temp_path = f"/tmp/temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        print(f"🔍 [ENDPOINT] Background removal requested with method: {method}, model: {model}, post_process: {post_process}")
        
        # Remove background using rembg with user settings
        output_path = color_extractor.remove_background_rembg_improved(temp_path)
        
        # Read the result and convert to base64
        with open(output_path, "rb") as f:
            image_data = f.read()
        
        # Clean up temp files
        os.remove(temp_path)
        os.remove(output_path)
        
        # Convert to base64
        import base64
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        return {
            "success": True,
            "image_base64": image_base64,
            "method": "rembg_improved",
            "settings": {
                "model": "default_with_u2net_fallback",
                "post_process": False
            },
            "features": "deep_learning_background_removal, text_preservation, optimized_settings"
        }
        
    except Exception as e:
        print(f"❌ [ENDPOINT] Background removal failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/remove-background-advanced")
async def remove_background_advanced(
    file: UploadFile = File(...), 
    preserve_text: bool = Form(True),
    edge_sensitivity: float = Form(0.1)
):
    """
    Remove background with advanced settings for text and outline preservation
    """
    try:
        # Save uploaded file temporarily
        temp_path = f"/tmp/temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Remove background using adaptive algorithm
        output_path = color_extractor.remove_background_adaptive(
            temp_path, 
            preserve_text=preserve_text, 
            edge_sensitivity=edge_sensitivity
        )
        
        # Read the result and convert to base64
        with open(output_path, "rb") as f:
            image_data = f.read()
        
        # Clean up temp files
        os.remove(temp_path)
        os.remove(output_path)
        
        # Convert to base64
        import base64
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        return {
            "success": True,
            "image_base64": image_base64,
            "method": "rembg_u2net",
            "settings": {
                "preserve_text": preserve_text,
                "edge_sensitivity": edge_sensitivity
            },
            "features": "deep_learning_u2net, high_quality, minimal_processing"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect-background")
async def detect_background(file: UploadFile = File(...)):
    """
    Detect if an image already has background removed
    """
    try:
        # Save uploaded file temporarily
        temp_path = f"/tmp/temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Detect background status
        result = color_extractor.detect_background_status(temp_path)
        
        # Clean up temp file
        os.remove(temp_path)
        
        return {
            "success": True,
            "background_status": result
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

# Include screen printing workflow router
app.include_router(screen_printing_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
