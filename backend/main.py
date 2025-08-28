from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import io
import tempfile
import os
from typing import List, Dict, Any
import json

app = FastAPI(title="DTF Color Extractor API", version="1.0.0")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ColorExtractor:
    def __init__(self):
        self.color_names = {
            # Common color names for better identification
            (255, 0, 0): "Red", (0, 255, 0): "Green", (0, 0, 255): "Blue",
            (255, 255, 0): "Yellow", (255, 0, 255): "Magenta", (0, 255, 255): "Cyan",
            (255, 255, 255): "White", (0, 0, 0): "Black", (128, 128, 128): "Gray"
        }
    
    def rgb_to_hex(self, rgb: tuple) -> str:
        """Convert RGB tuple to HEX string"""
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
    
    def rgb_to_lab(self, rgb: tuple) -> tuple:
        """Convert RGB to LAB color space for better color similarity"""
        # Simple RGB to LAB approximation
        r, g, b = rgb[0]/255, rgb[1]/255, rgb[2]/255
        
        # Convert to XYZ
        r = r ** 2.2 if r > 0.04045 else r / 12.92
        g = g ** 2.2 if g > 0.04045 else g / 12.92
        b = b ** 2.2 if b > 0.04045 else b / 12.92
        
        x = r * 0.4124 + g * 0.3576 + b * 0.1805
        y = r * 0.2126 + g * 0.7152 + b * 0.0722
        z = r * 0.0193 + g * 0.1192 + b * 0.9505
        
        # Convert to LAB
        x = x / 0.95047
        z = z / 1.08883
        
        x = x ** (1/3) if x > 0.008856 else (7.787 * x) + (16/116)
        y = y ** (1/3) if y > 0.008856 else (7.787 * y) + (16/116)
        z = z ** (1/3) if z > 0.008856 else (7.787 * z) + (16/116)
        
        l = (116 * y) - 16
        a = 500 * (x - y)
        b = 200 * (y - z)
        
        return (l, a, b)
    
    def color_similarity(self, color1: tuple, color2: tuple) -> float:
        """Calculate color similarity using LAB color space"""
        lab1 = self.rgb_to_lab(color1)
        lab2 = self.rgb_to_lab(color2)
        
        # Euclidean distance in LAB space
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(lab1, lab2)))
    
    def merge_similar_colors(self, colors: List[tuple], percentages: List[float], threshold: float = 30.0) -> tuple:
        """Merge colors that are visually similar"""
        if len(colors) <= 1:
            return colors, percentages
        
        merged_colors = []
        merged_percentages = []
        used = set()
        
        for i, (color, percentage) in enumerate(zip(colors, percentages)):
            if i in used:
                continue
                
            similar_colors = [color]
            similar_percentages = [percentage]
            used.add(i)
            
            for j, (other_color, other_percentage) in enumerate(zip(colors, percentages)):
                if j in used:
                    continue
                    
                if self.color_similarity(color, other_color) < threshold:
                    similar_colors.append(other_color)
                    similar_percentages.append(other_percentage)
                    used.add(j)
            
            # Average the similar colors
            avg_color = tuple(int(np.mean([c[i] for c in similar_colors])) for i in range(3))
            total_percentage = sum(similar_percentages)
            
            merged_colors.append(avg_color)
            merged_percentages.append(total_percentage)
        
        return merged_colors, merged_percentages
    
    def extract_colors_advanced(self, image_path: str, num_colors: int = 12, min_percentage: float = 1.0, merge_threshold: float = 30.0) -> Dict[str, Any]:
        """Advanced color extraction using multiple algorithms"""
        try:
            # Load image with OpenCV
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("Could not load image")
            
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Reshape for clustering
            pixels = img_rgb.reshape(-1, 3)
            
            # Remove fully transparent/white/black pixels if they dominate
            # Filter out very light or very dark pixels that might be noise
            brightness = np.mean(pixels, axis=1)
            valid_pixels = pixels[(brightness > 10) & (brightness < 245)]
            
            if len(valid_pixels) == 0:
                valid_pixels = pixels
            
            # Advanced K-means clustering
            kmeans = KMeans(
                n_clusters=min(num_colors * 2, len(valid_pixels)), 
                random_state=42, 
                n_init=10,
                max_iter=300
            )
            kmeans.fit(valid_pixels)
            
            # Get cluster centers and labels
            colors = kmeans.cluster_centers_.astype(int)
            labels = kmeans.labels_
            
            # Calculate actual percentages
            unique, counts = np.unique(labels, return_counts=True)
            percentages = (counts / len(valid_pixels)) * 100
            
            # Sort by percentage (most common first)
            sorted_indices = np.argsort(percentages)[::-1]
            colors = colors[sorted_indices]
            percentages = percentages[sorted_indices]
            
            # Filter by minimum percentage
            significant_mask = percentages >= min_percentage
            colors = colors[significant_mask]
            percentages = percentages[significant_mask]
            
            # Merge similar colors
            colors, percentages = self.merge_similar_colors(colors, percentages, merge_threshold)
            
            # Limit to requested number of colors
            colors = colors[:num_colors]
            percentages = percentages[:num_colors]
            
            # Normalize percentages
            total_percentage = sum(percentages)
            if total_percentage > 0:
                percentages = [p / total_percentage * 100 for p in percentages]
            
            # Prepare response
            color_data = []
            for color, percentage in zip(colors, percentages):
                color_info = {
                    "rgb": tuple(color),
                    "hex": self.rgb_to_hex(color),
                    "percentage": round(percentage, 2),
                    "name": self.color_names.get(tuple(color), "Custom")
                }
                color_data.append(color_info)
            
            return {
                "success": True,
                "colors": color_data,
                "total_colors": len(color_data),
                "total_pixels_analyzed": len(valid_pixels),
                "algorithm": "Advanced K-means with LAB similarity merging",
                "settings": {
                    "requested_colors": num_colors,
                    "min_percentage": min_percentage,
                    "merge_threshold": merge_threshold
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "colors": [],
                "total_colors": 0
            }

# Initialize color extractor
color_extractor = ColorExtractor()

@app.get("/")
async def root():
    return {"message": "DTF Color Extractor API", "version": "1.0.0"}

@app.post("/extract-colors")
async def extract_colors(
    file: UploadFile = File(...),
    num_colors: int = 12,
    min_percentage: float = 1.0,
    merge_threshold: float = 30.0
):
    """
    Extract dominant colors from uploaded image
    
    Parameters:
    - file: Image file (PNG, JPG, JPEG, WebP)
    - num_colors: Maximum number of colors to extract (1-20)
    - min_percentage: Minimum percentage threshold (0.1-10.0)
    - merge_threshold: Color similarity threshold for merging (10-100)
    """
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Validate parameters
    if not 1 <= num_colors <= 20:
        raise HTTPException(status_code=400, detail="num_colors must be between 1 and 20")
    
    if not 0.1 <= min_percentage <= 10.0:
        raise HTTPException(status_code=400, detail="min_percentage must be between 0.1 and 10.0")
    
    if not 10 <= merge_threshold <= 100:
        raise HTTPException(status_code=400, detail="merge_threshold must be between 10 and 100")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Extract colors
        result = color_extractor.extract_colors_advanced(
            temp_file_path, 
            num_colors, 
            min_percentage, 
            merge_threshold
        )
        
        # Clean up temporary file
        os.unlink(temp_file_path)
        
        if result["success"]:
            return JSONResponse(content=result)
        else:
            raise HTTPException(status_code=500, detail=result["error"])
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "DTF Color Extractor"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
