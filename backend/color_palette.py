"""
Predefined Color Palette for Screen Printing/DTF
Industry-standard colors with similarity matching
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class PaletteColor:
    """Represents a color in our predefined palette"""
    name: str
    hex: str
    rgb: Tuple[int, int, int]
    category: str
    print_ready: bool = True

class ColorPaletteMatcher:
    """Matches detected colors to predefined palette colors"""
    
    def __init__(self):
        self.palette = self._initialize_palette()
        self.similarity_threshold = 35.0  # Increased Delta E threshold for better white detection
    
    def _initialize_palette(self) -> List[PaletteColor]:
        """Initialize the predefined color palette"""
        return [
            # Whites & Grays
            PaletteColor("Pure White", "#ffffff", (255, 255, 255), "Neutral"),
            PaletteColor("White", "#fefefe", (254, 254, 254), "Neutral"),
            PaletteColor("Off White", "#fafafa", (250, 250, 250), "Neutral"),
            PaletteColor("Light White", "#f5f5f5", (245, 245, 245), "Neutral"),
            PaletteColor("Very Light Gray", "#f0f0f0", (240, 240, 240), "Neutral"),
            PaletteColor("Cream", "#fffdd0", (255, 253, 208), "Neutral"),
            PaletteColor("Light Gray", "#d3d3d3", (211, 211, 211), "Neutral"),
            PaletteColor("Silver", "#c0c0c0", (192, 192, 192), "Neutral"),
            PaletteColor("Pure Black", "#000000", (0, 0, 0), "Neutral"),
            PaletteColor("Deep Black", "#0a0a0a", (10, 10, 10), "Neutral"),
            PaletteColor("Charcoal", "#646a69", (100, 106, 105), "Neutral"),
            PaletteColor("Gray", "#99999a", (153, 153, 154), "Neutral"),
            PaletteColor("Ice Gray", "#bdbbbb", (189, 187, 187), "Neutral"),
            
            # Pinks & Reds
            PaletteColor("Hot Pink", "#ff0091", (255, 0, 145), "Pink/Red"),
            PaletteColor("Pink", "#ffafbe", (255, 175, 190), "Pink/Red"),
            PaletteColor("Charity Pink", "#ff8cbe", (255, 140, 190), "Pink/Red"),
            PaletteColor("Magenta", "#b4468c", (180, 70, 140), "Pink/Red"),
            PaletteColor("Maroon", "#7d2d3c", (125, 45, 60), "Pink/Red"),
            PaletteColor("Vibrant Red", "#ef3340", (239, 51, 64), "Pink/Red"),
            PaletteColor("Cardinal", "#960032", (150, 0, 50), "Pink/Red"),
            PaletteColor("Red", "#c80f28", (200, 15, 40), "Pink/Red"),
            
            # Oranges & Yellows
            PaletteColor("Orange", "#fa4b0f", (250, 75, 15), "Orange/Yellow"),
            PaletteColor("Team Orange", "#ff8200", (255, 130, 0), "Orange/Yellow"),
            PaletteColor("Athletic Gold", "#ffb419", (255, 180, 25), "Orange/Yellow"),
            PaletteColor("Gold", "#ffc828", (255, 200, 40), "Orange/Yellow"),
            PaletteColor("Yellow", "#ffdc00", (255, 220, 0), "Orange/Yellow"),
            PaletteColor("Lemon", "#faeb5f", (250, 235, 95), "Orange/Yellow"),
            PaletteColor("Old Gold", "#8c6923", (140, 105, 35), "Orange/Yellow"),
            
            # Greens
            PaletteColor("Mint", "#a2e4b8", (162, 228, 184), "Green"),
            PaletteColor("Vibrant Lime", "#a5e100", (165, 225, 0), "Green"),
            PaletteColor("Green", "#3ca51e", (60, 165, 30), "Green"),
            PaletteColor("Kelly", "#006937", (0, 105, 55), "Green"),
            PaletteColor("Forest", "#2d5032", (45, 80, 50), "Green"),
            
            # Blues & Teals
            PaletteColor("Teal", "#007378", (0, 115, 120), "Blue/Teal"),
            PaletteColor("Turquoise", "#009bb4", (0, 155, 180), "Blue/Teal"),
            PaletteColor("Sky Blue", "#23aff0", (35, 175, 240), "Blue/Teal"),
            PaletteColor("Baby Blue", "#91beeb", (145, 190, 235), "Blue/Teal"),
            PaletteColor("Blue", "#005fa0", (0, 95, 160), "Blue/Teal"),
            PaletteColor("Vibrant Blue", "#0047bb", (0, 71, 187), "Blue/Teal"),
            PaletteColor("Royal", "#003c82", (0, 60, 130), "Blue/Teal"),
            PaletteColor("Navy", "#00325a", (0, 50, 90), "Blue/Teal"),
            
            # Purples
            PaletteColor("Lavender", "#aa7dc8", (170, 125, 200), "Purple"),
            PaletteColor("Grape", "#bb29bb", (187, 41, 187), "Purple"),
            PaletteColor("Purple", "#5a3287", (90, 50, 135), "Purple"),
            PaletteColor("Plum", "#641e64", (100, 30, 100), "Purple"),
            
            # Browns & Neutrals
            PaletteColor("Deep Brown", "#402d1c", (64, 45, 28), "Brown/Neutral"),
            PaletteColor("Brown", "#82502d", (130, 80, 45), "Brown/Neutral"),
            PaletteColor("Bronze", "#a66f41", (166, 111, 65), "Brown/Neutral"),
            PaletteColor("Sand", "#cda073", (205, 160, 115), "Brown/Neutral"),
            PaletteColor("Apricot", "#ffbe78", (255, 190, 120), "Brown/Neutral"),
            PaletteColor("Warm Ivory", "#eed8ac", (238, 216, 172), "Brown/Neutral"),
        ]
    
    def rgb_to_lab(self, rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
        """Convert RGB to LAB color space for accurate similarity calculation"""
        # Normalize RGB values
        r, g, b = rgb[0]/255.0, rgb[1]/255.0, rgb[2]/255.0
        
        # Apply gamma correction
        r = r ** 2.2 if r > 0.04045 else r / 12.92
        g = g ** 2.2 if g > 0.04045 else g / 12.92
        b = b ** 2.2 if b > 0.04045 else b / 12.92
        
        # Convert to XYZ using sRGB matrix
        x = r * 0.4124 + g * 0.3576 + b * 0.1805
        y = r * 0.2126 + g * 0.7152 + b * 0.0722
        z = r * 0.0193 + g * 0.1192 + b * 0.9505
        
        # Normalize XYZ
        x = x / 0.95047
        z = z / 1.08883
        
        # Convert to LAB
        x = x ** (1/3) if x > 0.008856 else (7.787 * x) + (16/116)
        y = y ** (1/3) if y > 0.008856 else (7.787 * y) + (16/116)
        z = z ** (1/3) if z > 0.008856 else (7.787 * z) + (16/116)
        
        l = (116 * y) - 16
        a = 500 * (x - y)
        b = 200 * (y - z)
        
        return (l, a, b)
    
    def delta_e_2000(self, lab1: Tuple[float, float, float], lab2: Tuple[float, float, float]) -> float:
        """Calculate Delta E 2000 (CIE2000) color difference - industry standard"""
        l1, a1, b1 = lab1
        l2, a2, b2 = lab2
        
        # Weighting factors
        kL, kC, kH = 1, 1, 1
        
        # Calculate C1, C2
        C1 = np.sqrt(a1**2 + b1**2)
        C2 = np.sqrt(a2**2 + b2**2)
        Cb = (C1 + C2) / 2
        
        # Calculate G
        G = 0.5 * (1 - np.sqrt(Cb**7 / (Cb**7 + 25**7)))
        
        # Calculate a1', a2'
        a1_prime = a1 * (1 + G)
        a2_prime = a2 * (1 + G)
        
        # Calculate C1', C2'
        C1_prime = np.sqrt(a1_prime**2 + b1**2)
        C2_prime = np.sqrt(a2_prime**2 + b2**2)
        Cb_prime = (C1_prime + C2_prime) / 2
        
        # Calculate h1', h2'
        h1_prime = np.arctan2(b1, a1_prime)
        h2_prime = np.arctan2(b2, a2_prime)
        
        # Handle hue differences
        dH_prime = h2_prime - h1_prime
        if abs(dH_prime) > np.pi:
            if dH_prime > 0:
                dH_prime -= 2 * np.pi
            else:
                dH_prime += 2 * np.pi
        
        # Calculate H'
        if abs(h1_prime - h2_prime) <= np.pi:
            H_prime = (h1_prime + h2_prime) / 2
        else:
            H_prime = (h1_prime + h2_prime + 2 * np.pi) / 2
        
        # Calculate T
        T = 1 - 0.17 * np.cos(H_prime - np.pi/6) + 0.24 * np.cos(2*H_prime) + 0.32 * np.cos(3*H_prime + np.pi/30) - 0.2 * np.cos(4*H_prime - np.pi/20)
        
        # Calculate SL, SC, SH
        SL = 1 + (0.015 * (l1 + l2)**2) / (20 + (l1 + l2)**2)
        SC = 1 + 0.045 * Cb_prime
        SH = 1 + 0.015 * Cb_prime * T
        
        # Calculate RT
        RT = -2 * np.sqrt(Cb_prime**7 / (Cb_prime**7 + 25**7)) * np.sin(2 * np.pi/3 * np.exp(-((H_prime * 180/np.pi - 275)/25)**2))
        
        # Calculate Delta E
        dL = l2 - l1
        dC = C2_prime - C1_prime
        dH = 2 * np.sqrt(C1_prime * C2_prime) * np.sin(dH_prime / 2)
        
        delta_E = np.sqrt(
            (dL / (kL * SL))**2 + 
            (dC / (kC * SC))**2 + 
            (dH / (kH * SH))**2 + 
            RT * (dC / (kC * SC)) * (dH / (kH * SH))
        )
        
        return delta_E
    
    def find_closest_palette_color(self, detected_rgb: Tuple[int, int, int]) -> Optional[PaletteColor]:
        """Find the closest palette color to a detected color"""
        detected_lab = self.rgb_to_lab(detected_rgb)
        
        best_match = None
        best_distance = float('inf')
        
        # Special handling for very light and very dark colors (whites, blacks, and near-whites/blacks)
        r, g, b = detected_rgb
        is_very_light = (r > 235 and g > 235 and b > 235)
        is_very_dark = (r < 20 and g < 20 and b < 20)
        is_pure_white = (r > 250 and g > 250 and b > 250)
        is_pure_black = (r < 5 and g < 5 and b < 5)
        
        for palette_color in self.palette:
            palette_lab = self.rgb_to_lab(palette_color.rgb)
            distance = self.delta_e_2000(detected_lab, palette_lab)
            
            # Use more lenient threshold for very light and very dark colors
            threshold = self.similarity_threshold
            if is_very_light or is_very_dark:
                threshold = 50.0  # Much more lenient for whites and blacks
            if is_pure_white or is_pure_black:
                threshold = 60.0  # Extremely lenient for pure white/black
            
            if distance < best_distance and distance <= threshold:
                best_distance = distance
                best_match = palette_color
        
        return best_match
    
    def match_colors_to_palette(self, detected_colors: List[Tuple[int, int, int]], 
                               percentages: List[float]) -> List[Dict]:
        """Match detected colors to palette colors and return consolidated matches"""
        # Group detected colors by their closest palette match
        palette_groups = {}
        
        for i, (detected_rgb, percentage) in enumerate(zip(detected_colors, percentages)):
            closest_palette = self.find_closest_palette_color(detected_rgb)
            
            if closest_palette:
                palette_key = closest_palette.name
                
                if palette_key not in palette_groups:
                    # First time seeing this palette color
                    palette_groups[palette_key] = {
                        "palette_color": closest_palette,
                        "detected_colors": [],
                        "total_percentage": 0.0,
                        "best_similarity": float('inf')
                    }
                
                # Add this detected color to the group
                similarity = self.delta_e_2000(
                    self.rgb_to_lab(detected_rgb), 
                    self.rgb_to_lab(closest_palette.rgb)
                )
                
                palette_groups[palette_key]["detected_colors"].append({
                    "rgb": [int(c) for c in detected_rgb],
                    "hex": self.rgb_to_hex(detected_rgb),
                    "percentage": float(percentage),
                    "similarity": float(similarity)
                })
                
                palette_groups[palette_key]["total_percentage"] += float(percentage)
                
                # Track the best (lowest) similarity score
                if similarity < palette_groups[palette_key]["best_similarity"]:
                    palette_groups[palette_key]["best_similarity"] = float(similarity)
            else:
                # No close match found - add as unmatched
                if "unmatched" not in palette_groups:
                    palette_groups["unmatched"] = {
                        "palette_color": None,
                        "detected_colors": [],
                        "total_percentage": 0.0,
                        "best_similarity": None
                    }
                
                palette_groups["unmatched"]["detected_colors"].append({
                    "rgb": [int(c) for c in detected_rgb],
                    "hex": self.rgb_to_hex(detected_rgb),
                    "percentage": float(percentage),
                    "similarity": None
                })
                palette_groups["unmatched"]["total_percentage"] += float(percentage)
        
        # Convert groups to final result format
        matches = []
        
        for group_key, group_data in palette_groups.items():
            if group_key == "unmatched":
                # Handle unmatched colors
                for detected_color in group_data["detected_colors"]:
                    matches.append({
                        "detected_rgb": detected_color["rgb"],
                        "detected_hex": detected_color["hex"],
                        "detected_percentage": detected_color["percentage"],
                        "matched_palette": None,
                        "similarity_score": None,
                        "consolidated": False
                    })
            else:
                # Handle matched palette colors
                palette_color = group_data["palette_color"]
                detected_colors = group_data["detected_colors"]
                total_percentage = group_data["total_percentage"]
                best_similarity = group_data["best_similarity"]
                
                if len(detected_colors) == 1:
                    # Single color match - no consolidation needed
                    detected_color = detected_colors[0]
                    matches.append({
                        "detected_rgb": detected_color["rgb"],
                        "detected_hex": detected_color["hex"],
                        "detected_percentage": detected_color["percentage"],
                        "matched_palette": {
                            "name": palette_color.name,
                            "hex": palette_color.hex,
                            "rgb": [int(c) for c in palette_color.rgb],
                            "category": palette_color.category,
                            "print_ready": palette_color.print_ready
                        },
                        "similarity_score": detected_color["similarity"],
                        "consolidated": False
                    })
                else:
                    # Multiple colors match the same palette - consolidate them
                    matches.append({
                        "detected_rgb": [int(c) for c in palette_color.rgb],  # Use palette color as representative
                        "detected_hex": palette_color.hex,
                        "detected_percentage": total_percentage,
                        "matched_palette": {
                            "name": palette_color.name,
                            "hex": palette_color.hex,
                            "rgb": [int(c) for c in palette_color.rgb],
                            "category": palette_color.category,
                            "print_ready": palette_color.print_ready
                        },
                        "similarity_score": best_similarity,
                        "consolidated": True,
                        "consolidated_details": {
                            "original_colors_count": len(detected_colors),
                            "original_colors": detected_colors
                        }
                    })
        
        # Sort by percentage (highest first)
        matches.sort(key=lambda x: x["detected_percentage"], reverse=True)
        return matches
    
    def rgb_to_hex(self, rgb: Tuple[int, int, int]) -> str:
        """Convert RGB tuple to HEX string"""
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
    
    def get_palette_summary(self) -> Dict:
        """Get summary of the color palette"""
        categories = {}
        for color in self.palette:
            if color.category not in categories:
                categories[color.category] = []
            categories[color.category].append({
                "name": color.name,
                "hex": color.hex,
                "rgb": color.rgb
            })
        
        return {
            "total_colors": len(self.palette),
            "categories": categories,
            "print_ready": all(color.print_ready for color in self.palette)
        }
    
    def get_colors_by_category(self, category: str) -> List[PaletteColor]:
        """Get all colors in a specific category"""
        return [color for color in self.palette if color.category == category]
    
    def search_colors(self, query: str) -> List[PaletteColor]:
        """Search colors by name or hex"""
        query = query.lower()
        results = []
        
        for color in self.palette:
            if (query in color.name.lower() or 
                query in color.hex.lower() or
                query in str(color.rgb)):
                results.append(color)
        
        return results

# Global instance
color_matcher = ColorPaletteMatcher()
