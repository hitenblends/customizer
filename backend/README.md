# Background Removal System

This directory contains a robust background removal solution using the `rembg` library with OpenCV post-processing for clean, production-ready results.

## 🚀 Features

- **Deep Learning Background Removal**: Uses `rembg`'s U2Net model for accurate background detection
- **OpenCV Post-Processing**: Advanced edge refinement and artifact removal
- **Color Decontamination**: Eliminates background color bleeding
- **Soft Edges**: Configurable alpha blur for better printing results
- **Fallback Support**: Automatic fallback to OpenCV method if rembg fails

## 📁 Files

- **`background_remover.py`** - Main BackgroundRemover class
- **`cli.py`** - Command-line interface
- **`main.py`** - FastAPI integration (updated)
- **`requirements.txt`** - Dependencies
- **`test_background_remover.py`** - Test script

## 🛠️ Installation

1. **Install dependencies**:
   ```bash
   pip3 install -r requirements.txt
   ```

2. **Verify installation**:
   ```bash
   python3 test_background_remover.py
   ```

## 🎯 Usage

### As a Python Class

```python
from background_remover import BackgroundRemover

# Initialize with default settings
remover = BackgroundRemover()

# Process a single image
result = remover.remove_background("input.png", "output.png")

# Process a folder
remover.process_folder("input_folder/", "output_folder/")
```

### Command Line Interface

```bash
# Process single file
python3 cli.py input.png -o output.png

# Process folder
python3 cli.py input_folder/ -o output_folder/

# Custom settings
python3 cli.py input.png -o output.png --strength 0.8 --no-blur
```

### FastAPI Integration

The background removal is now integrated into the main FastAPI server:

```bash
# Start server
python3 start.py

# Use the /remove-background endpoint
curl -X POST "http://localhost:8000/remove-background" \
     -F "file=@input.png"
```

## ⚙️ Configuration Options

### BackgroundRemover Parameters

- **`alpha_blur`** (bool): Apply Gaussian blur to alpha channel for soft edges
- **`erode_dilate`** (bool): Apply morphological operations to clean alpha channel
- **`color_decontamination`** (bool): Remove background color contamination
- **`post_process_strength`** (float): Strength of post-processing (0.0 to 1.0)

### CLI Options

- **`--no-blur`**: Disable alpha blur
- **`--no-erode`**: Disable morphological operations
- **`--no-decontamination`**: Disable color decontamination
- **`--strength`**: Post-processing strength (0.0 to 1.0)

## 🔧 Advanced Usage

### Custom Post-Processing

```python
# High-quality settings for screen printing
remover = BackgroundRemover(
    alpha_blur=True,           # Soft edges
    erode_dilate=True,         # Clean artifacts
    color_decontamination=True, # Remove color bleeding
    post_process_strength=0.7   # Strong processing
)
```

### Batch Processing

```python
# Process multiple folders
folders = ["designs/", "logos/", "artwork/"]
for folder in folders:
    remover.process_folder(folder, f"processed_{folder}")
```

### Error Handling

```python
try:
    result = remover.remove_background("input.png")
except Exception as e:
    print(f"Background removal failed: {e}")
    # Handle error appropriately
```

## 📊 Performance

- **Speed**: ~2-5 seconds per image (depending on size and complexity)
- **Memory**: ~500MB-1GB RAM usage during processing
- **Quality**: Professional-grade results suitable for screen printing

## 🐛 Troubleshooting

### Common Issues

1. **Import Error**: Ensure all dependencies are installed
   ```bash
   pip3 install rembg Pillow numpy opencv-python
   ```

2. **Memory Issues**: For large images, try reducing post-processing strength
   ```python
   remover = BackgroundRemover(post_process_strength=0.3)
   ```

3. **Quality Issues**: Adjust settings based on image type
   ```python
   # For text-heavy images
   remover = BackgroundRemover(alpha_blur=False, erode_dilate=True)
   
   # For smooth graphics
   remover = BackgroundRemover(alpha_blur=True, erode_dilate=False)
   ```

### Fallback Mode

If `rembg` fails, the system automatically falls back to the OpenCV method:

```python
# The fallback is automatic, but you can force it:
remover = BackgroundRemover(post_process_strength=0.0)
```

## 🔄 Integration with Existing Code

The new system is designed to be a drop-in replacement:

```python
# Old way (still works)
result = self.remove_background_grabcut(image_path)

# New way (better quality)
from background_remover import BackgroundRemover
remover = BackgroundRemover()
result = remover.remove_background(image_path)
```

## 📈 Future Improvements

- **GPU Acceleration**: CUDA support for faster processing
- **Batch Processing**: Parallel processing of multiple images
- **Custom Models**: Support for custom U2Net models
- **API Endpoints**: Additional processing options via FastAPI

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is part of the DTF Customizer system.
