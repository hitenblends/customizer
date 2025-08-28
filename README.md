# DTF Color Extractor Backend

Advanced color extraction backend using Python, FastAPI, OpenCV, and scikit-learn for professional DTF and screen printing color analysis.

## 🚀 Features

- **Advanced Color Extraction**: Uses OpenCV and scikit-learn for professional-grade color analysis
- **LAB Color Space**: Better color similarity detection using perceptual color space
- **Smart Filtering**: Automatically filters out noise and insignificant colors
- **Color Merging**: Intelligently merges visually similar colors
- **Percentage Thresholds**: Configurable minimum percentage filters
- **Multiple Formats**: Supports PNG, JPG, JPEG, WebP
- **RESTful API**: Clean HTTP endpoints for easy integration

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup
1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the server**:
   ```bash
   python start.py
   ```

3. **Access the API**:
   - **Server**: http://localhost:8000
   - **API Docs**: http://localhost:8000/docs
   - **Health Check**: http://localhost:8000/health

## 📡 API Endpoints

### POST `/extract-colors`
Extract dominant colors from an uploaded image.

**Parameters**:
- `file`: Image file (PNG, JPG, JPEG, WebP)
- `num_colors`: Maximum number of colors (1-20, default: 12)
- `min_percentage`: Minimum percentage threshold (0.1-10.0, default: 1.0)
- `merge_threshold`: Color similarity threshold (10-100, default: 30.0)

**Response**:
```json
{
  "success": true,
  "colors": [
    {
      "rgb": [255, 0, 0],
      "hex": "#ff0000",
      "percentage": 25.5,
      "name": "Red"
    }
  ],
  "total_colors": 1,
  "total_pixels_analyzed": 1000000,
  "algorithm": "Advanced K-means with LAB similarity merging",
  "settings": {
    "requested_colors": 12,
    "min_percentage": 1.0,
    "merge_threshold": 30.0
  }
}
```

## 🔧 Configuration

### Environment Variables
- `PORT`: Server port (default: 8000)
- `HOST`: Server host (default: 0.0.0.0)

### Algorithm Parameters
- **num_colors**: How many colors to extract (1-20)
- **min_percentage**: Minimum percentage for a color to be included
- **merge_threshold**: LAB color space similarity threshold for merging

## 🎨 How It Works

1. **Image Loading**: Uses OpenCV for fast image processing
2. **Pixel Analysis**: Converts image to RGB pixel array
3. **Noise Filtering**: Removes very light/dark pixels that might be noise
4. **K-means Clustering**: Advanced clustering with multiple initializations
5. **LAB Color Space**: Converts to LAB for perceptual color similarity
6. **Smart Merging**: Merges visually similar colors using LAB distance
7. **Percentage Calculation**: Accurate pixel count percentages
8. **Result Filtering**: Applies minimum percentage thresholds

## 🚀 Deployment

### Local Development
```bash
python start.py
```

### Production (Gunicorn)
```bash
pip install gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Docker
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["python", "start.py"]
```

## 🔗 Frontend Integration

Update your frontend to use the backend API:

```javascript
async function extractColorsBackend(imageFile) {
    const formData = new FormData();
    formData.append('file', imageFile);
    formData.append('num_colors', 12);
    formData.append('min_percentage', 1.0);
    formData.append('merge_threshold', 30.0);
    
    try {
        const response = await fetch('http://localhost:8000/extract-colors', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.success) {
            // Update UI with backend results
            updateColorPalette(result.colors);
            updateColorCount(result.total_colors);
        }
        
    } catch (error) {
        console.error('Backend processing failed:', error);
        // Fallback to frontend method
    }
}
```

## 📊 Performance

- **Processing Speed**: 2-5 seconds for 1MP images
- **Memory Usage**: Efficient pixel processing with numpy
- **Accuracy**: Professional-grade color analysis
- **Scalability**: Handles multiple concurrent requests

## 🐛 Troubleshooting

### Common Issues
1. **OpenCV not found**: Install with `pip install opencv-python`
2. **Memory errors**: Reduce image size or increase system memory
3. **Slow processing**: Check image dimensions and reduce if necessary

### Debug Mode
Enable debug logging by setting log level:
```python
uvicorn.run(app, log_level="debug")
```

## 📝 License

This project is part of the DTF Customizer tool suite.

## 🤝 Contributing

Feel free to submit issues and enhancement requests!
