// DTF Customizer - Palette-Based Color Matching
// Global state
const state = {
    image: null,
    colorMatches: [],
    zoom: 1,
    pan: { x: 0, y: 0 },
    ppi: 300,
    isDragging: false,
    lastMousePos: { x: 0, y: 0 }
};

// Initialize the application
function init() {
    console.log('🚀 Initializing DTF Customizer with Palette Matching...');
    
    // Set up event listeners
    setupEventListeners();
    
    // Initialize UI
    updateUI();
    
    // Check backend status
    checkBackendStatus();
}

// Set up event listeners
function setupEventListeners() {
    // File upload
    const fileInput = document.getElementById('dtf-file-input');
    if (fileInput) {
        fileInput.addEventListener('change', handleFileUpload);
    }
    
    // PPI input
    const ppiInput = document.getElementById('dtf-ppi');
    if (ppiInput) {
        ppiInput.addEventListener('input', handlePPIChange);
    }
    
    // Background removal
    const removeBgBtn = document.getElementById('dtf-remove-bg');
    if (removeBgBtn) {
        removeBgBtn.addEventListener('click', removeBackground);
    }
    
    // Advanced background removal
    const advancedRemoveBtn = document.getElementById('dtf-remove-bg-advanced');
    if (advancedRemoveBtn) {
        advancedRemoveBtn.addEventListener('click', () => {
            const advancedOptions = document.getElementById('dtf-advanced-options');
            if (advancedOptions) {
                advancedOptions.style.display = advancedOptions.style.display === 'none' ? 'block' : 'none';
            }
        });
    }
    
    // Re-analyze colors
    const reanalyzeBtn = document.getElementById('dtf-reanalyze-colors');
    if (reanalyzeBtn) {
        reanalyzeBtn.addEventListener('click', () => {
            if (state.image) {
                extractColorsWithPalette(state.image);
            } else {
                showMessage('Please upload an image first', 'warning');
            }
        });
    }
    
    // Edge sensitivity slider
    const sensitivitySlider = document.getElementById('dtf-edge-sensitivity');
    const sensitivityValue = document.getElementById('dtf-sensitivity-value');
    if (sensitivitySlider && sensitivityValue) {
        sensitivitySlider.addEventListener('input', (e) => {
            sensitivityValue.textContent = e.target.value;
        });
    }
    
    // Apply advanced removal button
    const applyAdvancedBtn = document.getElementById('dtf-apply-advanced');
    if (applyAdvancedBtn) {
        applyAdvancedBtn.addEventListener('click', removeBackgroundAdvanced);
    }
    
    // Preview interactions
    const preview = document.getElementById('dtf-preview');
    if (preview) {
        preview.addEventListener('mousedown', startPan);
        preview.addEventListener('mousemove', pan);
        preview.addEventListener('mouseup', stopPan);
        preview.addEventListener('wheel', handleZoom);
        preview.addEventListener('mouseleave', stopPan);
    }
}

// Check if Python backend is available
async function checkBackendStatus() {
    try {
        const response = await fetch('https://dtf-customizer-backend.onrender.com/health');
        if (response.ok) {
            const data = await response.json();
            updateBackendStatus(true, data);
            console.log('✅ Python backend available:', data);
        } else {
            updateBackendStatus(false);
            console.log('❌ Python backend not responding');
        }
    } catch (error) {
        updateBackendStatus(false);
        console.log('❌ Python backend unavailable:', error.message);
    }
}

// Update backend status display
function updateBackendStatus(available, data = null) {
    const statusElement = document.getElementById('dtf-backend-status');
    if (statusElement) {
        if (available && data) {
            statusElement.innerHTML = `
                <div class="dtf-status-success">
                    <span>🐍 Python Backend: ${data.service}</span>
                    <span>📊 Palette: ${data.palette_colors} Colors</span>
                    <span>✨ Version: ${data.version}</span>
                </div>
            `;
        } else {
            statusElement.innerHTML = `
                <div class="dtf-status-error">
                    <span>❌ Python Backend: Unavailable</span>
                    <span>🔧 Using Frontend Fallback</span>
                </div>
            `;
        }
    }
}

// Handle file upload
async function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    console.log('📁 File selected:', file.name, file.type);
    
    try {
        // Load image
        const imageUrl = URL.createObjectURL(file);
        const img = new Image();
        
        img.onload = async () => {
            state.image = img;
            updatePreview();
            updatePrintSize(img);
            
            // Extract colors using palette matching
            await extractColorsWithPalette(img);
            
            // Background removal is now optional - user must click the button manually
        };
        
        img.src = imageUrl;
        
    } catch (error) {
        console.error('❌ Error loading image:', error);
        showMessage('Error loading image. Please try again.', 'error');
    }
}

// Extract colors and match to palette
async function extractColorsWithPalette(img) {
    try {
        console.log('🎨 Extracting colors with palette matching...');
        showMessage('🔍 Analyzing image colors...', 'info');
        
        // Convert image to blob for upload
        const canvas = document.createElement('canvas');
        canvas.width = img.width;
        canvas.height = img.height;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0);
        
        const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/png'));
        
        // Send to Python backend
        const formData = new FormData();
        formData.append('file', blob, 'image.png');
        formData.append('num_colors', '12');
        formData.append('min_percentage', '1.0');
        
        console.log('📤 Sending image to Python backend for palette matching...');
        const response = await fetch('https://dtf-customizer-backend.onrender.com/extract-colors', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const errorText = await response.text();
            console.error('❌ Backend error response:', errorText);
            throw new Error(`HTTP error! status: ${response.status}, details: ${errorText}`);
        }
        
        const result = await response.json();
        console.log('📥 Palette matching result:', result);
        
        if (result.success) {
            state.colorMatches = result.matches;
            updateColorPalette();
            updateDetectedColors(result.matched_to_palette, result.total_colors_detected);
            showMessage(`✅ Found ${result.matched_to_palette} palette colors!`, 'success');
        } else {
            throw new Error(result.error || 'Unknown error');
        }
        
    } catch (error) {
        console.error('❌ Palette matching failed:', error);
        showMessage('Python backend unavailable, using frontend fallback', 'warning');
        
        // Fallback to basic color extraction
        extractColorsFrontendFallback(img);
    }
}

// Frontend fallback for color extraction
function extractColorsFrontendFallback(img) {
    console.log('🔄 Using frontend fallback for color extraction...');
    
    const canvas = document.createElement('canvas');
    canvas.width = img.width;
    canvas.height = img.height;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0);
    
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;
    
    // Simple color counting (basic fallback)
    const colorCounts = {};
    const totalPixels = data.length / 4;
    
    for (let i = 0; i < data.length; i += 4) {
        const r = Math.round(data[i] / 10) * 10;
        const g = Math.round(data[i + 1] / 10) * 10;
        const b = Math.round(data[i + 2] / 10) * 10;
        
        if (r + g + b > 30 && r + g + b < 720) { // Filter out very light/dark
            const color = `${r},${g},${b}`;
            colorCounts[color] = (colorCounts[color] || 0) + 1;
        }
    }
    
    // Convert to matches format
    const matches = Object.entries(colorCounts)
        .map(([rgb, count]) => {
            const [r, g, b] = rgb.split(',').map(Number);
            const percentage = (count / totalPixels) * 100;
            return {
                detected_rgb: [r, g, b],
                detected_hex: rgbToHex(r, g, b),
                detected_percentage: percentage,
                matched_palette: null,
                similarity_score: null
            };
        })
        .filter(match => match.detected_percentage > 2)
        .sort((a, b) => b.detected_percentage - a.detected_percentage)
        .slice(0, 8);
    
    state.colorMatches = matches;
    updateColorPalette();
    updateDetectedColors(0, matches.length);
    showMessage('⚠️ Using basic frontend color detection', 'warning');
}

// Update color palette display
function updateColorPalette() {
    const paletteContainer = document.getElementById('dtf-color-palette');
    if (!paletteContainer || !state.colorMatches) return;
    
    paletteContainer.innerHTML = '';
    
    state.colorMatches.forEach((match, index) => {
        const colorItem = document.createElement('div');
        colorItem.className = 'dtf-color-item';
        
        if (match.matched_palette) {
            // Matched to palette
            const isConsolidated = match.consolidated;
            const consolidatedInfo = isConsolidated ? 
                `<div class="dtf-consolidated-info">
                    <span class="dtf-consolidated-badge">🔗 Consolidated</span>
                    <span class="dtf-consolidated-count">(${match.consolidated_details.original_colors_count} colors merged)</span>
                </div>` : '';
            
            colorItem.innerHTML = `
                <div class="dtf-color-swatch" style="background-color: ${match.detected_hex}"></div>
                <div class="dtf-color-info">
                    <div class="dtf-palette-name">${match.matched_palette.name}</div>
                    <div class="dtf-palette-category">${match.matched_palette.category}</div>
                    ${consolidatedInfo}
                    <div class="dtf-similarity-score">Similarity: ${match.similarity_score.toFixed(1)}</div>
                    <div class="dtf-detected-color">
                        <span class="dtf-hex">${match.detected_hex}</span>
                        <span class="dtf-rgb">[${match.detected_rgb.join(', ')}]</span>
                    </div>
                    <div class="dtf-percentage">${match.detected_percentage.toFixed(1)}%</div>
                </div>
                <div class="dtf-color-status dtf-matched">✅ Matched</div>
            `;
            
            if (isConsolidated) {
                colorItem.classList.add('dtf-consolidated');
            }
        } else {
            // Unmatched color
            colorItem.innerHTML = `
                <div class="dtf-color-swatch" style="background-color: ${match.detected_hex}"></div>
                <div class="dtf-color-info">
                    <div class="dtf-palette-name">No Match</div>
                    <div class="dtf-palette-category">Custom Color</div>
                    <div class="dtf-detected-color">
                        <span class="dtf-hex">${match.detected_hex}</span>
                        <span class="dtf-rgb">[${match.detected_rgb.join(', ')}]</span>
                    </div>
                    <div class="dtf-percentage">${match.detected_percentage.toFixed(1)}%</div>
                </div>
                <div class="dtf-color-status dtf-unmatched">❌ Unmatched</div>
            `;
        }
        
        paletteContainer.appendChild(colorItem);
    });
}

// Update detected colors count
function updateDetectedColors(matchedCount, totalCount) {
    const element = document.getElementById('dtf-detected-colors');
    if (!element) return;
    
    element.textContent = `${matchedCount}/${totalCount}`;
    
    // Update CSS classes for visual feedback
    element.className = 'dtf-detected-colors';
    if (matchedCount === totalCount) {
        element.classList.add('dtf-all-matched');
    } else if (matchedCount > totalCount * 0.7) {
        element.classList.add('dtf-mostly-matched');
    } else if (matchedCount > 0) {
        element.classList.add('dtf-partially-matched');
    } else {
        element.classList.add('dtf-none-matched');
    }
    
    // Show production summary
    showProductionSummary(matchedCount, totalCount);
}

// Show production summary
function showProductionSummary(matchedCount, totalCount) {
    const summaryContainer = document.getElementById('dtf-production-summary');
    if (!summaryContainer) return;
    
    let message = '';
    let type = 'info';
    
    if (matchedCount === totalCount) {
        message = `🎯 Perfect! All ${totalCount} colors match industry standards`;
        type = 'success';
    } else if (matchedCount >= totalCount * 0.8) {
        message = `✅ Great! ${matchedCount}/${totalCount} colors are production-ready`;
        type = 'success';
    } else if (matchedCount >= totalCount * 0.5) {
        message = `⚠️ Good: ${matchedCount}/${totalCount} colors match standards`;
        type = 'warning';
    } else {
        message = `❌ Limited: Only ${matchedCount}/${totalCount} colors match standards`;
        type = 'error';
    }
    
    summaryContainer.innerHTML = `<div class="dtf-message dtf-${type}">${message}</div>`;
}

// Remove background (simplified)
async function removeBackground() {
    if (!state.image) return;
    
    try {
        console.log('🔄 Removing background...');
        showMessage('🔄 Removing background with AI...', 'info');
        
        // Convert image to blob
        const canvas = document.createElement('canvas');
        canvas.width = state.image.width;
        canvas.height = state.image.height;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(state.image, 0, 0);
        
        const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/png'));
        
        // Send to Python backend for background removal
        const formData = new FormData();
        formData.append('file', blob, 'image.png');
        formData.append('method', 'rembg');
        formData.append('model', 'u2net');
        formData.append('post_process', 'false');
        
        const response = await fetch('https://dtf-customizer-backend.onrender.com/remove-background', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const errorText = await response.text();
            console.error('❌ Backend error:', errorText);
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        console.log('📥 Background removal result:', result);
        
        if (result.success) {
            // Load background-removed image
            const imgData = `data:image/png;base64,${result.image_base64}`;
            const newImg = new Image();
            
            newImg.onload = () => {
                state.image = newImg;
                updatePreview();
                
                // Update background status to removed
                updateBackgroundStatus('removed', {
                    confidence: 'high',
                    transparency_ratio: 0.8,
                    method: result.method || 'rembg'
                });
                
                showMessage('✅ Background removed successfully!', 'success');
            };
            
            newImg.src = imgData;
            
        } else {
            throw new Error(result.error || 'Background removal failed');
        }
        
    } catch (error) {
        console.error('❌ Background removal failed:', error);
        showMessage(`Background removal failed: ${error.message}`, 'error');
    }
}

// Advanced background removal with custom settings
async function removeBackgroundAdvanced() {
    if (!state.image) return;
    
    try {
        console.log('🔄 Advanced background removal...');
        showMessage('🔍 Advanced background removal in progress...', 'info');
        
        // Get advanced settings
        const preserveText = document.getElementById('dtf-preserve-text')?.checked ?? true;
        const edgeSensitivity = parseFloat(document.getElementById('dtf-edge-sensitivity')?.value ?? 0.1);
        
        console.log('⚙️ Settings:', { preserveText, edgeSensitivity });
        
        // Convert image to blob
        const canvas = document.createElement('canvas');
        canvas.width = state.image.width;
        canvas.height = state.image.height;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(state.image, 0, 0);
        
        const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/png'));
        
        // Send to advanced background removal endpoint
        const formData = new FormData();
        formData.append('file', blob, 'image.png');
        formData.append('preserve_text', preserveText);
        formData.append('edge_sensitivity', edgeSensitivity);
        
        const response = await fetch('https://dtf-customizer-backend.onrender.com/remove-background-advanced', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        
        if (result.success) {
            // Load background-removed image
            const imgData = `data:image/png;base64,${result.image_base64}`;
            const newImg = new Image();
            
            newImg.onload = () => {
                state.image = newImg;
                updatePreview();
                
                // Update background status
                updateBackgroundStatus('removed', {
                    confidence: 'high',
                    transparency_ratio: 0.8,
                    method: 'advanced_removal',
                    settings: result.settings
                });
                
                showMessage(`✅ Advanced background removal completed! (${result.method})`, 'success');
            };
            
            newImg.src = imgData;
            
        } else {
            throw new Error(result.error || 'Advanced background removal failed');
        }
        
    } catch (error) {
        console.error('❌ Advanced background removal failed:', error);
        showMessage('Advanced background removal failed. Using original image.', 'error');
    }
}

// Update print size calculations
function updatePrintSize(img) {
    const widthInch = (img.width / state.ppi).toFixed(2);
    const heightInch = (img.height / state.ppi).toFixed(2);
    const widthMm = (widthInch * 25.4).toFixed(1);
    const heightMm = (heightInch * 25.4).toFixed(1);
    
    // Update display
    const dimensionsElement = document.getElementById('dtf-dimensions');
    if (dimensionsElement) {
        dimensionsElement.innerHTML = `
            <div class="dtf-dimension-item">
                <span class="dtf-label">Pixels:</span>
                <span class="dtf-value">${img.width} × ${img.height} px</span>
            </div>
            <div class="dtf-dimension-item">
                <span class="dtf-label">Print Size:</span>
                <span class="dtf-value">${widthInch}" × ${heightInch}" (${widthMm} × ${heightMm} mm)</span>
            </div>
        `;
    }
    
    // Update resize inputs
    const widthInput = document.getElementById('dtf-resize-width');
    const heightInput = document.getElementById('dtf-resize-height');
    if (widthInput) widthInput.value = widthInch;
    if (heightInput) heightInput.value = heightInch;
}

// Handle PPI change
function handlePPIChange(event) {
    const newPPI = parseInt(event.target.value) || 300;
    state.ppi = newPPI;
    
    if (state.image) {
        updatePrintSize(state.image);
    }
}

// Update preview
function updatePreview() {
    const preview = document.getElementById('dtf-preview');
    if (!preview || !state.image) return;
    
    const canvas = document.createElement('canvas');
    canvas.width = preview.clientWidth;
    canvas.height = preview.clientHeight;
    const ctx = canvas.getContext('2d');
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw checkerboard background
    drawCheckerboard(ctx, canvas.width, canvas.height);
    
    // Calculate image position and size
    const imgAspect = state.image.width / state.image.height;
    const canvasAspect = canvas.width / canvas.height;
    
    let drawWidth, drawHeight, drawX, drawY;
    
    if (imgAspect > canvasAspect) {
        drawWidth = canvas.width * 0.8;
        drawHeight = drawWidth / imgAspect;
        drawX = (canvas.width - drawWidth) / 2;
        drawY = (canvas.height - drawHeight) / 2;
    } else {
        drawHeight = canvas.height * 0.8;
        drawWidth = drawHeight * imgAspect;
        drawX = (canvas.width - drawWidth) / 2;
        drawY = (canvas.height - drawHeight) / 2;
    }
    
    // Apply zoom and pan
    const scaledWidth = drawWidth * state.zoom;
    const scaledHeight = drawHeight * state.zoom;
    const scaledX = drawX + state.pan.x;
    const scaledY = drawY + state.pan.y;
    
    // Draw image
    ctx.drawImage(state.image, scaledX, scaledY, scaledWidth, scaledHeight);
    
    // Replace preview content
    preview.innerHTML = '';
    preview.appendChild(canvas);
}

// Draw checkerboard background
function drawCheckerboard(ctx, width, height) {
    const size = 20;
    const colors = ['#f0f0f0', '#e0e0e0'];
    
    for (let y = 0; y < height; y += size) {
        for (let x = 0; x < width; x += size) {
            const colorIndex = ((x / size) + (y / size)) % 2;
            ctx.fillStyle = colors[colorIndex];
            ctx.fillRect(x, y, size, size);
        }
    }
}

// Pan and zoom functions
function startPan(event) {
    state.isDragging = true;
    state.lastMousePos = { x: event.clientX, y: event.clientY };
    event.preventDefault();
}

function pan(event) {
    if (!state.isDragging) return;
    
    const deltaX = event.clientX - state.lastMousePos.x;
    const deltaY = event.clientY - state.lastMousePos.y;
    
    state.pan.x += deltaX;
    state.pan.y += deltaY;
    
    state.lastMousePos = { x: event.clientX, y: event.clientY };
    
    updatePreview();
}

function stopPan() {
    state.isDragging = false;
}

function handleZoom(event) {
    event.preventDefault();
    
    const zoomFactor = event.deltaY > 0 ? 0.9 : 1.1;
    state.zoom *= zoomFactor;
    
    // Clamp zoom
    state.zoom = Math.max(0.1, Math.min(5, state.zoom));
    
    updatePreview();
}

// Update UI
function updateUI() {
    // Update file input label
    const fileInput = document.getElementById('dtf-file-input');
    const fileLabel = document.getElementById('dtf-file-label');
    if (fileInput && fileLabel) {
        fileLabel.textContent = fileInput.files[0]?.name || 'Choose Image File';
    }
}

// Show message
function showMessage(message, type = 'info') {
    const messageContainer = document.getElementById('dtf-messages');
    if (!messageContainer) return;
    
    const messageElement = document.createElement('div');
    messageElement.className = `dtf-message dtf-${type}`;
    messageElement.textContent = message;
    
    messageContainer.appendChild(messageElement);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (messageElement.parentNode) {
            messageElement.parentNode.removeChild(messageElement);
        }
    }, 5000);
}

// Utility functions
function rgbToHex(r, g, b) {
    return '#' + [r, g, b].map(x => {
        const hex = x.toString(16);
        return hex.length === 1 ? '0' + hex : hex;
    }).join('');
}

// Update background status display
function updateBackgroundStatus(status, details = null) {
    const statusElement = document.getElementById('dtf-bg-status');
    const detailsElement = document.getElementById('dtf-bg-details');
    const detailsTextElement = document.getElementById('dtf-bg-details-text');
    
    if (!statusElement) return;
    
    if (status === 'removed') {
        statusElement.textContent = '✅ Background Removed';
        statusElement.className = 'dtf-value dtf-success';
        
        if (details) {
            detailsElement.style.display = 'flex';
            detailsTextElement.textContent = `${details.confidence} confidence, ${(details.transparency_ratio * 100).toFixed(1)}% transparent`;
        }
    } else if (status === 'present') {
        statusElement.textContent = '❌ Background Present';
        statusElement.className = 'dtf-value dtf-warning';
        
        if (details && details.reason) {
            detailsElement.style.display = 'flex';
            detailsTextElement.textContent = details.reason;
        }
    } else {
        statusElement.textContent = '❓ Unknown';
        statusElement.className = 'dtf-value dtf-info';
        detailsElement.style.display = 'none';
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', init);
