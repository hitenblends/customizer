// DTF Design Tool - Main Application
(function() {
    'use strict';

    // Global state
    const state = {
        image: null,
        originalImage: null,
        ppi: 300,
        zoom: 1,
        pan: { x: 0, y: 0 },
        isDragging: false,
        lastMousePos: { x: 0, y: 0 },
        colorPalette: [],
        backgroundMask: null,
        excludeBackground: false,
        lockAspectRatio: true,
        targetWidth: 0,
        targetHeight: 0,
        isManualBgEdit: false,
        brushMode: 'erase',
        brushSize: 20
    };

    // DOM elements
    const elements = {
        fileInput: document.getElementById('dtf-file-input'),
        resetBtn: document.getElementById('dtf-reset-btn'),
        fileInfo: document.getElementById('dtf-file-info'),
        previewCanvas: document.getElementById('dtf-preview-canvas'),
        zoomIn: document.getElementById('dtf-zoom-in'),
        zoomOut: document.getElementById('dtf-zoom-out'),
        zoomFit: document.getElementById('dtf-zoom-fit'),
        colorCount: document.getElementById('dtf-color-count'),
        reanalyzeBtn: document.getElementById('dtf-reanalyze-btn'),
        colorPalette: document.getElementById('dtf-color-palette'),
        addColorBtn: document.getElementById('dtf-add-color-btn'),
        ppi: document.getElementById('dtf-ppi'),
        pixelDimensions: document.getElementById('dtf-pixel-dimensions'),
        printSize: document.getElementById('dtf-print-size'),
        sizeWarnings: document.getElementById('dtf-size-warnings'),
        excludeBg: document.getElementById('dtf-exclude-bg'),
        autoBgRemove: document.getElementById('dtf-auto-bg-remove'),
        manualBgEdit: document.getElementById('dtf-manual-bg-edit'),
        bgManualControls: document.getElementById('dtf-bg-manual-controls'),
        brushSize: document.getElementById('dtf-brush-size'),
        eraseBtn: document.getElementById('dtf-erase-btn'),
        restoreBtn: document.getElementById('dtf-restore-btn'),
        bgToggle: document.getElementById('dtf-bg-toggle'),
        downloadBgRemoved: document.getElementById('dtf-download-bg-removed'),
        lockAspect: document.getElementById('dtf-lock-aspect'),
        widthInches: document.getElementById('dtf-width-inches'),
        heightInches: document.getElementById('dtf-height-inches'),
        widthPx: document.getElementById('dtf-width-px'),
        heightPx: document.getElementById('dtf-height-px'),
        applyResize: document.getElementById('dtf-apply-resize'),
        downloadResized: document.getElementById('dtf-download-resized'),
        exportJob: document.getElementById('dtf-export-job'),
        // Color popup elements
        colorPopup: document.getElementById('dtf-color-popup'),
        colorPopupClose: document.getElementById('dtf-color-popup-close'),
        colorPopupCancel: document.getElementById('dtf-color-popup-cancel'),
        colorPopupSave: document.getElementById('dtf-color-popup-save'),
        colorPreview: document.getElementById('dtf-color-preview'),
        colorPicker: document.getElementById('dtf-color-picker'),
        colorHexInput: document.getElementById('dtf-color-hex-input'),
        colorRgbInput: document.getElementById('dtf-color-rgb-input'),
        colorHexError: document.getElementById('dtf-color-hex-error'),
        colorRgbError: document.getElementById('dtf-color-rgb-error'),
        detectedColors: document.getElementById('dtf-detected-colors'), // Added this element
        minPercentage: document.getElementById('dtf-min-percentage')
    };

    // Utility functions
    const utils = {
        seededRandom: (function() {
            let seed = 1;
            return function() {
                seed = (seed * 9301 + 49297) % 233280;
                return seed / 233280;
            };
        })(),

        rgbToHex: function(r, g, b) {
            return '#' + [r, g, b].map(x => {
                const hex = x.toString(16);
                return hex.length === 1 ? '0' + hex : hex;
            }).join('');
        },

        hexToRgb: function(hex) {
            const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
            return result ? {
                r: parseInt(result[1], 16),
                g: parseInt(result[2], 16),
                b: parseInt(result[3], 16)
            } : null;
        },

        formatFileSize: function(bytes) {
            if (bytes === 0) return '0 Bytes';
            const k = 1024;
            const sizes = ['Bytes', 'KB', 'MB', 'GB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
        },

        downloadCanvas: function(canvas, filename) {
            const link = document.createElement('a');
            link.download = filename;
            link.href = canvas.toDataURL('image/png');
            link.click();
        },

        // Color popup management
        showColorPopup: function(color, index) {
            state.editingColorIndex = index;
            state.editingColor = { ...color };
            
            // Update popup elements
            elements.colorPreview.style.backgroundColor = color.hex;
            elements.colorPicker.value = color.hex;
            elements.colorHexInput.value = color.hex;
            elements.colorRgbInput.value = `${color.rgb.r}, ${color.rgb.g}, ${color.rgb.b}`;
            
            // Clear errors
            elements.colorHexError.classList.remove('show');
            elements.colorRgbError.classList.remove('show');
            
            // Show popup
            elements.colorPopup.classList.add('show');
        },

        hideColorPopup: function() {
            elements.colorPopup.classList.remove('show');
            state.editingColorIndex = null;
            state.editingColor = null;
        },

        validateHex: function(hex) {
            return /^#[0-9A-F]{6}$/i.test(hex);
        },

        validateRgb: function(rgbStr) {
            const parts = rgbStr.split(',').map(s => s.trim());
            if (parts.length !== 3) return false;
            
            const r = parseInt(parts[0]);
            const g = parseInt(parts[1]);
            const b = parseInt(parts[2]);
            
            return !isNaN(r) && !isNaN(g) && !isNaN(b) && 
                   r >= 0 && r <= 255 && 
                   g >= 0 && g <= 255 && 
                   b >= 0 && b <= 255;
        },

        updateColorFromHex: function(hex) {
            if (this.validateHex(hex)) {
                const rgb = this.hexToRgb(hex);
                elements.colorRgbInput.value = `${rgb.r}, ${rgb.g}, ${rgb.b}`;
                elements.colorPreview.style.backgroundColor = hex;
                elements.colorPicker.value = hex;
                elements.colorHexError.classList.remove('show');
                return true;
            } else {
                elements.colorHexError.classList.add('show');
                return false;
            }
        },

        updateColorFromRgb: function() {
            const r = parseInt(elements.colorRInput.value) || 0;
            const g = parseInt(elements.colorGInput.value) || 0;
            const b = parseInt(elements.colorBInput.value) || 0;
            
            if (r >= 0 && r <= 255 && g >= 0 && g <= 255 && b >= 0 && b <= 255) {
                const hex = this.rgbToHex(r, g, b);
                elements.colorHexInput.value = hex;
                elements.colorPreview.style.backgroundColor = hex;
                elements.colorPicker.value = hex;
                elements.colorRgbError.classList.remove('show');
                return true;
            } else {
                elements.colorRgbError.classList.add('show');
                return false;
            }
        }
    };

    // Color extraction and palette management
    const colorManager = {
                        extractColors: async function(imageData, excludeBackground = false, mask = null) {
                    try {
                        // Convert canvas to blob
                        const canvas = document.createElement('canvas');
                        canvas.width = imageData.width;
                        canvas.height = imageData.height;
                        const ctx = canvas.getContext('2d');
                        ctx.putImageData(imageData, 0, 0);
                        
                        // Convert to blob
                        const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/png'));
                        
                        // Create form data
                        const formData = new FormData();
                        formData.append('image', blob, 'image.png');
                        formData.append('num_colors', parseInt(elements.colorCount.value) || 6);
                        formData.append('merge_threshold', 6);
                        formData.append('min_percentage', parseFloat(elements.minPercentage.value) || 1.0);
                        
                        // Send to backend
                        const response = await fetch('http://localhost:8000/extract-colors', {
                            method: 'POST',
                            body: formData
                        });
                        
                        if (!response.ok) {
                            throw new Error(`HTTP error! status: ${response.status}`);
                        }
                        
                        const result = await response.json();
                        
                        // Convert backend format to frontend format
                        const colors = result.colors.map((color, index) => ({
                            id: index + 1,
                            rgb: color.rgb,
                            hex: color.hex,
                            percentage: parseFloat(color.percentage)
                        }));
                        
                        return {
                            colors: colors,
                            totalColors: result.total_colors
                        };
                    } catch (error) {
                        console.error('Backend color extraction failed, falling back to frontend:', error);
                        
                        // Fallback to frontend extraction
                        return this.extractColorsFrontend(imageData, excludeBackground, mask);
                    }
                },
                
                // Frontend fallback color extraction
                extractColorsFrontend: function(imageData, excludeBackground = false, mask = null) {
                    const pixels = imageData.data;
                    const width = imageData.width;
                    const height = imageData.height;
                    const colorMap = new Map();
                    let totalPixels = 0;

                    for (let i = 0; i < pixels.length; i += 4) {
                        const x = (i / 4) % width;
                        const y = Math.floor((i / 4) / width);
                        
                        if (excludeBackground && mask && mask[y * width + x] === 0) {
                            continue;
                        }

                        const r = pixels[i];
                        const g = pixels[i + 1];
                        const b = pixels[i + 2];
                        const a = pixels[i + 3];

                        if (a === 0) continue;

                        const hex = utils.rgbToHex(r, g, b);
                        const count = colorMap.get(hex) || 0;
                        colorMap.set(hex, count + 1);
                        totalPixels++;
                    }

                    const colorArray = Array.from(colorMap.entries()).map(([hex, count]) => ({
                        hex,
                        rgb: utils.hexToRgb(hex),
                        count,
                        percentage: (count / totalPixels) * 100
                    })).sort((a, b) => b.count - a.count);

                    // Filter colors by minimum percentage threshold
                    const minPercentage = parseFloat(elements.minPercentage.value) || 1.0;
                    const significantColors = colorArray.filter(color => color.percentage >= minPercentage);
                    
                    const totalUniqueColors = significantColors.length;
                    const quantizedColors = this.quantizeColors(significantColors, parseInt(elements.colorCount.value));
                    
                    return {
                        colors: quantizedColors,
                        totalColors: totalUniqueColors
                    };
                },

        quantizeColors: function(colors, k) {
            if (colors.length <= k) return colors;

            const clusters = [];
            const centroids = [];

            for (let i = 0; i < k; i++) {
                centroids.push(colors[i].rgb);
            }

            for (let iteration = 0; iteration < 10; iteration++) {
                clusters.length = 0;
                for (let i = 0; i < k; i++) {
                    clusters.push([]);
                }

                for (const color of colors) {
                    let minDistance = Infinity;
                    let bestCluster = 0;

                    for (let i = 0; i < k; i++) {
                        const distance = Math.sqrt(
                            Math.pow(color.rgb.r - centroids[i].r, 2) +
                            Math.pow(color.rgb.g - centroids[i].g, 2) +
                            Math.pow(color.rgb.b - centroids[i].b, 2)
                        );
                        if (distance < minDistance) {
                            minDistance = distance;
                            bestCluster = i;
                        }
                    }
                    clusters[bestCluster].push(color);
                }

                for (let i = 0; i < k; i++) {
                    if (clusters[i].length > 0) {
                        const avgR = clusters[i].reduce((sum, c) => sum + c.rgb.r, 0) / clusters[i].length;
                        const avgG = clusters[i].reduce((sum, c) => sum + c.rgb.g, 0) / clusters[i].length;
                        const avgB = clusters[i].reduce((sum, c) => sum + c.rgb.b, 0) / clusters[i].length;
                        centroids[i] = { r: Math.round(avgR), g: Math.round(avgG), b: Math.round(avgB) };
                    }
                }
            }

            const mergedColors = [];
            const threshold = 6;

            for (const cluster of clusters) {
                if (cluster.length === 0) continue;

                const dominantColor = cluster.reduce((prev, current) => 
                    (prev.count > current.count) ? prev : current
                );

                let shouldMerge = false;
                for (const mergedColor of mergedColors) {
                    const distance = Math.sqrt(
                        Math.pow(dominantColor.rgb.r - mergedColor.rgb.r, 2) +
                        Math.pow(dominantColor.rgb.g - mergedColor.rgb.g, 2) +
                        Math.pow(dominantColor.rgb.b - mergedColor.rgb.b, 2)
                    );
                    if (distance < threshold) {
                        shouldMerge = true;
                        break;
                    }
                }

                if (!shouldMerge) {
                    mergedColors.push(dominantColor);
                }
            }

            return mergedColors.slice(0, k);
        },

        renderPalette: function(colors) {
            elements.colorPalette.innerHTML = '';
            
            colors.forEach((color, index) => {
                const colorItem = document.createElement('div');
                colorItem.className = 'dtf-color-item';
                colorItem.innerHTML = `
                    <div class="dtf-color-swatch" style="background-color: ${color.hex}" data-index="${index}"></div>
                    <div class="dtf-color-info">
                        <div class="dtf-color-hex">${color.hex}</div>
                        <div class="dtf-color-percentage">${color.percentage.toFixed(1)}%</div>
                    </div>
                    <div class="dtf-color-controls-btns">
                        <button class="dtf-color-lock" data-index="${index}">🔒</button>
                        <button class="dtf-btn dtf-btn-secondary dtf-remove-color" data-index="${index}">×</button>
                    </div>
                `;
                elements.colorPalette.appendChild(colorItem);
            });

            this.addPaletteEventListeners();
        },

        addPaletteEventListeners: function() {
            document.querySelectorAll('.dtf-color-swatch').forEach(swatch => {
                swatch.addEventListener('click', function() {
                    const index = parseInt(this.dataset.index);
                    const color = state.colorPalette[index];
                    utils.showColorPopup(color, index);
                });
            });

            document.querySelectorAll('.dtf-color-lock').forEach(lock => {
                lock.addEventListener('click', function() {
                    const index = parseInt(this.dataset.index);
                    const color = state.colorPalette[index];
                    color.locked = !color.locked;
                    this.textContent = color.locked ? '🔒' : '🔓';
                    this.classList.toggle('locked', color.locked);
                });
            });

            document.querySelectorAll('.dtf-remove-color').forEach(btn => {
                btn.addEventListener('click', function() {
                    const index = parseInt(this.dataset.index);
                    state.colorPalette.splice(index, 1);
                    colorManager.renderPalette(state.colorPalette);
                });
            });
        }
    };

    // Image processing and preview management
    const imageManager = {
        loadImage: function(file) {
            return new Promise((resolve, reject) => {
                const reader = new FileReader();
                reader.onload = function(e) {
                    const img = new Image();
                    img.onload = function() {
                        resolve(img);
                    };
                    img.onerror = reject;
                    img.src = e.target.result;
                };
                reader.onerror = reject;
                reader.readAsDataURL(file);
            });
        },

        createCanvas: function(width, height) {
            const canvas = document.createElement('canvas');
            canvas.width = width;
            canvas.height = height;
            return canvas;
        },

        drawImage: function(canvas, image, x = 0, y = 0, width = null, height = null) {
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            if (width && height) {
                ctx.drawImage(image, x, y, width, height);
            } else {
                ctx.drawImage(image, x, y);
            }
        },

        getImageData: function(canvas) {
            return canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height);
        },

        resizeImage: function(canvas, newWidth, newHeight, quality = 'high') {
            const newCanvas = this.createCanvas(newWidth, newHeight);
            const ctx = newCanvas.getContext('2d');
            
            ctx.imageSmoothingEnabled = true;
            ctx.imageSmoothingQuality = quality === 'high' ? 'high' : 'medium';
            
            this.drawImage(newCanvas, canvas, 0, 0, newWidth, newHeight);
            return newCanvas;
        },

        updatePreview: function() {
            if (!state.image) return;

            const previewCtx = elements.previewCanvas.getContext('2d');
            const previewWidth = elements.previewCanvas.width;
            const previewHeight = elements.previewCanvas.height;

            previewCtx.clearRect(0, 0, previewWidth, previewHeight);

            previewCtx.save();
            previewCtx.translate(state.pan.x, state.pan.y);
            previewCtx.scale(state.zoom, state.zoom);

            const scale = Math.min(
                previewWidth / state.image.width,
                previewHeight / state.image.height
            ) * state.zoom;

            const scaledWidth = state.image.width * scale;
            const scaledHeight = state.image.height * scale;
            const x = (previewWidth - scaledWidth) / 2;
            const y = (previewHeight - scaledHeight) / 2;

            previewCtx.drawImage(state.image, x, y, scaledWidth, scaledHeight);
            previewCtx.restore();
        }
    };

    // Background removal functionality
    const backgroundManager = {
        createMask: function(imageData) {
            const width = imageData.width;
            const height = imageData.height;
            const mask = new Uint8Array(width * height);
            
            const cornerColors = this.getCornerColors(imageData);
            
            for (let y = 0; y < height; y++) {
                for (let x = 0; x < width; x++) {
                    const index = y * width + x;
                    const pixelIndex = (y * width + x) * 4;
                    
                    const r = imageData.data[pixelIndex];
                    const g = imageData.data[pixelIndex + 1];
                    const b = imageData.data[pixelIndex + 2];
                    const a = imageData.data[pixelIndex + 3];
                    
                    let isBackground = false;
                    for (const cornerColor of cornerColors) {
                        const distance = Math.sqrt(
                            Math.pow(r - cornerColor.r, 2) +
                            Math.pow(g - cornerColor.g, 2) +
                            Math.pow(b - cornerColor.b, 2)
                        );
                        if (distance < 30) {
                            isBackground = true;
                            break;
                        }
                    }
                    
                    mask[index] = isBackground ? 0 : 1;
                }
            }
            
            return mask;
        },

        getCornerColors: function(imageData) {
            const width = imageData.width;
            const height = imageData.height;
            const corners = [
                { x: 0, y: 0 },
                { x: width - 1, y: 0 },
                { x: 0, y: height - 1 },
                { x: width - 1, y: height - 1 }
            ];
            
            const colors = [];
            for (const corner of corners) {
                const index = (corner.y * width + corner.x) * 4;
                colors.push({
                    r: imageData.data[index],
                    g: imageData.data[index + 1],
                    b: imageData.data[index + 2]
                });
            }
            
            return colors;
        },

        applyMask: function(canvas, mask) {
            const ctx = canvas.getContext('2d');
            const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
            const pixels = imageData.data;
            const width = canvas.width;
            
            for (let i = 0; i < pixels.length; i += 4) {
                const x = (i / 4) % width;
                const y = Math.floor((i / 4) / width);
                const maskIndex = y * width + x;
                
                if (mask[maskIndex] === 0) {
                    pixels[i + 3] = 0;
                }
            }
            
            ctx.putImageData(imageData, 0, 0);
        }
    };

    // Event handlers
    const eventHandlers = {
        fileInput: function(e) {
            const file = e.target.files[0];
            if (!file) return;

            const validTypes = ['image/png', 'image/jpeg', 'image/webp', 'image/svg+xml'];
            if (!validTypes.includes(file.type)) {
                alert('Please select a valid image file (PNG, JPEG, WebP, or SVG)');
                return;
            }

            if (file.size > 30 * 1024 * 1024) {
                alert('File size is too large. Please select a file smaller than 30MB.');
                return;
            }

            this.loadImage(file);
        },

        loadImage: async function(file) {
            try {
                const img = await imageManager.loadImage(file);
                
                state.originalImage = img;
                state.image = img;
                
                const workCanvas = imageManager.createCanvas(img.width, img.height);
                imageManager.drawImage(workCanvas, img);
                
                this.updateFileInfo(file, img);
                this.updateSizeInfo(img);
                this.setupPreview(img);
                
                const imageData = imageManager.getImageData(workCanvas);
                const colorResult = await colorManager.extractColors(imageData, state.excludeBackground, state.backgroundMask);
                state.colorPalette = colorResult.colors;
                
                // Update the detected colors count
                elements.detectedColors.textContent = colorResult.totalColors;
                
                colorManager.renderPalette(state.colorPalette);
                
                elements.resetBtn.style.display = 'inline-block';
                this.updateSizeInputs();
                
            } catch (error) {
                console.error('Error loading image:', error);
                alert('Error loading image. Please try again.');
            }
        },

        updateFileInfo: function(file, img) {
            const info = `
                <strong>${file.name}</strong> (${utils.formatFileSize(file.size)})<br>
                Dimensions: ${img.width} × ${img.height} pixels<br>
                Type: ${file.type}
            `;
            elements.fileInfo.innerHTML = info;
            elements.fileInfo.style.display = 'block';
        },

        updateSizeInfo: function(img) {
            elements.pixelDimensions.textContent = `${img.width} × ${img.height} px`;
            this.updatePrintSize();
        },

        updatePrintSize: function() {
            if (!state.image) return;
            
            const widthInches = state.image.width / state.ppi;
            const heightInches = state.image.height / state.ppi;
            const widthMm = widthInches * 25.4;
            const heightMm = heightInches * 25.4;
            
            elements.printSize.textContent = 
                `${widthInches.toFixed(2)}" × ${heightInches.toFixed(2)}" ` +
                `(${widthMm.toFixed(1)} × ${heightMm.toFixed(1)} mm)`;
            
            this.updateSizeWarnings();
        },

        updateSizeWarnings: function() {
            if (!state.image) return;
            
            const warnings = [];
            const widthInches = state.image.width / state.ppi;
            const heightInches = state.image.height / state.ppi;
            
            if (state.targetWidth && widthInches < state.targetWidth) {
                warnings.push('Width may be low resolution at selected size/PPI');
            }
            if (state.targetHeight && heightInches < state.targetHeight) {
                warnings.push('Height may be low resolution at selected size/PPI');
            }
            
            if (warnings.length > 0) {
                elements.sizeWarnings.innerHTML = warnings.map(w => `<div>⚠️ ${w}</div>`).join('');
                elements.sizeWarnings.style.display = 'block';
            } else {
                elements.sizeWarnings.style.display = 'none';
            }
        },

        setupPreview: function(img) {
            const previewCanvas = elements.previewCanvas;
            const container = previewCanvas.parentElement;
            
            previewCanvas.width = container.clientWidth - 40;
            previewCanvas.height = 400;
            
            state.zoom = 1;
            state.pan = { x: 0, y: 0 };
            
            imageManager.updatePreview();
            this.setupPreviewEvents();
        },

        setupPreviewEvents: function() {
            const canvas = elements.previewCanvas;
            
            canvas.addEventListener('mousedown', this.onMouseDown.bind(this));
            canvas.addEventListener('mousemove', this.onMouseMove.bind(this));
            canvas.addEventListener('mouseup', this.onMouseUp.bind(this));
            canvas.addEventListener('wheel', this.onWheel.bind(this));
            
            canvas.addEventListener('touchstart', this.onTouchStart.bind(this));
            canvas.addEventListener('touchmove', this.onTouchMove.bind(this));
            canvas.addEventListener('touchend', this.onTouchEnd.bind(this));
        },

        onMouseDown: function(e) {
            if (state.isManualBgEdit) return;
            
            state.isDragging = true;
            state.lastMousePos = { x: e.clientX, y: e.clientY };
            canvas.style.cursor = 'grabbing';
        },

        onMouseMove: function(e) {
            if (!state.isDragging || state.isManualBgEdit) return;
            
            const deltaX = e.clientX - state.lastMousePos.x;
            const deltaY = e.clientY - state.lastMousePos.y;
            
            state.pan.x += deltaX;
            state.pan.y += deltaY;
            
            state.lastMousePos = { x: e.clientX, y: e.clientY };
            imageManager.updatePreview();
        },

        onMouseUp: function() {
            state.isDragging = false;
            canvas.style.cursor = 'grab';
        },

        onWheel: function(e) {
            e.preventDefault();
            
            const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
            state.zoom = Math.max(0.1, Math.min(5, state.zoom * zoomFactor));
            
            imageManager.updatePreview();
        },

        onTouchStart: function(e) {
            if (e.touches.length === 2) {
                const touch1 = e.touches[0];
                const touch2 = e.touches[1];
                const distance = Math.sqrt(
                    Math.pow(touch2.clientX - touch1.clientX, 2) +
                    Math.pow(touch2.clientY - touch1.clientY, 2)
                );
                state.initialTouchDistance = distance;
            }
        },

        onTouchMove: function(e) {
            if (e.touches.length === 2 && state.initialTouchDistance) {
                const touch1 = e.touches[0];
                const touch2 = e.touches[1];
                const distance = Math.sqrt(
                    Math.pow(touch2.clientX - touch1.clientX, 2) +
                    Math.pow(touch2.clientY - touch1.clientY, 2)
                );
                
                const zoomFactor = distance / state.initialTouchDistance;
                state.zoom = Math.max(0.1, Math.min(5, state.zoom * zoomFactor));
                imageManager.updatePreview();
            }
        },

        onTouchEnd: function() {
            state.initialTouchDistance = null;
        },

        updateSizeInputs: function() {
            if (!state.image) return;
            
            const widthInches = state.image.width / state.ppi;
            const heightInches = state.image.height / state.ppi;
            
            elements.widthInches.value = widthInches.toFixed(2);
            elements.heightInches.value = heightInches.toFixed(2);
            
            this.updatePixelEquivalents();
        },

        updatePixelEquivalents: function() {
            const widthInches = parseFloat(elements.widthInches.value) || 0;
            const heightInches = parseFloat(elements.heightInches.value) || 0;
            
            const widthPx = Math.round(widthInches * state.ppi);
            const heightPx = Math.round(heightInches * state.ppi);
            
            elements.widthPx.textContent = `(${widthPx} px)`;
            elements.heightPx.textContent = `(${heightPx} px)`;
        }
    };

    // Initialize the application
    function init() {
        elements.fileInput.addEventListener('change', eventHandlers.fileInput.bind(eventHandlers));
        elements.resetBtn.addEventListener('click', reset);
        elements.zoomIn.addEventListener('click', () => {
            state.zoom = Math.min(5, state.zoom * 1.2);
            imageManager.updatePreview();
        });
        elements.zoomOut.addEventListener('click', () => {
            state.zoom = Math.max(0.1, state.zoom / 1.2);
            imageManager.updatePreview();
        });
        elements.zoomFit.addEventListener('click', () => {
            state.zoom = 1;
            state.pan = { x: 0, y: 0 };
            imageManager.updatePreview();
        });

        // Color count change
        elements.colorCount.addEventListener('change', reanalyzeColors);

        // Minimum percentage change
        elements.minPercentage.addEventListener('change', reanalyzeColors);
        elements.reanalyzeBtn.addEventListener('click', reanalyzeColors);
        elements.addColorBtn.addEventListener('click', addCustomColor);

        elements.ppi.addEventListener('change', function() {
            state.ppi = parseInt(this.value);
            eventHandlers.updatePrintSize();
            eventHandlers.updateSizeInputs();
        });

        elements.excludeBg.addEventListener('change', function() {
            state.excludeBackground = this.checked;
            if (state.image) {
                reanalyzeColors();
            }
        });
        elements.removeBg.addEventListener('click', async () => await removeBackground());


        elements.lockAspect.addEventListener('change', function() {
            state.lockAspectRatio = this.checked;
        });
        elements.widthInches.addEventListener('input', updateSizeFromInput);
        elements.heightInches.addEventListener('input', updateSizeFromInput);

        elements.applyResize.addEventListener('click', applyResize);
        elements.downloadResized.addEventListener('click', downloadResized);
        elements.exportJob.addEventListener('click', exportJob);

        setupAccordion();
        setupColorPopup();
    }

    // Utility functions
    function reset() {
        state.image = null;
        state.originalImage = null;
        state.colorPalette = [];
        state.backgroundMask = null;
        state.zoom = 1;
        state.pan = { x: 0, y: 0 };
        
        elements.fileInput.value = '';
        elements.fileInfo.style.display = 'none';
        elements.resetBtn.style.display = 'none';
        elements.colorPalette.innerHTML = '';
        elements.previewCanvas.getContext('2d').clearRect(0, 0, elements.previewCanvas.width, elements.previewCanvas.height);
        
        elements.pixelDimensions.textContent = '-';
        elements.printSize.textContent = '-';
        elements.sizeWarnings.style.display = 'none';
        elements.widthInches.value = '';
        elements.heightInches.value = '';
        elements.widthPx.textContent = '-';
        elements.heightPx.textContent = '-';
    }

    async function reanalyzeColors() {
        if (!state.image) return;
        
        const workCanvas = imageManager.createCanvas(state.image.width, state.image.height);
        imageManager.drawImage(workCanvas, state.image);
        
        const imageData = imageManager.getImageData(workCanvas);
        const colorResult = await colorManager.extractColors(imageData, state.excludeBackground, state.backgroundMask);
        state.colorPalette = colorResult.colors;
        
        // Update the detected colors count
        elements.detectedColors.textContent = colorResult.totalColors;
        
        colorManager.renderPalette(state.colorPalette);
    }

    function addCustomColor() {
        const newColor = {
            hex: '#FF0000',
            rgb: { r: 255, g: 0, b: 0 },
            count: 0,
            percentage: 0,
            custom: true
        };
        state.colorPalette.push(newColor);
        colorManager.renderPalette(state.colorPalette);
        
        // Open popup to edit the new color
        const index = state.colorPalette.length - 1;
        utils.showColorPopup(newColor, index);
    }

    async function removeBackground() {
        if (!state.image) return;
        
        const workCanvas = imageManager.createCanvas(state.image.width, state.image.height);
        imageManager.drawImage(workCanvas, state.image);
        
        const imageData = imageManager.getImageData(workCanvas);
        state.backgroundMask = backgroundManager.createMask(imageData);
        
        const previewCanvas = imageManager.createCanvas(state.image.width, state.image.height);
        imageManager.drawImage(previewCanvas, state.image);
        backgroundManager.applyMask(previewCanvas, state.backgroundMask);
        
        state.image = previewCanvas;
        imageManager.updatePreview();
        
        // Automatically update colors after background removal
        await reanalyzeColors();
        
        // Show success message
        alert('Background removed successfully! Colors have been updated.');
    }



    function updateSizeFromInput() {
        if (!state.lockAspectRatio) {
            eventHandlers.updatePixelEquivalents();
            return;
        }
        
        const widthInches = parseFloat(elements.widthInches.value) || 0;
        const heightInches = parseFloat(elements.heightInches.value) || 0;
        
        if (widthInches > 0 && state.image) {
            const aspectRatio = state.image.height / state.image.width;
            elements.heightInches.value = (widthInches * aspectRatio).toFixed(2);
        } else if (heightInches > 0 && state.image) {
            const aspectRatio = state.image.width / state.image.height;
            elements.widthInches.value = (heightInches * aspectRatio).toFixed(2);
        }
        
        eventHandlers.updatePixelEquivalents();
    }

    function applyResize() {
        if (!state.image) return;
        
        const widthInches = parseFloat(elements.widthInches.value);
        const heightInches = parseFloat(elements.heightInches.value);
        
        if (!widthInches || !heightInches) {
            alert('Please enter valid dimensions');
            return;
        }
        
        const newWidth = Math.round(widthInches * state.ppi);
        const newHeight = Math.round(heightInches * state.ppi);
        
        if (newWidth > state.originalImage.width * 1.5 || newHeight > state.originalImage.height * 1.5) {
            if (!confirm('Upscaling beyond 150% may result in poor quality. Continue?')) {
                return;
            }
        }
        
        const resizedCanvas = imageManager.resizeImage(state.image, newWidth, newHeight, 'high');
        state.image = resizedCanvas;
        
        imageManager.updatePreview();
        eventHandlers.updateSizeInfo(resizedCanvas);
        eventHandlers.updateSizeInputs();
        
        alert('Image resized successfully!');
    }

    function downloadResized() {
        if (!state.image) {
            alert('Please load an image first');
            return;
        }
        
        const filename = `resized_${Date.now()}.png`;
        utils.downloadCanvas(state.image, filename);
    }

    function exportJob() {
        if (!state.image) {
            alert('Please load an image first');
            return;
        }
        
        const jobData = {
            filename: elements.fileInfo.textContent.split('(')[0].trim(),
            timestamp: new Date().toISOString(),
            image: {
                width: state.image.width,
                height: state.image.height,
                ppi: state.ppi
            },
            printSize: {
                width: parseFloat(elements.widthInches.value) || 0,
                height: parseFloat(elements.heightInches.value) || 0
            },
            colors: state.colorPalette.map(color => ({
                hex: color.hex,
                rgb: color.rgb,
                percentage: color.percentage,
                locked: color.locked || false
            })),
            backgroundRemoved: !!state.backgroundMask,
            settings: {
                excludeBackground: state.excludeBackground,
                lockAspectRatio: state.lockAspectRatio
            }
        };
        
        const blob = new Blob([JSON.stringify(jobData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `dtf_job_${Date.now()}.json`;
        link.click();
        URL.revokeObjectURL(url);
    }

    function setupAccordion() {
        document.querySelectorAll('.dtf-block-header').forEach(header => {
            header.addEventListener('click', function() {
                const target = this.dataset.target;
                const content = document.getElementById(`dtf-${target}-content`);
                const toggle = this.querySelector('.dtf-toggle');
                
                if (content.classList.contains('collapsed')) {
                    content.classList.remove('collapsed');
                    this.classList.remove('collapsed');
                    toggle.textContent = '▼';
                } else {
                    content.classList.add('collapsed');
                    this.classList.add('collapsed');
                    toggle.textContent = '▶';
                }
            });
        });
    }

    // Color popup functionality
    function setupColorPopup() {
        // Close popup events
        elements.colorPopupClose.addEventListener('click', utils.hideColorPopup);
        elements.colorPopupCancel.addEventListener('click', utils.hideColorPopup);
        
        // Close popup when clicking outside
        elements.colorPopup.addEventListener('click', function(e) {
            if (e.target === elements.colorPopup) {
                utils.hideColorPopup();
            }
        });
        
        // Color picker change
        elements.colorPicker.addEventListener('change', function() {
            const hex = this.value;
            utils.updateColorFromHex(hex);
        });
        
        // HEX input change
        elements.colorHexInput.addEventListener('input', function() {
            utils.updateColorFromHex(this.value);
        });
        
        // RGB inputs change
        elements.colorRInput.addEventListener('input', function() {
            utils.updateColorFromRgb();
        });
        elements.colorGInput.addEventListener('input', function() {
            utils.updateColorFromRgb();
        });
        elements.colorBInput.addEventListener('input', function() {
            utils.updateColorFromRgb();
        });
        
        // Save button
        elements.colorPopupSave.addEventListener('click', function() {
            if (state.editingColorIndex !== null && state.editingColor) {
                const index = state.editingColorIndex;
                const color = state.colorPalette[index];
                
                // Update color with new values
                color.hex = elements.colorHexInput.value.toUpperCase();
                color.rgb = utils.hexToRgb(color.hex);
                
                // Re-render palette
                colorManager.renderPalette(state.colorPalette);
                
                // Hide popup
                utils.hideColorPopup();
            }
        });
        
        // Keyboard shortcuts
        document.addEventListener('keydown', function(e) {
            if (elements.colorPopup.classList.contains('show')) {
                if (e.key === 'Escape') {
                    utils.hideColorPopup();
                } else if (e.key === 'Enter' && e.ctrlKey) {
                    elements.colorPopupSave.click();
                }
            }
        });
    }

    // Start the application
    init();
})();
