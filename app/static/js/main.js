/**
 * AI Waste Classification System - Frontend JavaScript
 * Handles user interactions, API calls, and dynamic content updates
 */

// Initialize on page load
document.addEventListener('DOMContentLoaded', function () {
    initializeApp();
});

function initializeApp() {
    updateDateTime();
    loadWasteCategories();
    loadStatistics();

    // Update time every second
    setInterval(updateDateTime, 1000);

    // Setup image upload handler
    document.getElementById('imageUpload').addEventListener('change', handleImageUpload);
}

// Camera state
let isCameraOn = true;

function toggleCamera() {
    const btn = document.getElementById('cameraToggleBtn');
    const videoFeed = document.getElementById('videoFeed');
    const statusBadge = document.querySelector('.status-badge');

    if (isCameraOn) {
        // Turn OFF
        fetch('/cleanup')
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    isCameraOn = false;
                    // Update UI
                    btn.innerHTML = '<i class="fas fa-video"></i> Start Camera';
                    btn.classList.remove('btn-danger');
                    btn.classList.add('btn-success');

                    // Stop video feed by removing source
                    videoFeed.src = '';
                    videoFeed.style.backgroundColor = '#000';

                    // Update badge
                    statusBadge.classList.remove('live');
                    statusBadge.innerHTML = '<i class="fas fa-video-slash"></i> OFFLINE';
                    statusBadge.style.background = 'rgba(255, 50, 50, 0.2)';
                    statusBadge.style.color = '#ff5050';

                    showToast('Camera stopped', 'info');
                }
            })
            .catch(err => {
                console.error('Error stopping camera:', err);
                showToast('Error stopping camera', 'error');
            });
    } else {
        // Turn ON
        // Trigger page reload or re-assign src to restart stream
        // Simple way: re-assign src. The backend will re-init camera on new request to /video_feed
        isCameraOn = true;

        // Update UI
        btn.innerHTML = '<i class="fas fa-video-slash"></i> Stop Camera';
        btn.classList.remove('btn-success');
        btn.classList.add('btn-danger');

        // Restart video feed
        videoFeed.src = "/video_feed?" + new Date().getTime(); // Prevent caching
        videoFeed.style.backgroundColor = 'transparent';

        // Update badge
        statusBadge.classList.add('live');
        statusBadge.innerHTML = '<span class="pulse-dot"></span> LIVE';
        statusBadge.style.background = ''; // Reset to CSS default
        statusBadge.style.color = '';

        showToast('Camera started', 'success');
    }
}

// ============================================
// Date & Time Functions
// ============================================

function updateDateTime() {
    const now = new Date();

    // Update time
    const timeString = now.toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        hour12: false
    });
    document.getElementById('currentTime').textContent = timeString;

    // Update date
    const dateString = now.toLocaleDateString('en-US', {
        weekday: 'long',
        year: 'numeric',
        month: 'long',
        day: 'numeric'
    });
    document.getElementById('currentDate').textContent = dateString;
}

// ============================================
// Statistics Functions
// ============================================

async function loadStatistics() {
    try {
        const response = await fetch('/waste_stats');
        const stats = await response.json();

        document.getElementById('totalCategories').textContent = stats.total_categories;
        document.getElementById('biodegradableCount').textContent = stats.biodegradable_count;
        document.getElementById('recyclableCount').textContent = stats.recyclable_count;
        document.getElementById('organicCount').textContent = stats.organic_count;

        // Animate numbers
        animateValue('totalCategories', 0, stats.total_categories, 1000);
        animateValue('biodegradableCount', 0, stats.biodegradable_count, 1000);
        animateValue('recyclableCount', 0, stats.recyclable_count, 1000);
        animateValue('organicCount', 0, stats.organic_count, 1000);
    } catch (error) {
        console.error('Error loading statistics:', error);
    }
}

function animateValue(id, start, end, duration) {
    const element = document.getElementById(id);
    const range = end - start;
    const increment = range / (duration / 16);
    let current = start;

    const timer = setInterval(() => {
        current += increment;
        if (current >= end) {
            element.textContent = end;
            clearInterval(timer);
        } else {
            element.textContent = Math.floor(current);
        }
    }, 16);
}

// ============================================
// Waste Categories Functions
// ============================================

async function loadWasteCategories() {
    try {
        const response = await fetch('/waste_info');
        const wasteTypes = await response.json();

        const container = document.getElementById('wasteCategories');
        container.innerHTML = '';

        for (const [type, info] of Object.entries(wasteTypes)) {
            const card = createCategoryCard(type, info);
            container.appendChild(card);
        }
    } catch (error) {
        console.error('Error loading waste categories:', error);
        showToast('Failed to load waste categories', 'error');
    }
}

function createCategoryCard(type, info) {
    const card = document.createElement('div');
    card.className = 'category-card';

    card.innerHTML = `
        <div class="category-header">
            <div class="category-color" style="background-color: rgb(${info.color_rgb})"></div>
            <div class="category-name">${type}</div>
        </div>
        <div class="category-info">
            <p><strong>Type:</strong> ${info.type}</p>
            <p><strong>Category:</strong> ${info.category}</p>
            <p><strong>Biodegradable:</strong> ${info.biodegradable ? 'Yes ✓' : 'No ✗'}</p>
            <p><strong>Recyclable:</strong> ${info.recyclable === true ? 'Yes ✓' : info.recyclable === false ? 'No ✗' : info.recyclable}</p>
            <p><strong>Decomposition:</strong> ${info.decomposition_time}</p>
            <p style="margin-top: 10px;"><em>${info.description}</em></p>
        </div>
    `;

    return card;
}

// ============================================
// Image Capture & Upload Functions
// ============================================

async function captureImage() {
    try {
        showToast('Capturing image...', 'info');

        const response = await fetch('/capture');
        const data = await response.json();

        if (data.success) {
            // Inject filename into prediction object for use in displayResults
            if (data.filename) {
                data.prediction.filename = data.filename;
            }
            displayResults(data.prediction);
            showToast('Image captured and analyzed successfully!', 'success');
        } else {
            showToast('Failed to capture image: ' + data.error, 'error');
        }
    } catch (error) {
        console.error('Error capturing image:', error);
        showToast('Error capturing image: ' + error.message, 'error');
    }
}

function uploadImage() {
    document.getElementById('imageUpload').click();
}

async function handleImageUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    // Validate file type
    if (!file.type.startsWith('image/')) {
        showToast('Please select a valid image file', 'error');
        return;
    }

    // Validate file size (max 16MB)
    if (file.size > 16 * 1024 * 1024) {
        showToast('Image size must be less than 16MB', 'error');
        return;
    }

    const formData = new FormData();
    formData.append('image', file);

    try {
        showToast('Analyzing image...', 'info');

        const response = await fetch('/detect', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            displayResults(data.prediction);

            // Show the uploaded/annotated image in separate container
            if (data.image) {
                const container = document.getElementById('uploadedImageContainer');
                const img = document.getElementById('uploadedPredictionImage');

                img.src = data.image;
                container.style.display = 'block';

                // Scroll to view the image
                container.scrollIntoView({ behavior: 'smooth' });
            }

            showToast('Image analyzed successfully!', 'success');
        } else {
            showToast('Analysis failed: ' + data.error, 'error');
        }
    } catch (error) {
        console.error('Error analyzing image:', error);
        showToast('Error analyzing image: ' + error.message, 'error');
    }

    // Reset input
    event.target.value = '';
}

// ============================================
// Results Display Functions
// ============================================

function displayResults(prediction) {
    const resultsDiv = document.getElementById('results');

    if (!prediction) {
        resultsDiv.innerHTML = `
            <div class="waiting-state">
                <p>⚠️ No waste detected</p>
                <small>Try a different image</small>
            </div>
        `;
        return;
    }

    const confidenceClass = prediction.confidence > 70 ? 'confidence-high' :
        prediction.confidence > 40 ? 'confidence-medium' : 'confidence-low';

    // Global variable to store current filename for training
    window.currentCaptureFilename = prediction.filename || null;
    // Note: implementation of capture route in app.py needs to return filename in prediction or separately
    // Based on previous code, /capture returns {success, filename, prediction}
    // So prediction object passed here might not have filename. 

    resultsDiv.innerHTML = `
        <div class="result-card">
            <div class="result-header">
                <div class="waste-color-indicator" style="background-color: rgb(${prediction.color_rgb})"></div>
                <div class="result-title">
                    <div class="waste-type">${prediction.class}</div>
                    <span class="confidence-badge ${confidenceClass}">
                        ${prediction.confidence.toFixed(1)}% Confidence
                    </span>
                </div>
            </div>
            
            <div class="result-details">
                <div class="detail-row">
                    <span class="detail-label">Type:</span>
                    <span class="detail-value">${prediction.type}</span>
                </div>
                
                <div class="detail-row">
                    <span class="detail-label">Category:</span>
                    <span class="detail-value">${prediction.category}</span>
                </div>
                
                <div class="detail-row">
                    <span class="detail-label">Biodegradable:</span>
                    <span class="detail-value">
                        ${prediction.biodegradable ?
            '<span class="badge badge-success">Yes ✓</span>' :
            '<span class="badge badge-danger">No ✗</span>'}
                    </span>
                </div>
                
                <div class="detail-row">
                    <span class="detail-label">Recyclable:</span>
                    <span class="detail-value">
                        ${prediction.recyclable === true ?
            '<span class="badge badge-success">Yes ✓</span>' :
            prediction.recyclable === false ?
                '<span class="badge badge-danger">No ✗</span>' :
                prediction.recyclable}
                    </span>
                </div>
                
                <!-- Training Contribution Section -->
                <div class="training-section" style="margin-top: 20px; padding-top: 15px; border-top: 1px solid rgba(255,255,255,0.1);">
                    <h4 style="margin-bottom: 10px; font-size: 0.9em; color: rgba(255,255,255,0.7);">
                        <i class="fas fa-robot"></i> Improve AI Accuracy
                    </h4>
                    <div style="display: flex; gap: 10px; align-items: center;">
                        <select id="trainingLabel" class="form-select" style="background: rgba(0,0,0,0.2); border: 1px solid rgba(255,255,255,0.2); color: white; padding: 5px; border-radius: 4px;">
                            <option value="cardboard" ${prediction.class === 'cardboard' ? 'selected' : ''}>Cardboard</option>
                            <option value="glass" ${prediction.class === 'glass' ? 'selected' : ''}>Glass</option>
                            <option value="metal" ${prediction.class === 'metal' ? 'selected' : ''}>Metal</option>
                            <option value="paper" ${prediction.class === 'paper' ? 'selected' : ''}>Paper</option>
                            <option value="plastic" ${prediction.class === 'plastic' ? 'selected' : ''}>Plastic</option>
                            <option value="trash" ${prediction.class === 'trash' ? 'selected' : ''}>Trash</option>
                        </select>
                        <button onclick="saveForTraining()" class="btn btn-sm btn-outline-primary" style="padding: 5px 10px; font-size: 0.8em;">
                            Save to Dataset
                        </button>
                    </div>
                </div>

                <div class="detail-row">
                    <span class="detail-label">Decomposition Time:</span>
                    <span class="detail-value">${prediction.decomposition_time}</span>
                </div>
                
                <div class="detail-row">
                    <span class="detail-label">Description:</span>
                    <span class="detail-value">${prediction.description}</span>
                </div>
                
                <div class="detail-row">
                    <span class="detail-label">Disposal Method:</span>
                    <span class="detail-value">${prediction.disposal}</span>
                </div>
                
                <div class="detail-row">
                    <span class="detail-label">Environmental Impact:</span>
                    <span class="detail-value">${prediction.environmental_impact}</span>
                </div>
                
                <div class="detail-row">
                    <span class="detail-label">Recycling Process:</span>
                    <span class="detail-value">${prediction.recycling_process}</span>
                </div>
                
                <div class="detail-row">
                    <span class="detail-label">Tips:</span>
                    <span class="detail-value">${prediction.tips}</span>
                </div>
            </div>
        </div>
    `;
}

async function saveForTraining() {
    if (!window.currentCaptureFilename) {
        showToast('No captured image to save. Please capture first.', 'error');
        return;
    }

    const label = document.getElementById('trainingLabel').value;

    try {
        const response = await fetch('/save_for_training', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                filename: window.currentCaptureFilename,
                label: label
            })
        });

        const data = await response.json();

        if (data.success) {
            showToast(`Saved! Total ${label} images: ${data.count}`, 'success');
        } else {
            showToast('Failed to save: ' + data.error, 'error');
        }
    } catch (error) {
        console.error('Error saving training data:', error);
        showToast('Error saving data', 'error');
    }
}

async function startBurstTrain() {
    const label = document.getElementById('burstLabel').value;

    if (!confirm(`Point camera at ${label.toUpperCase()} and keep steady.\nThis will capture 10 images for training.\n\nReady?`)) {
        return;
    }

    try {
        showToast(`Starting burst capture for ${label}...`, 'info');

        const response = await fetch('/train_burst', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ label: label })
        });

        const data = await response.json();

        if (data.success) {
            showToast(data.message, 'success');
        } else {
            showToast('Training failed: ' + data.error, 'error');
        }
    } catch (error) {
        console.error('Error in burst training:', error);
        showToast('Error: ' + error.message, 'error');
    }
}


// ============================================
// Toast Notification Functions
// ============================================

function showToast(message, type = 'info') {
    // Remove existing toasts
    const existingToasts = document.querySelectorAll('.toast');
    existingToasts.forEach(toast => toast.remove());

    // Create new toast
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;

    const icon = type === 'success' ? '✓' : type === 'error' ? '✗' : 'ℹ';

    toast.innerHTML = `
        <div style="display: flex; align-items: center; gap: 10px;">
            <span style="font-size: 1.2rem;">${icon}</span>
            <span>${message}</span>
        </div>
    `;

    document.body.appendChild(toast);

    // Auto remove after 4 seconds
    setTimeout(() => {
        toast.style.animation = 'slideInRight 0.3s ease reverse';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

// ============================================
// Cleanup on page unload
// ============================================

window.addEventListener('beforeunload', function () {
    fetch('/cleanup').catch(err => console.error('Cleanup error:', err));
});

function closeUploadedImage() {
    const container = document.getElementById('uploadedImageContainer');
    container.style.display = 'none';
    document.getElementById('uploadedPredictionImage').src = '';
}
