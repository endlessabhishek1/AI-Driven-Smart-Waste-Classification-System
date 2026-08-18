let stream = null;
const video = document.getElementById('videoFeed');
const canvas = document.getElementById('captureCanvas');
const toggleBtn = document.getElementById('cameraToggleBtn');
const resultsDiv = document.getElementById('results');

// Live Clock
function updateTime() {
    const now = new Date();
    const timeEl = document.getElementById('currentTime');
    const dateEl = document.getElementById('currentDate');
    if (timeEl) timeEl.textContent = now.toLocaleTimeString();
    if (dateEl) dateEl.textContent = now.toLocaleDateString(undefined, { weekday: 'short', year: 'numeric', month: 'short', day: 'numeric' });
}
setInterval(updateTime, 1000);
updateTime();

// Camera Start / Stop
async function startCamera() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ 
            video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "environment" } 
        });
        video.srcObject = stream;
        video.play();
        if (toggleBtn) {
            toggleBtn.innerHTML = '<i class="fas fa-video-slash"></i> Stop Camera';
            toggleBtn.className = 'btn btn-danger';
        }
    } catch (err) {
        console.error("Camera access error:", err);
        alert("Camera access denied or not available. Please allow camera permissions.");
    }
}

function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        video.srcObject = null;
        stream = null;
        if (toggleBtn) {
            toggleBtn.innerHTML = '<i class="fas fa-video"></i> Start Camera';
            toggleBtn.className = 'btn btn-success';
        }
    }
}

function toggleCamera() {
    if (stream && stream.active) {
        stopCamera();
    } else {
        startCamera();
    }
}

window.addEventListener('DOMContentLoaded', startCamera);

// Capture Frame & Analyze
async function captureImage() {
    if (!video.videoWidth) {
        alert("Camera is not active. Please start the camera first.");
        return;
    }

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(async (blob) => {
        const formData = new FormData();
        formData.append('file', blob, 'capture.jpg');
        await sendPredictionRequest(formData);
    }, 'image/jpeg');
}

// Upload Image
function uploadImage() {
    const fileInput = document.getElementById('imageUpload');
    fileInput.click();
    fileInput.onchange = async () => {
        if (fileInput.files.length > 0) {
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            
            const reader = new FileReader();
            reader.onload = (e) => {
                const imgContainer = document.getElementById('uploadedImageContainer');
                const imgEl = document.getElementById('uploadedPredictionImage');
                if (imgContainer && imgEl) {
                    imgEl.src = e.target.result;
                    imgContainer.style.display = 'block';
                }
            };
            reader.readAsDataURL(fileInput.files[0]);

            await sendPredictionRequest(formData);
        }
    };
}

function closeUploadedImage() {
    const imgContainer = document.getElementById('uploadedImageContainer');
    if (imgContainer) imgContainer.style.display = 'none';
}

// API Call to Flask Server
async function sendPredictionRequest(formData) {
    resultsDiv.innerHTML = `
        <div class="waiting-state">
            <div class="spinner"></div>
            <p>Analyzing Waste Image...</p>
        </div>`;

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();

        if (response.ok) {
            displayResults(data);
        } else {
            resultsDiv.innerHTML = `<p style="color:red;">Error: ${data.error || 'Failed to analyze'}</p>`;
        }
    } catch (error) {
        console.error("Prediction error:", error);
        resultsDiv.innerHTML = `<p style="color:red;">Server connection failed.</p>`;
    }
}

function displayResults(data) {
    resultsDiv.innerHTML = `
        <div style="padding: 15px; text-align: left;">
            <h3 style="color: #4facfe; margin-bottom: 8px;">Class: ${data.class || data.prediction}</h3>
            <p><strong>Confidence:</strong> ${(data.confidence * 100).toFixed(2)}%</p>
            <p><strong>Type:</strong> ${data.waste_type || 'General'}</p>
            <p style="margin-top: 10px; font-size: 0.9em; opacity: 0.85;">${data.guidelines || 'Properly segregate in designated bin.'}</p>
        </div>`;
}
