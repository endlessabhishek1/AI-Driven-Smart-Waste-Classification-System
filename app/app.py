"""
Flask application for AI-powered waste classification system.
Provides real-time waste detection and comprehensive waste information.
"""

import cv2
import numpy as np
import os
from flask import Flask, render_template, Response, jsonify, request
import base64
from datetime import datetime
from app.ai_model import WasteClassifier

app = Flask(__name__)
app.config['SECRET_KEY'] = 'waste-classification-secret-key'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize AI classifier
print("Initializing AI Waste Classifier...")
classifier = WasteClassifier(model_path='models/waste_classifier.pth')

from app.camera import VideoCamera

# Global camera object
video_camera = None

def get_video_camera():
    """Get or create global video camera instance."""
    global video_camera
    if video_camera is None:
        video_camera = VideoCamera(classifier)
    return video_camera

def generate_frames():
    """Stream frames from video camera."""
    cam = get_video_camera()
    while cam.is_running:
        frame = cam.get_frame()
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            # Small sleep 
            import time
            time.sleep(0.01)

@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detect', methods=['POST'])
def detect_waste():
    """Detect waste from uploaded image."""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read image
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Invalid image format'}), 400
        
        # Copy frame for detection (so original 'frame' stays clean)
        detection_frame = frame.copy()
        
        # Detect waste without using or modifying prediction history
        prediction = classifier.predict(detection_frame, use_smoothing=False)
        
        # Encode result image (Return RAW image without annotations as per user request)
        _, buffer = cv2.imencode('.jpg', frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'image': f'data:image/jpeg;base64,{img_base64}'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/waste_info')
def waste_info():
    """Get comprehensive information about all waste types."""
    return jsonify(classifier.waste_info)

@app.route('/waste_stats')
def waste_stats():
    """Get waste classification statistics."""
    stats = classifier.get_waste_statistics()
    return jsonify(stats)

@app.route('/capture', methods=['GET'])
def capture():
    """Capture current frame from webcam and analyze."""
    try:
        cam = get_video_camera()
        
        # Get latest prediction directly if available, or force a predict
        prediction = cam.get_prediction()
        if prediction is None:
             # If no prediction yet (should be rare if streaming), could wait or return error
             return jsonify({'success': False, 'error': 'Camera initializing...'}), 503

        # Capture snapshot using the camera method
        filename = cam.capture_snapshot()
        
        if filename:
             return jsonify({
                'success': True,
                'filename': filename,
                'prediction': prediction
            })
        else:
             return jsonify({'success': False, 'error': 'Failed to capture frame'}), 500
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/save_for_training', methods=['POST'])
def save_for_training():
    """Save a captured image to the dataset for training."""
    try:
        data = request.json
        if not data or 'filename' not in data or 'label' not in data:
            return jsonify({'success': False, 'error': 'Missing filename or label'}), 400
        
        filename = data['filename']
        label = data['label'].lower()
        
        # Validate label
        valid_labels = classifier.class_names
        if label not in valid_labels:
            return jsonify({'success': False, 'error': f'Invalid label. Must be one of {valid_labels}'}), 400
        
        # Source path (relative to app root)
        src_path = filename
        if not os.path.exists(src_path):
             return jsonify({'success': False, 'error': 'Source file not found'}), 404
             
        # Target directory
        target_dir = os.path.join('datasets', 'waste', label)
        os.makedirs(target_dir, exist_ok=True)
        
        # Target filename
        basename = os.path.basename(src_path)
        target_path = os.path.join(target_dir, basename)
        
        # Copy file (keep original in captures for now/display)
        import shutil
        shutil.copy2(src_path, target_path)
        
        # Also save to "trashnet" structure if that's what we are using, 
        # but for now let's stick to the 'datasets/waste' which train_model.py uses.
        
        return jsonify({
            'success': True, 
            'message': f'Saved to {label} training data',
            'count': len(os.listdir(target_dir))
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/train_burst', methods=['POST'])
def train_burst():
    """Capture burst of images and save directly to training dataset."""
    try:
        data = request.json
        label = data.get('label', '').lower()
        
        # Validate label
        valid_labels = classifier.class_names
        if label not in valid_labels:
            return jsonify({'success': False, 'error': f'Invalid label. Must be one of {valid_labels}'}), 400
            
        cam = get_video_camera()
        
        # Capture burst (returns list of temp filenames)
        temp_files = cam.capture_burst(count=10, delay=0.2)
        
        # Move files to dataset directory
        target_dir = os.path.join('datasets', 'waste', label)
        os.makedirs(target_dir, exist_ok=True)
        
        saved_count = 0
        import shutil
        
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                basename = os.path.basename(temp_file)
                # Prefix with 'burst' to identify manual training data
                new_name = f"burst_{basename}"
                target_path = os.path.join(target_dir, new_name)
                shutil.move(temp_file, target_path)
                saved_count += 1
                
        return jsonify({
            'success': True,
            'message': f'Successfully captured {saved_count} images for "{label}"',
            'count': len(os.listdir(target_dir))
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/cleanup')
def cleanup():
    """Clean up camera resources."""
    global video_camera
    if video_camera is not None:
        video_camera.stop()
        video_camera = None
    return jsonify({'success': True})

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('static/captures', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    print("\n" + "=" * 70)
    print("  AI Waste Classification System")
    print("=" * 70)
    print("\nStarting Flask server...")
    print("Open browser to: http://localhost:5000")
    print("=" * 70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)