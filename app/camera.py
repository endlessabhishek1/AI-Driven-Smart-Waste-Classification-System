import cv2
import threading
import time
import numpy as np
from queue import Queue

class VideoCamera:
    def __init__(self, classifier):
        # Open camera
        self.video = cv2.VideoCapture(0)
        # Set resolution to 1280x720 for better quality
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.classifier = classifier
        
        # Frame synchronization
        self.current_frame = None  # Raw frame from camera
        self.processed_frame = None # Frame with bounding boxes
        self.last_prediction = None
        
        # Threads control
        self.is_running = True
        self.lock = threading.Lock()
        
        # Start capture thread (reads from webcam)
        self.capture_thread = threading.Thread(target=self.update_camera)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        # Start processing thread (runs AI)
        self.process_thread = threading.Thread(target=self.process_ai)
        self.process_thread.daemon = True
        self.process_thread.start()
    
    def update_camera(self):
        """Thread 1: Capture frames from webcam at max FPS"""
        while self.is_running:
            success, frame = self.video.read()
            if success:
                with self.lock:
                    self.current_frame = frame
            else:
                # If camera fails, try to reconnect
                time.sleep(0.1)
    
    def process_ai(self):
        """Thread 2: Run AI inference on latest frame"""
        while self.is_running:
            frame_to_process = None
            
            # Get latest frame safely
            with self.lock:
                if self.current_frame is not None:
                    frame_to_process = self.current_frame.copy()
            
            if frame_to_process is not None:
                # Run detection (this might take 100-200ms)
                # We don't lock here to avoid blocking capture thread
                try:
                    annotated_frame, prediction = self.classifier.detect_and_draw(frame_to_process)
                    
                    # Update results
                    with self.lock:
                        self.processed_frame = annotated_frame
                        self.last_prediction = prediction
                except Exception as e:
                    print(f"Error in AI processing: {e}")
            
            # Avoid hogging CPU if no new frame
            time.sleep(0.01)
    
    def get_frame(self):
        """Get best available frame for display"""
        with self.lock:
            # Prefer processed frame (with bounding boxes) if available
            # But if it's too old/unavailable, fall back to current frame
            if self.processed_frame is not None:
                ret, jpeg = cv2.imencode('.jpg', self.processed_frame)
                return jpeg.tobytes()
            elif self.current_frame is not None:
                ret, jpeg = cv2.imencode('.jpg', self.current_frame)
                return jpeg.tobytes()
            
        return None
    
    def get_prediction(self):
        """Get latest prediction"""
        with self.lock:
            return self.last_prediction
    
    def stop(self):
        """Stop threads and release camera"""
        self.is_running = False
        if self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
        if self.process_thread.is_alive():
            self.process_thread.join(timeout=1.0)
            
        if self.video.isOpened():
            self.video.release()
    
    def capture_snapshot(self):
        """Capture and save current raw frame"""
        frame_to_save = None
        with self.lock:
            if self.current_frame is not None:
                frame_to_save = self.current_frame.copy()
                
        if frame_to_save is not None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"static/captures/snapshot_{timestamp}.jpg"
            # Ensure directory exists
            import os
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            cv2.imwrite(filename, frame_to_save)
            return filename
        return None

    def capture_burst(self, count=10, delay=0.1):
        """Capture multiple frames in quick succession for training data."""
        filenames = []
        import os
        
        for i in range(count):
            with self.lock:
                if self.current_frame is not None:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"static/captures/burst_{timestamp}_{i}.jpg"
                    os.makedirs(os.path.dirname(filename), exist_ok=True)
                    cv2.imwrite(filename, self.current_frame)
                    filenames.append(filename)
            time.sleep(delay)
            
        return filenames