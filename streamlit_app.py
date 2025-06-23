"""
Anti-Spoofing Web Interface
Streamlit-based web application for face anti-spoofing detection
Supports both image upload and live webcam detection
"""

import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import subprocess
import threading
from PIL import Image
import time
import queue
import joblib
import onnxruntime as ort
from skimage.feature import hog
from skimage import exposure
from pathlib import Path
import streamlit_webrtc as webrtc
from streamlit_webrtc import VideoProcessorBase
from av import VideoFrame

# Page configuration
st.set_page_config(
    page_title="Face Anti-Spoofing Detection",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.result-real {
    background-color: #d4edda;
    border: 2px solid #28a745;
    border-radius: 10px;
    padding: 1rem;
    margin: 1rem 0;
}
.result-fake {
    background-color: #f8d7da;
    border: 2px solid #dc3545;
    border-radius: 10px;
    padding: 1rem;
    margin: 1rem 0;
}
.result-uncertain {
    background-color: #fff3cd;
    border: 2px solid #ffc107;
    border-radius: 10px;
    padding: 1rem;
    margin: 1rem 0;
}
.confidence-bar {
    height: 25px;
    border-radius: 12px;
    margin: 5px 0;
}
</style>
""", unsafe_allow_html=True)

class YOLOFaceDetector:
    """YOLO-based face detector using ONNX model"""
    
    def __init__(self, model_path, conf_threshold=0.5, nms_threshold=0.4):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        
        # Load ONNX model
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def preprocess_image(self, image):
        """Preprocess image for YOLO inference"""
        resized = cv2.resize(image, (self.input_width, self.input_height))
        input_image = resized.astype(np.float32) / 255.0
        input_image = np.transpose(input_image, (2, 0, 1))
        input_image = np.expand_dims(input_image, axis=0)
        return input_image

    def postprocess_detections(self, outputs, original_shape):
        """Post-process YOLO outputs to get face bounding boxes"""
        orig_height, orig_width = original_shape[:2]
        scale_x = orig_width / self.input_width
        scale_y = orig_height / self.input_height
        
        boxes = []
        confidences = []
        
        try:
            if isinstance(outputs, list) and len(outputs) > 0:
                output = outputs[0]
            else:
                output = outputs
                
            if hasattr(output, 'squeeze'):
                output = output.squeeze()
            
            if len(output.shape) == 3:
                output = output[0]
                
            for detection in output:
                if len(detection) >= 5:
                    confidence = detection[4]
                    
                    if confidence > self.conf_threshold:
                        x_center = detection[0] * scale_x
                        y_center = detection[1] * scale_y
                        width = detection[2] * scale_x
                        height = detection[3] * scale_y
                        
                        x1 = int(x_center - width / 2)
                        y1 = int(y_center - height / 2)
                        x2 = int(x_center + width / 2)
                        y2 = int(y_center + height / 2)
                        
                        x1 = max(0, min(x1, orig_width))
                        y1 = max(0, min(y1, orig_height))
                        x2 = max(0, min(x2, orig_width))
                        y2 = max(0, min(y2, orig_height))
                        
                        if x2 > x1 and y2 > y1:
                            boxes.append([x1, y1, x2, y2])
                            confidences.append(float(confidence))
        
        except Exception:
            return [], []
        
        # Apply NMS
        if boxes:
            try:
                boxes_xywh = []
                for box in boxes:
                    x1, y1, x2, y2 = box
                    boxes_xywh.append([x1, y1, x2 - x1, y2 - y1])
                
                indices = cv2.dnn.NMSBoxes(
                    boxes_xywh, confidences, 
                    self.conf_threshold, self.nms_threshold
                )
                
                final_boxes = []
                final_confidences = []
                
                if len(indices) > 0:
                    if hasattr(indices, 'flatten'):
                        indices = indices.flatten()
                    else:
                        indices = np.array(indices).flatten()
                        
                    for i in indices:
                        final_boxes.append(boxes[i])
                        final_confidences.append(confidences[i])
                
                return final_boxes, final_confidences
            
            except Exception:
                return boxes, confidences
        
        return [], []

    def detect_faces(self, image):
        """Detect faces in an image"""
        try:
            input_image = self.preprocess_image(image)
            outputs = self.session.run(None, {self.input_name: input_image})
            boxes, confidences = self.postprocess_detections(outputs, image.shape)
            return boxes, confidences
        except Exception:
            return [], []


class AntiSpoofingClassifier:
    """Anti-spoofing classifier using trained HOG+SVM model"""
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.load_model()

    def load_model(self):
        """Load the trained anti-spoofing model"""
        try:
            model_data = joblib.load(self.model_path)
            
            if 'model' in model_data:
                self.svm_model = model_data['model']
            elif 'svm_model' in model_data:
                self.svm_model = model_data['svm_model']
            else:
                self.svm_model = model_data
            
            self.scaler = model_data.get('scaler', None)
            self.hog_params = model_data.get('hog_params', {
                'orientations': 9,
                'pixels_per_cell': (8, 8),
                'cells_per_block': (2, 2),
                'visualize': False,
                'transform_sqrt': True,
                'block_norm': 'L2-Hys'
            })
            self.image_size = model_data.get('image_size', (64, 64))
            self.class_mapping = model_data.get('class_mapping', {0: 'fake', 1: 'real'})
            self.class_names = model_data.get('class_names', ['fake', 'real'])
            
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")

    def extract_hog_features(self, image):
        """Extract HOG features from face image"""
        try:
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            image_normalized = exposure.equalize_hist(image)
            features = hog(image_normalized, **self.hog_params)
            return features
        except Exception:
            return np.zeros(1764)  # Default HOG feature size

    def classify_face(self, face_image):
        """Classify a face as real or fake"""
        try:
            face_resized = cv2.resize(face_image, self.image_size)
            features = self.extract_hog_features(face_resized)
            
            if self.scaler:
                features_scaled = self.scaler.transform([features])
            else:
                features_scaled = [features]
            
            prediction = self.svm_model.predict(features_scaled)[0]
            
            try:
                probabilities = self.svm_model.predict_proba(features_scaled)[0]
                confidence = probabilities[prediction]
            except:
                try:
                    decision = self.svm_model.decision_function(features_scaled)[0]
                    confidence = 1.0 / (1.0 + np.exp(-decision))
                    probabilities = [1-confidence, confidence] if prediction == 1 else [confidence, 1-confidence]
                except:
                    confidence = 0.5
                    probabilities = [0.5, 0.5]
            
            prediction_label = self.class_mapping.get(prediction, 'unknown')
            return prediction_label, confidence, probabilities
            
        except Exception:
            return 'unknown', 0.5, [0.5, 0.5]


@st.cache_resource
def load_models():
    """Load the face detection and anti-spoofing models"""
    try:
        # Load YOLO face detection model
        yolo_path = "face_detection_model/yolov5s-face.onnx"
        if not os.path.exists(yolo_path):
            return None, None, f"YOLO model not found: {yolo_path}"
        
        # Auto-detect the latest anti-spoofing model
        model_dir = Path("anti_spoofing_model")
        if not model_dir.exists():
            return None, None, "Models directory not found"
        
        model_files = list(model_dir.glob("*.pkl"))
        if not model_files:
            return None, None, "No .pkl model files found in models/ directory"
        
        # Use the most recent model
        antispoofing_path = str(sorted(model_files)[-1])
        
        face_detector = YOLOFaceDetector(yolo_path, conf_threshold=0.5, nms_threshold=0.4)
        anti_spoof = AntiSpoofingClassifier(antispoofing_path)
        
        return face_detector, anti_spoof, None
    except Exception as e:
        return None, None, str(e)

def increased_crop(img, bbox, bbox_inc=1.5):
    """Crop face region with increased bounding box"""
    real_h, real_w = img.shape[:2]
    
    x, y, w, h = bbox
    w, h = w - x, h - y
    l = max(w, h)
    
    xc, yc = x + w/2, y + h/2
    x, y = int(xc - l*bbox_inc/2), int(yc - l*bbox_inc/2)
    x1 = 0 if x < 0 else x 
    y1 = 0 if y < 0 else y
    x2 = real_w if x + l*bbox_inc > real_w else x + int(l*bbox_inc)
    y2 = real_h if y + l*bbox_inc > real_h else y + int(l*bbox_inc)
    
    img = img[y1:y2, x1:x2, :]
    img = cv2.copyMakeBorder(img, 
                             y1-y, int(l*bbox_inc-y2+y), 
                             x1-x, int(l*bbox_inc)-x2+x, 
                             cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return img

def predict_image(img, face_detector, anti_spoof):
    """Make prediction on uploaded image"""
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    img_array = np.array(img)
    
    # Detect faces
    boxes, confidences = face_detector.detect_faces(img_array)
    
    if not boxes:
        return None, None, None, None, "No face detected"
    
    # Get the first face
    bbox = boxes[0]
    
    # Crop face for anti-spoofing
    x1, y1, x2, y2 = bbox
    cropped_face = img_array[y1:y2, x1:x2]
    
    # Make prediction
    prediction_label, confidence, probabilities = anti_spoof.classify_face(cropped_face)
    
    # Extract probabilities
    if prediction_label == 'real':
        real_confidence = confidence
        fake_confidence = 1 - confidence
        label = 1
    else:
        fake_confidence = confidence
        real_confidence = 1 - confidence
        label = 0
    
    return bbox, label, real_confidence, fake_confidence, None

def draw_bounding_box(img, bbox, label, real_conf, fake_conf, threshold=0.5):
    """Draw bounding box and prediction on image"""
    img_array = np.array(img)
    
    if bbox is not None:
        x1, y1, x2, y2 = bbox
          # Determine color
        if label == 1 and real_conf > threshold:
            color = (0, 255, 0)  # Green for real
            text = f"REAL: {real_conf:.3f}"
        elif label == 1:
            color = (255, 165, 0)  # Orange for uncertain
            text = f"UNCERTAIN: {real_conf:.3f}"
        else:
            color = (255, 0, 0)  # Red for fake
            text = f"FAKE: {fake_conf:.3f}"
        
        # Draw bounding box
        cv2.rectangle(img_array, (x1, y1), (x2, y2), color, 3)
        
        # Add text
        cv2.putText(img_array, text, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    return Image.fromarray(img_array)

def create_confidence_bars(real_conf, fake_conf):
    """Create HTML confidence bars"""
    real_percentage = real_conf * 100
    fake_percentage = fake_conf * 100
    
    html = f"""
    <div style="margin: 1rem 0;">
        <div style="margin-bottom: 0.5rem;">
            <strong>Real Face Confidence: {real_percentage:.1f}%</strong>
            <div style="background-color: #e0e0e0; border-radius: 12px; height: 25px;">
                <div class="confidence-bar" style="width: {real_percentage}%; background-color: #28a745;"></div>
            </div>
        </div>
        <div>
            <strong>Fake Face Confidence: {fake_percentage:.1f}%</strong>
            <div style="background-color: #e0e0e0; border-radius: 12px; height: 25px;">
                <div class="confidence-bar" style="width: {fake_percentage}%; background-color: #dc3545;"></div>
            </div>
        </div>
    </div>
    """
    return html

def main():
    # Header
    st.markdown('<h1 class="main-header">🔒 Face Anti-Spoofing Detection</h1>', unsafe_allow_html=True)
    
    # Main content (only live webcam detection)
    st.header("Live Webcam Detection (Real-Time)")

    # Load models
    with st.spinner("Loading AI models..."):
        face_detector, anti_spoof, error = load_models()

    if error:
        st.error(f"❌ Error loading models: {error}")
        st.stop()

    st.success("✅ Models loaded successfully!")

    # Sidebar
    st.sidebar.header("⚙️ Settings")
    threshold = st.sidebar.slider(
        "Detection Threshold", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.5, 
        step=0.05,
        help="Higher values = more strict detection"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📖 About")
    st.sidebar.markdown("""
    This application uses advanced AI to detect face spoofing attacks:
    - **Real Face**: Live person detected
    - **Fake Face**: Photo, video, or mask detected
    - **Uncertain**: Low confidence prediction
    """)    # Create video processor factory
    class LocalVideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.face_detector, self.anti_spoof, self.error = load_models()
            self.threshold = threshold
            self.stats = {
                'total_frames': 0,
                'real_count': 0,
                'fake_count': 0,
                'uncertain_count': 0,
                'no_face_count': 0
            }

        def recv(self, frame: VideoFrame) -> VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Check if models are loaded
            if self.face_detector is None or self.anti_spoof is None:
                return frame
            
            # Detect faces
            boxes, confidences = self.face_detector.detect_faces(rgb_img)
            
            self.stats['total_frames'] += 1
            
            if not boxes:
                self.stats['no_face_count'] += 1
                return frame
            
            # Get the first face
            bbox = boxes[0]
            x1, y1, x2, y2 = bbox
            
            # Crop face for anti-spoofing
            cropped_face = rgb_img[y1:y2, x1:x2]
            
            # Make prediction
            prediction_label, confidence, probabilities = self.anti_spoof.classify_face(cropped_face)
            
            # Determine confidence values and label
            if prediction_label == 'real':
                real_confidence = confidence
                fake_confidence = 1 - confidence
                label = 1
            else:
                fake_confidence = confidence
                real_confidence = 1 - confidence
                label = 0
            
            # Determine display color and text
            if label == 1 and real_confidence > self.threshold:
                self.stats['real_count'] += 1
                color = (0, 255, 0)  # Green for real
                text = f"REAL: {real_confidence:.3f}"
            elif label == 1:
                self.stats['uncertain_count'] += 1
                color = (255, 165, 0)  # Orange for uncertain
                text = f"UNCERTAIN: {real_confidence:.3f}"
            else:
                self.stats['fake_count'] += 1
                color = (0, 0, 255)  # Red for fake
                text = f"FAKE: {fake_confidence:.3f}"
            
            # Draw bounding box and text
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            cv2.putText(img, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            return VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_ctx = webrtc.webrtc_streamer(
        key="face-anti-spoofing",
        mode=webrtc.WebRtcMode.SENDRECV,
        video_processor_factory=LocalVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    # Show stats if webrtc context exists
    if webrtc_ctx.video_processor:
        stats = webrtc_ctx.video_processor.stats
    else:
        stats = {'total_frames': 0, 'real_count': 0, 'fake_count': 0, 'uncertain_count': 0, 'no_face_count': 0}
    stats_col1, stats_col2, stats_col3, stats_col4, stats_col5 = st.columns(5)
    with stats_col1:
        st.metric("Total Frames", stats['total_frames'])
    with stats_col2:
        st.metric("Real Faces", stats['real_count'], delta=None)
    with stats_col3:
        st.metric("Fake Faces", stats['fake_count'], delta=None)
    with stats_col4:
        st.metric("Uncertain", stats['uncertain_count'], delta=None)
    with stats_col5:
        st.metric("No Face", stats['no_face_count'], delta=None)

    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col2:
        st.markdown("### 🛡️ Security Notice")
        st.info("This tool is for educational and testing purposes. For production security systems, additional validation and testing is recommended.")

if __name__ == "__main__":
    main()
