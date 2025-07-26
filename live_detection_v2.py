#!/usr/bin/env python3
"""
Live Face Anti-Spoofing Detection System v2

This script combines YOLO face detection with HOG+SVM anti-spoofing classification
for real-time detection of real vs fake faces from webcam or video files.

Features:
- Real-time webcam detection
- Video file processing
- YOLO face detection + Anti-spoofing classification
- Live confidence display
- Recording capabilities
- Improved compatibility and error handling

Author: Face Anti-Spoofing System
Date: 2025
"""

import cv2
import numpy as np
import argparse
import joblib
import onnxruntime as ort
from pathlib import Path
import time
from skimage.feature import hog
from skimage import exposure
import os
import sys


class YOLOFaceDetector:
    """YOLO-based face detector using ONNX model"""
    
    def __init__(self, model_path, conf_threshold=0.5, nms_threshold=0.4):
        """
        Initialize YOLO face detector
        
        Args:
            model_path (str): Path to YOLO ONNX model
            conf_threshold (float): Confidence threshold for detections
            nms_threshold (float): NMS threshold
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        
        # Load ONNX model
        print(f"📥 Loading YOLO face detection model from {model_path}")
        try:
            self.session = ort.InferenceSession(model_path)
            
            # Get model input details
            self.input_name = self.session.get_inputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            self.input_height = self.input_shape[2]
            self.input_width = self.input_shape[3]
            
            print(f"✅ YOLO model loaded successfully")
            print(f"   Input shape: {self.input_shape}")
            print(f"   Input size: {self.input_width}x{self.input_height}")
            
        except Exception as e:
            print(f"❌ Error loading YOLO model: {str(e)}")
            raise

    def preprocess_image(self, image):
        """Preprocess image for YOLO inference"""
        # Resize image
        resized = cv2.resize(image, (self.input_width, self.input_height))
        
        # Normalize to [0, 1]
        input_image = resized.astype(np.float32) / 255.0
        
        # Change from HWC to CHW format
        input_image = np.transpose(input_image, (2, 0, 1))
        
        # Add batch dimension
        input_image = np.expand_dims(input_image, axis=0)
        
        return input_image

    def postprocess_detections(self, outputs, original_shape):
        """Post-process YOLO outputs to get face bounding boxes"""
        # Get original image dimensions
        orig_height, orig_width = original_shape[:2]
        
        # Scale factors
        scale_x = orig_width / self.input_width
        scale_y = orig_height / self.input_height
        
        boxes = []
        confidences = []
        
        # Process outputs (assuming YOLOv5 format)
        try:
            # Handle different output formats
            if isinstance(outputs, list) and len(outputs) > 0:
                output = outputs[0]
            else:
                output = outputs
                
            # Ensure output is numpy array
            if hasattr(output, 'squeeze'):
                output = output.squeeze()
            
            # If output has batch dimension, remove it
            if len(output.shape) == 3:
                output = output[0]
                
            # Process detections
            for detection in output:
                if len(detection) >= 5:
                    confidence = detection[4]
                    
                    if confidence > self.conf_threshold:
                        # Convert from center format to corner format
                        x_center = detection[0] * scale_x
                        y_center = detection[1] * scale_y
                        width = detection[2] * scale_x
                        height = detection[3] * scale_y
                        
                        x1 = int(x_center - width / 2)
                        y1 = int(y_center - height / 2)
                        x2 = int(x_center + width / 2)
                        y2 = int(y_center + height / 2)
                        
                        # Ensure coordinates are within image bounds
                        x1 = max(0, min(x1, orig_width))
                        y1 = max(0, min(y1, orig_height))
                        x2 = max(0, min(x2, orig_width))
                        y2 = max(0, min(y2, orig_height))
                        
                        # Only add if valid box
                        if x2 > x1 and y2 > y1:
                            boxes.append([x1, y1, x2, y2])
                            confidences.append(float(confidence))
        
        except Exception as e:
            print(f"⚠️  Warning in postprocessing: {str(e)}")
            return [], []
        
        # Apply Non-Maximum Suppression
        if boxes:
            try:
                # Convert to format expected by cv2.dnn.NMSBoxes
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
                    # Handle different OpenCV versions
                    if hasattr(indices, 'flatten'):
                        indices = indices.flatten()
                    else:
                        indices = np.array(indices).flatten()
                        
                    for i in indices:
                        final_boxes.append(boxes[i])
                        final_confidences.append(confidences[i])
                
                return final_boxes, final_confidences
            
            except Exception as e:
                print(f"⚠️  Warning in NMS: {str(e)}")
                # Return original boxes if NMS fails
                return boxes, confidences
        
        return [], []

    def detect_faces(self, image):
        """
        Detect faces in an image
        
        Args:
            image (numpy.ndarray): Input image (BGR format)
            
        Returns:
            tuple: (boxes, confidences) where boxes are [x1, y1, x2, y2] format
        """
        try:
            # Preprocess
            input_image = self.preprocess_image(image)
            
            # Run inference
            outputs = self.session.run(None, {self.input_name: input_image})
            
            # Post-process
            boxes, confidences = self.postprocess_detections(outputs, image.shape)
            
            return boxes, confidences
        
        except Exception as e:
            print(f"⚠️  Warning in face detection: {str(e)}")
            return [], []


class AntiSpoofingClassifier:
    """Anti-spoofing classifier using trained HOG+SVM model"""
    
    def __init__(self, model_path):
        """
        Initialize anti-spoofing classifier
        
        Args:
            model_path (str): Path to trained model file
        """
        self.model_path = model_path
        self.load_model()

    def load_model(self):
        """Load the trained anti-spoofing model"""
        print(f"📥 Loading anti-spoofing model from {self.model_path}")
        
        try:
            model_data = joblib.load(self.model_path)
            
            # Handle different model formats
            if 'model' in model_data:
                self.svm_model = model_data['model']
            elif 'svm_model' in model_data:
                self.svm_model = model_data['svm_model']
            else:
                # Assume the whole file is the model
                self.svm_model = model_data
            
            self.scaler = model_data.get('scaler', None)
            
            # Check training type to determine correct HOG parameters
            training_type = model_data.get('training_type', 'unknown')
            metadata = model_data.get('metadata', {})
            
            if training_type == 'improved_inference_aligned':
                # Use improved HOG parameters matching the training
                self.hog_params = {
                    'orientations': 12,
                    'pixels_per_cell': (6, 6),
                    'cells_per_block': (2, 2),
                    'visualize': False,
                    'block_norm': 'L2-Hys'
                }
            else:
                # Use default parameters for older models
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
            
            print(f"✅ Anti-spoofing model loaded successfully")
            print(f"   Training type: {training_type}")
            print(f"   Image size: {self.image_size}")
            print(f"   Classes: {self.class_names}")
            print(f"   HOG parameters:")
            print(f"     - Orientations: {self.hog_params['orientations']}")
            print(f"     - Pixels per cell: {self.hog_params['pixels_per_cell']}")
            print(f"     - Cells per block: {self.hog_params['cells_per_block']}")
            print(f"     - Block norm: {self.hog_params['block_norm']}")
            
            # Calculate expected feature size for verification
            expected_features = self.calculate_hog_feature_size()
            print(f"   Expected HOG features: {expected_features}")
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            raise

    def calculate_hog_feature_size(self):
        """Calculate expected HOG feature size based on parameters"""
        try:
            img_h, img_w = self.image_size
            ppc_h, ppc_w = self.hog_params['pixels_per_cell']
            cpb_h, cpb_w = self.hog_params['cells_per_block']
            orientations = self.hog_params['orientations']
            
            # Calculate number of cells
            cells_h = img_h // ppc_h
            cells_w = img_w // ppc_w
            
            # Calculate number of blocks
            blocks_h = cells_h - cpb_h + 1
            blocks_w = cells_w - cpb_w + 1
            
            # Calculate total features
            total_features = blocks_h * blocks_w * cpb_h * cpb_w * orientations
            
            return total_features
        except:
            return "unknown"

    def extract_hog_features(self, image):
        """Extract HOG features from face image"""
        try:
            # Ensure image is grayscale
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Normalize image
            image_normalized = exposure.equalize_hist(image)
            
            # Extract HOG features with current parameters
            features = hog(image_normalized, **self.hog_params)
            
            # Verify feature size matches expected
            expected_size = self.calculate_hog_feature_size()
            if isinstance(expected_size, int) and len(features) != expected_size:
                print(f"⚠️  Feature size mismatch: got {len(features)}, expected {expected_size}")
                print(f"   Using parameters: {self.hog_params}")
            
            return features
        
        except Exception as e:
            print(f"⚠️  Warning in HOG extraction: {str(e)}")
            print(f"   Image shape: {image.shape}")
            print(f"   HOG params: {self.hog_params}")
            
            # Try fallback with default parameters if the current ones fail
            try:
                print("   Trying fallback HOG parameters...")
                fallback_features = hog(
                    image_normalized,
                    orientations=9,
                    pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2),
                    block_norm='L2-Hys',
                    visualize=False
                )
                print(f"   Fallback successful: {len(fallback_features)} features")
                return fallback_features
            except:
                # Return zeros if everything fails
                print("   All HOG extraction methods failed, returning zeros")
                return np.zeros(1764)  # Default size

    def classify_face(self, face_image):
        """
        Classify a face as real or fake
        
        Args:
            face_image (numpy.ndarray): Face image
            
        Returns:
            tuple: (prediction, confidence, probabilities)
        """
        try:
            # Resize to expected size
            face_resized = cv2.resize(face_image, self.image_size)
            
            # Extract HOG features
            features = self.extract_hog_features(face_resized)
            
            # Scale features if scaler available
            if self.scaler:
                features_scaled = self.scaler.transform([features])
            else:
                features_scaled = [features]
            
            # Make prediction
            prediction = self.svm_model.predict(features_scaled)[0]
            
            # Get probabilities if available
            try:
                probabilities = self.svm_model.predict_proba(features_scaled)[0]
                confidence = probabilities[prediction]
            except:
                # If predict_proba not available, use decision function
                try:
                    decision = self.svm_model.decision_function(features_scaled)[0]
                    confidence = 1.0 / (1.0 + np.exp(-decision))  # Sigmoid
                    probabilities = [1-confidence, confidence] if prediction == 1 else [confidence, 1-confidence]
                except:
                    confidence = 0.5
                    probabilities = [0.5, 0.5]
            
            prediction_label = self.class_mapping.get(prediction, 'unknown')
            
            return prediction_label, confidence, probabilities
            
        except Exception as e:
            print(f"⚠️  Warning in classification: {str(e)}")
            return 'unknown', 0.5, [0.5, 0.5]


class LiveAntiSpoofingDetector:
    """Main live detection system combining YOLO + Anti-spoofing"""
    
    def __init__(self, yolo_model_path, antispoofing_model_path, 
                 conf_threshold=0.5, nms_threshold=0.4):
        """
        Initialize live detection system
        
        Args:
            yolo_model_path (str): Path to YOLO face detection model
            antispoofing_model_path (str): Path to anti-spoofing model
            conf_threshold (float): YOLO confidence threshold
            nms_threshold (float): YOLO NMS threshold
        """
        print("🚀 Initializing Live Anti-Spoofing Detection System")
        print("=" * 60)
        
        # Initialize components
        self.face_detector = YOLOFaceDetector(yolo_model_path, conf_threshold, nms_threshold)
        self.antispoofing_classifier = AntiSpoofingClassifier(antispoofing_model_path)
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_timer = time.time()
        self.current_fps = 0
        
        # Detection history for smoothing
        self.detection_history = {}
        self.history_length = 5
        
        print("✅ Live detection system ready!")
        print("=" * 60)

    def smooth_predictions(self, face_id, prediction, confidence):
        """Smooth predictions using history"""
        if face_id not in self.detection_history:
            self.detection_history[face_id] = []
        
        # Add current prediction
        self.detection_history[face_id].append((prediction, confidence))
        
        # Keep only recent history
        if len(self.detection_history[face_id]) > self.history_length:
            self.detection_history[face_id] = self.detection_history[face_id][-self.history_length:]
        
        # Calculate smoothed prediction
        history = self.detection_history[face_id]
        real_count = sum(1 for pred, _ in history if pred == 'real')
        fake_count = len(history) - real_count
        
        # Weighted average of confidences
        total_confidence = sum(conf for _, conf in history)
        avg_confidence = total_confidence / len(history)
        
        # Majority vote
        smoothed_prediction = 'real' if real_count > fake_count else 'fake'
        
        return smoothed_prediction, avg_confidence

    def update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_timer >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_timer = current_time

    def draw_detection_info(self, image, boxes, face_results):
        """Draw detection results on image"""
        height, width = image.shape[:2]
        
        # Draw FPS
        cv2.putText(image, f"FPS: {self.current_fps}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw face detections
        for i, (box, result) in enumerate(zip(boxes, face_results)):
            x1, y1, x2, y2 = box
            prediction, confidence, face_conf = result
            
            # Choose color based on prediction
            if prediction == 'real':
                color = (0, 255, 0)  # Green for real
                status = "✅ REAL"
            elif prediction == 'fake':
                color = (0, 0, 255)  # Red for fake
                status = "🚨 FAKE"
            else:
                color = (0, 255, 255)  # Yellow for unknown
                status = "❓ UNKNOWN"
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
            
            # Draw prediction label
            label = f"{status} ({confidence:.3f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Background for text
            cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Text
            cv2.putText(image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw face detection confidence
            face_label = f"Face: {face_conf:.3f}"
            cv2.putText(image, face_label, (x1, y2 + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw instructions
        instructions = [
            "Press 'q' to quit",
            "Press 's' to save screenshot",
            "Press 'r' to start/stop recording"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(image, instruction, (10, height - 60 + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def process_frame(self, frame):
        """Process a single frame"""
        # Detect faces
        boxes, face_confidences = self.face_detector.detect_faces(frame)
        
        face_results = []
        
        # Classify each detected face
        for i, (box, face_conf) in enumerate(zip(boxes, face_confidences)):
            x1, y1, x2, y2 = box
            
            # Extract face region with some padding
            padding = 10
            x1_pad = max(0, x1 - padding)
            y1_pad = max(0, y1 - padding)
            x2_pad = min(frame.shape[1], x2 + padding)
            y2_pad = min(frame.shape[0], y2 + padding)
            
            face_roi = frame[y1_pad:y2_pad, x1_pad:x2_pad]
            
            if face_roi.size > 0 and face_roi.shape[0] > 0 and face_roi.shape[1] > 0:
                # Classify face
                prediction, confidence, probabilities = self.antispoofing_classifier.classify_face(face_roi)
                
                # Apply smoothing
                smoothed_pred, smoothed_conf = self.smooth_predictions(i, prediction, confidence)
                
                face_results.append((smoothed_pred, smoothed_conf, face_conf))
            else:
                face_results.append(('unknown', 0.0, face_conf))
        
        return boxes, face_results

    def get_video_writer(self, output_path, fps, width, height):
        """Get video writer with fallback codecs"""
        # Try different codecs in order of preference
        codecs = ['mp4v', 'XVID', 'MJPG', 'X264']
        
        for codec in codecs:
            try:
                # Handle OpenCV version compatibility for FOURCC
                # Try to get FOURCC in a compatible way
                try:
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                except AttributeError:
                    try:
                        fourcc = cv2.cv.CV_FOURCC(*codec)
                    except AttributeError:
                        # As a last resort, try using CAP_PROP_FOURCC or raise error
                        try:
                            fourcc = int(cv2.CAP_PROP_FOURCC)
                        except Exception:
                            raise AttributeError("No suitable FOURCC function found in cv2 module.")
                writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                if writer.isOpened():
                    print(f"📹 Using codec: {codec}")
                    return writer
                writer.release()
            except:
                continue
        
        print("⚠️  Warning: Could not initialize video writer")
        return None

    def run_webcam(self, camera_id=0, save_video=False):
        """Run live detection on webcam"""
        print(f"🎥 Starting webcam detection (Camera {camera_id})")
        
        # Initialize webcam
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"❌ Error: Could not open camera {camera_id}")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Video recording setup
        video_writer = None
        recording = False
        
        if save_video:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            video_filename = f"antispoofing_detection_{timestamp}.avi"
            video_writer = self.get_video_writer(video_filename, 20.0, 640, 480)
            if video_writer:
                recording = True
                print(f"📹 Recording to: {video_filename}")
        
        print("🚀 Live detection started! Press 'q' to quit")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Error reading from camera")
                    break
                
                # Process frame
                boxes, face_results = self.process_frame(frame)
                
                # Draw results
                self.draw_detection_info(frame, boxes, face_results)
                
                # Update FPS
                self.update_fps()
                
                # Display frame
                cv2.imshow('Live Anti-Spoofing Detection', frame)
                
                # Record video if enabled
                if recording and video_writer:
                    video_writer.write(frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save screenshot
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    screenshot_name = f"screenshot_{timestamp}.jpg"
                    cv2.imwrite(screenshot_name, frame)
                    print(f"📸 Screenshot saved: {screenshot_name}")
                elif key == ord('r'):
                    # Toggle recording
                    if not recording and video_writer is None:
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        video_filename = f"antispoofing_detection_{timestamp}.avi"
                        video_writer = self.get_video_writer(video_filename, 20.0, 640, 480)
                        if video_writer:
                            recording = True
                            print(f"📹 Started recording: {video_filename}")
                    elif recording:
                        recording = False
                        if video_writer:
                            video_writer.release()
                            video_writer = None
                        print("⏹️  Recording stopped")
        
        except KeyboardInterrupt:
            print("\n🛑 Detection stopped by user")
        
        finally:
            # Cleanup
            cap.release()
            if video_writer:
                video_writer.release()
            cv2.destroyAllWindows()
            print("✅ Cleanup completed")

    def process_video_file(self, video_path, output_path=None):
        """Process a video file"""
        print(f"🎥 Processing video file: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"❌ Error: Could not open video file {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📊 Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Setup output video
        out = None
        if output_path:
            out = self.get_video_writer(output_path, fps, width, height)
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                boxes, face_results = self.process_frame(frame)
                
                # Draw results
                self.draw_detection_info(frame, boxes, face_results)
                
                # Write output frame
                if out:
                    out.write(frame)
                
                # Display progress
                frame_count += 1
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"⏳ Processing: {progress:.1f}% ({frame_count}/{total_frames})")
                
                # Display frame (optional)
                cv2.imshow('Video Processing', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        except KeyboardInterrupt:
            print("\n🛑 Processing stopped by user")
        
        finally:
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()
            print(f"✅ Video processing completed: {output_path}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Live Face Anti-Spoofing Detection System v2')
    parser.add_argument('--yolo_model', type=str, 
                       default='face_detection_model/yolov5s-face.onnx',
                       help='Path to YOLO face detection model')
    parser.add_argument('--antispoofing_model', type=str,
                        default="anti_spoofing_model/smart_antispoofing_model_20250621_152348_acc_0.8856.pkl",
                       help='Path to trained anti-spoofing model')
    parser.add_argument('--mode', type=str, choices=['webcam', 'video'], 
                       default='webcam',
                       help='Detection mode: webcam or video file')
    parser.add_argument('--camera_id', type=int, default=0,
                       help='Camera ID for webcam mode')
    parser.add_argument('--video_input', type=str,
                       help='Input video file path')
    parser.add_argument('--video_output', type=str,
                       help='Output video file path')
    parser.add_argument('--conf_threshold', type=float, default=0.5,
                       help='YOLO confidence threshold')
    parser.add_argument('--nms_threshold', type=float, default=0.4,
                       help='YOLO NMS threshold')
    parser.add_argument('--save_video', action='store_true',
                       help='Save detection video in webcam mode')
    
    args = parser.parse_args()
    
    # Auto-detect anti-spoofing model if not provided
    if not args.antispoofing_model:
        model_dir = Path("models")
        if model_dir.exists():
            model_files = list(model_dir.glob("*.pkl"))
            if model_files:
                # Use the most recent model
                args.antispoofing_model = str(sorted(model_files)[-1])
                print(f"🔍 Auto-detected model: {args.antispoofing_model}")
            else:
                print("❌ Error: No .pkl model files found in models/ directory")
                return
        else:
            print("❌ Error: models/ directory not found and --antispoofing_model not specified")
            return
    
    # Validate arguments
    if args.mode == 'video' and not args.video_input:
        print("❌ Error: --video_input required for video mode")
        return
    
    if not Path(args.yolo_model).exists():
        print(f"❌ Error: YOLO model not found: {args.yolo_model}")
        return
    
    if not Path(args.antispoofing_model).exists():
        print(f"❌ Error: Anti-spoofing model not found: {args.antispoofing_model}")
        return
    
    try:
        # Initialize detector
        detector = LiveAntiSpoofingDetector(
            yolo_model_path=args.yolo_model,
            antispoofing_model_path=args.antispoofing_model,
            conf_threshold=args.conf_threshold,
            nms_threshold=args.nms_threshold
        )
        
        # Run detection
        if args.mode == 'webcam':
            detector.run_webcam(args.camera_id, args.save_video)
        else:
            detector.process_video_file(args.video_input, args.video_output)
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
