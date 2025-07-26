#!/usr/bin/env python3
"""
Improved Anti-Spoofing Training with Better Diagnostics

This script addresses the issues found in the previous training:
1. Better face detection validation
2. Data augmentation for small datasets
3. Improved HOG parameters
4. Better class balancing
5. Comprehensive diagnostics

Author: Face Anti-Spoofing System
Date: 2025
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.feature import hog
from skimage import exposure
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib
import seaborn as sns
from tqdm import tqdm
import argparse
import json
import time
import onnxruntime as ort
from datetime import datetime
from sklearn.utils import resample


class ImprovedYOLOFaceDetector:
    """Improved YOLO face detector with better validation"""
    
    def __init__(self, model_path, conf_threshold=0.3, nms_threshold=0.4):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        
        print(f"📥 Loading YOLO face detection model from {model_path}")
        try:
            self.session = ort.InferenceSession(model_path)
            self.input_name = self.session.get_inputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            print(f"✅ Model loaded successfully. Input shape: {self.input_shape}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise

    def preprocess_image(self, image):
        """Preprocess image for YOLO inference"""
        input_size = 640
        
        h, w = image.shape[:2]
        scale = input_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        resized = cv2.resize(image, (new_w, new_h))
        
        # Create padded image
        padded = np.full((input_size, input_size, 3), 128, dtype=np.uint8)
        padded[:new_h, :new_w] = resized
        
        # Convert to RGB and normalize
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        
        # Transpose to CHW format and add batch dimension
        input_tensor = np.transpose(normalized, (2, 0, 1))[np.newaxis, ...]
        
        return input_tensor, scale

    def detect_faces(self, image):
        """Detect faces with improved parsing"""
        try:
            input_tensor, scale = self.preprocess_image(image)
            
            # Run inference
            outputs = self.session.run(None, {self.input_name: input_tensor})
            
            # Handle different output formats
            detections = outputs[0]
            if len(detections.shape) == 3:
                detections = detections[0]  # Remove batch dimension
            
            faces = []
            h, w = image.shape[:2]
            
            for detection in detections:
                if len(detection) >= 5:
                    confidence = detection[4]
                    if confidence > self.conf_threshold:
                        # Parse coordinates (could be in different formats)
                        if len(detection) >= 5:
                            # Try different coordinate formats
                            x1, y1, x2, y2 = detection[0], detection[1], detection[2], detection[3]
                            
                            # Scale back to original image size
                            x1 = int(x1 / scale) if x1 > 1 else int(x1 * w)
                            y1 = int(y1 / scale) if y1 > 1 else int(y1 * h)
                            x2 = int(x2 / scale) if x2 > 1 else int(x2 * w)
                            y2 = int(y2 / scale) if y2 > 1 else int(y2 * h)
                            
                            # Ensure coordinates are within bounds
                            x1, y1 = max(0, min(x1, x2)), max(0, min(y1, y2))
                            x2, y2 = min(w, max(x1, x2)), min(h, max(y1, y2))
                            
                            if x2 > x1 and y2 > y1 and (x2-x1) > 20 and (y2-y1) > 20:
                                faces.append((x1, y1, x2, y2, confidence))
            
            return faces
        except Exception as e:
            print(f"❌ Error in face detection: {e}")
            return []


class ImprovedTrainer:
    """Improved trainer with better data handling and augmentation"""
    
    def __init__(self, dataset_path, face_detector, image_size=(64, 64)):
        self.dataset_path = Path(dataset_path)
        self.face_detector = face_detector
        self.image_size = image_size
        
    def get_label_from_path(self, image_path):
        """Get ground truth label from YOLO label file"""
        image_path = Path(image_path)
        
        # Convert images path to labels path
        label_path = str(image_path).replace('\\images\\', '\\labels\\').replace('/images/', '/labels/')
        label_path = label_path.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt')
        
        try:
            with open(label_path, 'r') as f:
                line = f.readline().strip()
                if line:
                    class_id = int(line.split()[0])
                    return class_id  # 0 = fake, 1 = real
        except Exception as e:
            return None
        return None
    
    def augment_face(self, face):
        """Simple data augmentation for face images"""
        augmented_faces = [face]  # Original
        
        # Horizontal flip
        augmented_faces.append(cv2.flip(face, 1))
        
        # Brightness variations
        bright = cv2.convertScaleAbs(face, alpha=1.2, beta=10)
        dark = cv2.convertScaleAbs(face, alpha=0.8, beta=-10)
        augmented_faces.extend([bright, dark])
        
        # Slight rotations
        center = (face.shape[1]//2, face.shape[0]//2)
        for angle in [-5, 5]:
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(face, M, (face.shape[1], face.shape[0]))
            augmented_faces.append(rotated)
        
        return augmented_faces
    
    def extract_faces_and_labels(self, split='train', use_augmentation=True):
        """Extract face crops with improved validation and augmentation"""
        print(f"\n🔍 Extracting faces from {split} set...")
        
        images_dir = self.dataset_path / split / 'images'
        face_crops = []
        labels = []
        
        image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.jpeg')) + list(images_dir.glob('*.png'))
        print(f"Found {len(image_files)} images in {split} set")
        
        successful_detections = 0
        failed_detections = 0
        
        for image_path in tqdm(image_files, desc=f"Processing {split} images"):
            # Get ground truth label
            gt_label = self.get_label_from_path(image_path)
            if gt_label is None:
                continue
                
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                continue
                
            # Detect faces
            faces = self.face_detector.detect_faces(image)
            
            if faces:
                successful_detections += 1
                # Take the face with highest confidence
                best_face = max(faces, key=lambda x: x[4])
                x1, y1, x2, y2, confidence = best_face
                
                # Crop face
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                face_crop = gray[y1:y2, x1:x2]
                
                if face_crop.size > 0:
                    face_resized = cv2.resize(face_crop, self.image_size)
                    
                    # Add original face
                    face_crops.append(face_resized)
                    labels.append(gt_label)
                    
                    # Add augmented faces for training set only
                    if split == 'train' and use_augmentation:
                        augmented = self.augment_face(face_resized)
                        for aug_face in augmented[1:]:  # Skip original
                            face_crops.append(aug_face)
                            labels.append(gt_label)
            else:
                failed_detections += 1
        
        print(f"✅ Successful face detections: {successful_detections}")
        print(f"❌ Failed face detections: {failed_detections}")
        print(f"📊 Total faces extracted: {len(face_crops)}")
        print(f"   - Real faces: {sum(labels)}")
        print(f"   - Fake faces: {len(labels) - sum(labels)}")
        
        return np.array(face_crops), np.array(labels)
    
    def balance_dataset(self, X, y):
        """Balance the dataset using resampling"""
        print("\n⚖️ Balancing dataset...")
        
        # Separate classes
        X_fake = X[y == 0]
        X_real = X[y == 1]
        
        print(f"Before balancing: {len(X_fake)} fake, {len(X_real)} real")
        
        # Resample to balance classes
        min_samples = min(len(X_fake), len(X_real))
        max_samples = max(len(X_fake), len(X_real))
        
        if min_samples < max_samples:
            # Upsample minority class
            if len(X_fake) < len(X_real):
                X_fake_resampled, y_fake_resampled = resample(
                    X_fake, [0] * len(X_fake), 
                    n_samples=len(X_real), 
                    random_state=42
                )
                X_balanced = np.vstack([X_fake_resampled, X_real])
                y_balanced = np.hstack([y_fake_resampled, [1] * len(X_real)])
            else:
                X_real_resampled, y_real_resampled = resample(
                    X_real, [1] * len(X_real), 
                    n_samples=len(X_fake), 
                    random_state=42
                )
                X_balanced = np.vstack([X_fake, X_real_resampled])
                y_balanced = np.hstack([[0] * len(X_fake), y_real_resampled])
        else:
            X_balanced, y_balanced = X, y
        
        print(f"After balancing: {len(X_balanced[y_balanced == 0])} fake, {len(X_balanced[y_balanced == 1])} real")
        
        return X_balanced, y_balanced
    
    def extract_hog_features(self, images):
        """Extract HOG features with improved parameters"""
        print("🔧 Extracting HOG features...")
        
        features = []
        for img in tqdm(images, desc="Extracting HOG"):
            # Try multiple HOG configurations
            hog_features = hog(
                img,
                orientations=12,  # Increased from 9
                pixels_per_cell=(6, 6),  # Decreased from (8,8) for more detail
                cells_per_block=(2, 2),
                block_norm='L2-Hys',
                visualize=False,
                feature_vector=True
            )
            features.append(hog_features)
        
        return np.array(features)
    
    def train_svm(self, X_train, y_train):
        """Train SVM with comprehensive parameter search"""
        print("\n🤖 Training SVM classifier...")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Comprehensive parameter grid
        param_grid = [
            {
                'kernel': ['linear'],
                'C': [0.01, 0.1, 1, 10, 100, 1000],
                'class_weight': [None, 'balanced']
            },
            {
                'kernel': ['rbf'],
                'C': [0.1, 1, 10, 100, 1000],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0],
                'class_weight': [None, 'balanced']
            },
            {
                'kernel': ['poly'],
                'C': [0.1, 1, 10, 100],
                'degree': [2, 3],
                'gamma': ['scale', 'auto'],
                'class_weight': [None, 'balanced']
            }
        ]
        
        # Grid search with cross-validation
        svm = SVC(probability=True, random_state=42)
        grid_search = GridSearchCV(
            svm, param_grid, cv=5, scoring='accuracy',
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train_scaled, y_train)
        
        print(f"✅ Best SVM parameters: {grid_search.best_params_}")
        print(f"✅ Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_, scaler
    
    def evaluate_model(self, model, scaler, X_test, y_test):
        """Comprehensive model evaluation"""
        print("\n📊 Evaluating model...")
        
        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)
        
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=['Fake', 'Real'])
        
        print(f"✅ Test Accuracy: {accuracy:.4f}")
        print(f"📈 Sample distribution - Test set: {len(y_test[y_test==0])} fake, {len(y_test[y_test==1])} real")
        print("\n📈 Classification Report:")
        print(report)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n🔍 Confusion Matrix:")
        print(f"        Predicted")
        print(f"Actual  Fake  Real")
        print(f"Fake    {cm[0,0]:4d}  {cm[0,1]:4d}")
        print(f"Real    {cm[1,0]:4d}  {cm[1,1]:4d}")
        
        return accuracy, report
    
    def save_model(self, model, scaler, accuracy, metadata=None):
        """Save the improved model"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        model_dir = Path("anti_spoofing_model")
        model_dir.mkdir(exist_ok=True)
        
        model_filename = f"improved_model_{timestamp}_acc_{accuracy:.4f}.pkl"
        model_path = model_dir / model_filename
        
        model_data = {
            'model': model,
            'scaler': scaler,
            'image_size': self.image_size,
            'accuracy': accuracy,
            'timestamp': timestamp,
            'training_type': 'improved_inference_aligned',
            'metadata': metadata or {}
        }
        
        joblib.dump(model_data, model_path)
        print(f"✅ Model saved to: {model_path}")
        
        return model_path


def main():
    parser = argparse.ArgumentParser(description='Improved Anti-Spoofing Training')
    parser.add_argument('--dataset', type=str, default='Face Anti-Spoofing.v4i.yolov11',
                       help='Path to dataset directory')
    parser.add_argument('--yolo-model', type=str, default='face_detection_model/yolov5s-face.onnx',
                       help='Path to YOLO face detection model')
    parser.add_argument('--image-size', type=int, nargs=2, default=[64, 64],
                       help='Image size for face crops')
    parser.add_argument('--no-augmentation', action='store_true',
                       help='Disable data augmentation')
    
    args = parser.parse_args()
    
    print("🚀 IMPROVED ANTI-SPOOFING TRAINER")
    print("=" * 50)
    
    # Initialize improved face detector
    face_detector = ImprovedYOLOFaceDetector(args.yolo_model, conf_threshold=0.3)
    
    # Initialize improved trainer
    trainer = ImprovedTrainer(args.dataset, face_detector, tuple(args.image_size))
    
    # Extract and combine all data for better training
    print("\n📥 Extracting training data...")
    train_faces, train_labels = trainer.extract_faces_and_labels('train', not args.no_augmentation)
    
    print("\n📥 Extracting validation data...")
    val_faces, val_labels = trainer.extract_faces_and_labels('valid', False)
    
    # Combine train and validation for larger dataset
    if len(val_faces) > 0:
        print("\n🔄 Combining train and validation sets for larger dataset...")
        all_faces = np.vstack([train_faces, val_faces])
        all_labels = np.hstack([train_labels, val_labels])
    else:
        all_faces = train_faces
        all_labels = train_labels
    
    if len(all_faces) == 0:
        print("❌ No training data found!")
        return
    
    print(f"\n📊 Total dataset: {len(all_faces)} samples")
    print(f"   - Real faces: {sum(all_labels)}")
    print(f"   - Fake faces: {len(all_labels) - sum(all_labels)}")
    
    # Balance dataset
    all_faces_balanced, all_labels_balanced = trainer.balance_dataset(all_faces, all_labels)
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        all_faces_balanced, all_labels_balanced, 
        test_size=0.2, random_state=42, stratify=all_labels_balanced
    )
    
    print(f"\n📊 Final split:")
    print(f"   Training: {len(X_train)} samples")
    print(f"   Testing: {len(X_test)} samples")
    
    # Extract HOG features
    print("\n🔧 Feature extraction...")
    X_train_hog = trainer.extract_hog_features(X_train)
    X_test_hog = trainer.extract_hog_features(X_test)
    
    # Train SVM
    model, scaler = trainer.train_svm(X_train_hog, y_train)
    
    # Evaluate model
    accuracy, report = trainer.evaluate_model(model, scaler, X_test_hog, y_test)
    
    # Save model
    metadata = {
        'yolo_model': args.yolo_model,
        'total_samples': len(all_faces_balanced),
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'augmentation_used': not args.no_augmentation,
        'balanced_dataset': True
    }
    
    model_path = trainer.save_model(model, scaler, accuracy, metadata)
    
    print(f"\n🎉 Improved training completed!")
    print(f"📁 Model saved to: {model_path}")
    print(f"🎯 Final accuracy: {accuracy:.4f}")
    
    if accuracy < 0.7:
        print("\n⚠️  Accuracy is still low. Consider:")
        print("   1. Getting more diverse training data")
        print("   2. Checking face detection quality manually")
        print("   3. Using different feature extraction methods")
        print("   4. Training a custom YOLO model for your specific dataset")


if __name__ == "__main__":
    main()
