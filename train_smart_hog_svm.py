#!/usr/bin/env python3
"""
Fast & Smart Face Anti-Spoofing Trainer with Progress Monitoring

This script provides intelligent parameter tuning with progress tracking
and timeouts to prevent hanging.

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
import threading
from datetime import datetime


class ProgressMonitor:
    """Progress monitoring for long-running operations"""
    def __init__(self, operation_name, estimated_time=None):
        self.operation_name = operation_name
        self.start_time = time.time()
        self.estimated_time = estimated_time
        self.running = True
        
    def start_monitoring(self):
        """Start the progress monitoring thread"""
        def monitor():
            while self.running:
                elapsed = time.time() - self.start_time
                if self.estimated_time:
                    progress = min(elapsed / self.estimated_time * 100, 95)
                    print(f"  ⏱️  {self.operation_name}: {elapsed:.0f}s elapsed, ~{progress:.0f}% done")
                else:
                    print(f"  ⏱️  {self.operation_name}: {elapsed:.0f}s elapsed...")
                time.sleep(30)  # Update every 30 seconds
        
        self.thread = threading.Thread(target=monitor, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop monitoring"""
        self.running = False
        elapsed = time.time() - self.start_time
        print(f"  ✅ {self.operation_name} completed in {elapsed:.1f}s")


class SmartFaceAntiSpoofingTrainer:
    def __init__(self, dataset_path, image_size=(64, 64), test_size=0.2, random_state=42, 
                 quick_mode=False, output_dir='models'):
        """Initialize the Smart Face Anti-Spoofing Trainer"""
        self.dataset_path = Path(dataset_path)
        self.image_size = image_size
        self.test_size = test_size
        self.random_state = random_state
        self.quick_mode = quick_mode
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create timestamp
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Smart HOG configurations based on mode
        if quick_mode:
            self.hog_configs = [
                {'name': 'standard', 'orientations': 9, 'pixels_per_cell': (8, 8), 
                 'cells_per_block': (2, 2), 'block_norm': 'L2-Hys'}
            ]
        else:
            self.hog_configs = [
                {'name': 'standard', 'orientations': 9, 'pixels_per_cell': (8, 8), 
                 'cells_per_block': (2, 2), 'block_norm': 'L2-Hys'},
                {'name': 'fine', 'orientations': 12, 'pixels_per_cell': (4, 4), 
                 'cells_per_block': (2, 2), 'block_norm': 'L2-Hys'}
            ]
        
        # Model components
        self.best_hog_config = self.hog_configs[0]
        self.best_model = None
        self.scaler = StandardScaler()
        self.class_names = ['fake', 'real']
        self.class_mapping = {0: 'fake', 1: 'real'}
        
        print(f"🚀 Smart Face Anti-Spoofing Trainer Initialized")
        print(f"📁 Dataset: {self.dataset_path}")
        print(f"🖼️  Image size: {self.image_size}")
        print(f"⚡ Quick mode: {self.quick_mode}")
        print(f"💾 Output: {self.output_dir}")

    def parse_yolo_label(self, label_path):
        """Parse YOLO format label file"""
        if not os.path.exists(label_path):
            return None
        with open(label_path, 'r') as f:
            line = f.readline().strip()
            if line:
                parts = line.split()
                class_id = int(parts[0])
                bbox = [float(x) for x in parts[1:5]]
                return class_id, bbox
        return None

    def yolo_to_pixel_coords(self, bbox, img_width, img_height):
        """Convert YOLO normalized coordinates to pixel coordinates"""
        x_center, y_center, width, height = bbox
        x_center_px = x_center * img_width
        y_center_px = y_center * img_height
        width_px = width * img_width
        height_px = height * img_height
        
        x1 = int(x_center_px - width_px / 2)
        y1 = int(y_center_px - height_px / 2)
        x2 = int(x_center_px + width_px / 2)
        y2 = int(y_center_px + height_px / 2)
        
        return x1, y1, x2, y2

    def crop_face_from_image(self, image_path, label_path):
        """Crop face region from image using YOLO label"""
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                return None
                
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            img_height, img_width = gray.shape
            
            label_info = self.parse_yolo_label(label_path)
            if label_info is None:
                return None
                
            class_id, bbox = label_info
            x1, y1, x2, y2 = self.yolo_to_pixel_coords(bbox, img_width, img_height)
            
            # Ensure coordinates are within bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_width, x2), min(img_height, y2)
            
            face_crop = gray[y1:y2, x1:x2]
            
            if face_crop.size > 0:
                face_resized = cv2.resize(face_crop, self.image_size)
                return face_resized, class_id
            return None
                
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return None

    def extract_hog_features(self, image, config):
        """Extract HOG features using specified configuration"""
        image_normalized = exposure.equalize_hist(image)
        
        hog_params = {
            'orientations': config['orientations'],
            'pixels_per_cell': config['pixels_per_cell'],
            'cells_per_block': config['cells_per_block'],
            'block_norm': config['block_norm'],
            'visualize': False,
            'feature_vector': True
        }
        
        features = hog(image_normalized, **hog_params)
        return features

    def load_dataset(self):
        """Load and process the dataset"""
        print("📂 Loading dataset...")
        
        images = []
        labels = []
        
        for split in ['train', 'valid', 'test']:
            split_path = self.dataset_path / split
            images_path = split_path / 'images'
            labels_path = split_path / 'labels'
            
            if not images_path.exists():
                continue
                
            print(f"  Processing {split} set...")
            image_files = list(images_path.glob('*.jpg')) + list(images_path.glob('*.png'))
            
            for img_file in tqdm(image_files, desc=f"Loading {split}"):
                label_file = labels_path / (img_file.stem + '.txt')
                result = self.crop_face_from_image(img_file, label_file)
                
                if result is not None:
                    face_crop, class_id = result
                    images.append(face_crop)
                    labels.append(class_id)
        
        print(f"✅ Loaded {len(images)} samples")
        print(f"📊 Class distribution: {np.bincount(labels)}")
        
        return images, np.array(labels)

    def test_hog_configurations(self, images, labels):
        """Test HOG configurations to find the best one"""
        print("🔬 Testing HOG configurations...")
        
        best_score = 0
        best_config = self.hog_configs[0]
        results = []
        
        # Use subset for quick evaluation
        n_samples = min(800, len(images)) if not self.quick_mode else min(400, len(images))
        indices = np.random.choice(len(images), n_samples, replace=False)
        test_images = [images[i] for i in indices]
        test_labels = labels[indices]
        
        for config in self.hog_configs:
            print(f"  Testing {config['name']} configuration...")
            
            # Extract features
            features = []
            for img in tqdm(test_images, desc=f"HOG-{config['name']}", leave=False):
                feat = self.extract_hog_features(img, config)
                features.append(feat)
            features = np.array(features)
            
            # Quick evaluation
            X_train, X_test, y_train, y_test = train_test_split(
                features, test_labels, test_size=0.3, random_state=self.random_state, 
                stratify=test_labels
            )
            
            # Scale and train
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            svm = SVC(kernel='linear', C=1.0, random_state=self.random_state)
            svm.fit(X_train_scaled, y_train)
            
            score = svm.score(X_test_scaled, y_test)
            
            results.append({
                'config': config,
                'score': score,
                'feature_dim': features.shape[1]
            })
            
            print(f"    Accuracy: {score:.4f}, Features: {features.shape[1]}")
            
            if score > best_score:
                best_score = score
                best_config = config
                print(f"    🎯 New best configuration!")
        
        self.best_hog_config = best_config
        print(f"\n🏆 Best HOG: {best_config['name']} (score: {best_score:.4f})")
        
        return results

    def optimize_svm_parameters(self, X_train, y_train):
        """Optimize SVM parameters with comprehensive grid search for better accuracy"""
        print("🔧 Optimizing SVM parameters...")
        
        # Comprehensive parameter grids for better accuracy
        if self.quick_mode:
            # Quick mode: Still expanded but reasonable
            param_grids = [
                {
                    'kernel': ['linear'],
                    'C': [0.01, 0.1, 1, 10, 100],
                    'class_weight': [None, 'balanced']
                },
                {
                    'kernel': ['rbf'],
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                    'class_weight': [None, 'balanced']
                }
            ]
            cv_folds = 3
        else:
            # Comprehensive mode: Extended parameter search for maximum accuracy
            param_grids = [
                # Linear SVM with extensive C values
                {
                    'kernel': ['linear'],
                    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                    'class_weight': [None, 'balanced'],
                    'max_iter': [5000]  # Ensure convergence
                },
                # RBF SVM with comprehensive gamma and C combinations
                {
                    'kernel': ['rbf'],
                    'C': [0.01, 0.1, 1, 10, 100, 1000],
                    'gamma': ['scale', 'auto', 0.0001, 0.001, 0.01, 0.1, 1.0],
                    'class_weight': [None, 'balanced'],
                    'max_iter': [5000]
                },
                # Polynomial SVM for complex decision boundaries
                {
                    'kernel': ['poly'],
                    'C': [0.1, 1, 10, 100],
                    'degree': [2, 3, 4],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                    'coef0': [0.0, 0.1, 1.0],
                    'class_weight': [None, 'balanced'],
                    'max_iter': [5000]
                },
                # Sigmoid SVM (neural network-like)
                {
                    'kernel': ['sigmoid'],
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                    'coef0': [0.0, 0.1, 1.0],
                    'class_weight': [None, 'balanced'],
                    'max_iter': [5000]
                }
            ]
            cv_folds = 5  # More thorough cross-validation        
        # Calculate estimated time - more sophisticated calculation
        total_combinations = 0
        for grid in param_grids:
            combinations = 1
            for param_name, param_values in grid.items():
                if param_name != 'max_iter':  # Don't count max_iter in combinations
                    combinations *= len(param_values)
            total_combinations += combinations
        
        total_fits = total_combinations * cv_folds
        
        # More realistic time estimation based on kernel complexity
        if self.quick_mode:
            estimated_time = total_fits * 2  # ~2 seconds per fit in quick mode
        else:
            estimated_time = total_fits * 4  # ~4 seconds per fit in comprehensive mode
        
        print(f"  📊 Parameter combinations: {total_combinations}")
        print(f"  🔄 Total CV fits: {total_fits}")
        print(f"  ⏱️  Estimated time: {estimated_time:.0f}s ({estimated_time/60:.1f} min)")
        
        if not self.quick_mode and estimated_time > 1800:  # More than 30 minutes
            print(f"  ⚠️  This will take a while! Consider using --quick for faster results")
        
        # Setup parallel processing - more conservative for stability
        cpu_count = os.cpu_count()
        if self.quick_mode:
            n_jobs = min(2, cpu_count if cpu_count is not None else 1)
        else:
            n_jobs = min(3, cpu_count if cpu_count is not None else 1)  # More cores for comprehensive mode
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)        
        best_score = 0
        best_params = None
        best_model = None
        all_results = []
        
        for i, param_grid in enumerate(param_grids):
            kernel_name = param_grid['kernel'][0]
            
            # Calculate combinations for this specific grid
            grid_combinations = 1
            for param_name, param_values in param_grid.items():
                if param_name != 'max_iter':
                    grid_combinations *= len(param_values)
            
            print(f"\n  🧪 Testing {kernel_name} kernel ({i+1}/{len(param_grids)})")
            print(f"      Combinations: {grid_combinations}")
            
            svm = SVC(random_state=self.random_state, probability=True)
            
            # Use different verbosity based on grid size
            verbose_level = 1 if grid_combinations > 50 else 2
            
            grid_search = GridSearchCV(
                svm, param_grid, cv=cv, scoring='accuracy',
                n_jobs=n_jobs, verbose=verbose_level,
                return_train_score=True
            )
            
            # Start progress monitoring
            estimated_grid_time = grid_combinations * cv_folds * (4 if not self.quick_mode else 2)
            monitor = ProgressMonitor(f"{kernel_name} kernel search", estimated_grid_time)
            monitor.start_monitoring()
            
            start_time = time.time()
            print(f"    🚀 Starting at {time.strftime('%H:%M:%S')}")
            
            try:
                grid_search.fit(X_train_scaled, y_train)
                search_time = time.time() - start_time
                monitor.stop()
                
                print(f"    ✅ Best score: {grid_search.best_score_:.4f}")
                print(f"    ⚙️  Best params: {grid_search.best_params_}")
                print(f"    ⏱️  Time: {search_time:.1f}s")
                
                # Store results for this kernel
                all_results.append({
                    'kernel': kernel_name,
                    'best_score': grid_search.best_score_,
                    'best_params': grid_search.best_params_,
                    'search_time': search_time,
                    'combinations_tested': grid_combinations
                })
                
                if grid_search.best_score_ > best_score:
                    best_score = grid_search.best_score_
                    best_params = grid_search.best_params_
                    best_model = grid_search.best_estimator_
                    print(f"    🎯 New best model! (improvement: +{grid_search.best_score_ - best_score:.4f})")
                else:
                    print(f"    📊 Score: {grid_search.best_score_:.4f} (current best: {best_score:.4f})")
                
            except Exception as e:
                monitor.stop()
                search_time = time.time() - start_time
                print(f"    ❌ Error: {str(e)}")
                print(f"    ⏱️  Failed after: {search_time:.1f}s")
                
                # Store failed attempt
                all_results.append({
                    'kernel': kernel_name,
                    'best_score': 0.0,
                    'best_params': None,
                    'search_time': search_time,
                    'error': str(e),
                    'combinations_tested': grid_combinations
                })
                continue        
        # Fallback if no model found
        if best_model is None:
            print(f"\n  🔄 Using fallback linear SVM...")
            best_model = SVC(kernel='linear', C=1.0, probability=True, random_state=self.random_state)
            best_model.fit(X_train_scaled, y_train)
            best_score = 0.0
            best_params = {'kernel': 'linear', 'C': 1.0}
        
        self.best_model = best_model
        
        print(f"\n🏆 BEST SVM CONFIGURATION:")
        print(f"   Score: {best_score:.4f}")
        print(f"   Params: {best_params}")
        
        # Print summary of all tested kernels
        print(f"\n📊 KERNEL PERFORMANCE SUMMARY:")
        for result in all_results:
            if 'error' not in result:
                print(f"   {result['kernel']:>8}: {result['best_score']:.4f} "
                      f"({result['combinations_tested']} combinations, {result['search_time']:.1f}s)")
            else:
                print(f"   {result['kernel']:>8}: FAILED ({result['error'][:50]}...)")
        
        total_search_time = sum(r['search_time'] for r in all_results)
        
        return {
            'best_score': best_score,
            'best_params': best_params,
            'search_time': total_search_time,
            'all_kernel_results': all_results,
            'total_combinations_tested': sum(r['combinations_tested'] for r in all_results)
        }

    def evaluate_model(self, X_test, y_test):
        """Evaluate the final model"""
        if self.best_model is None:
            print("❌ No model to evaluate!")
            return None
            
        print("📊 Evaluating final model...")
        
        X_test_scaled = self.scaler.transform(X_test)
        
        print("  🔄 Making predictions...")
        y_pred = self.best_model.predict(X_test_scaled)
        y_pred_proba = self.best_model.predict_proba(X_test_scaled)
        
        print("  📈 Calculating metrics...")
        accuracy = accuracy_score(y_test, y_pred)
        
        try:
            auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])
        except:
            auc_score = None
        
        print(f"\n📈 FINAL RESULTS:")
        print(f"  Accuracy: {accuracy:.4f}")
        if auc_score:
            print(f"  AUC Score: {auc_score:.4f}")
        
        print(f"\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=self.class_names))
        
        # Confusion Matrix
        print("  📊 Generating confusion matrix...")
        cm = confusion_matrix(y_test, y_pred)
        self.plot_confusion_matrix(cm, accuracy)
        
        return {
            'accuracy': accuracy,
            'auc_score': auc_score,
            'predictions': y_pred,
            'probabilities': y_pred_proba,            'confusion_matrix': cm
        }

    def plot_confusion_matrix(self, cm, accuracy):
        """Plot and save confusion matrix"""
        print("📊 Creating confusion matrix visualization...")
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title(f'Confusion Matrix - Accuracy: {accuracy:.4f}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        cm_path = self.output_dir / f'confusion_matrix_{self.timestamp}_acc_{accuracy:.4f}.png'
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        
        # Don't show plot in automated training - just save it
        plt.close()  # Close the figure to free memory
        
        print(f"💾 Confusion matrix saved: {cm_path}")

    def save_model(self, accuracy):
        """Save the optimized model"""
        model_filename = f"smart_antispoofing_model_{self.timestamp}_acc_{accuracy:.4f}.pkl"
        model_path = self.output_dir / model_filename
        
        # Prepare HOG parameters
        hog_params = {
            'orientations': self.best_hog_config['orientations'],
            'pixels_per_cell': self.best_hog_config['pixels_per_cell'],
            'cells_per_block': self.best_hog_config['cells_per_block'],
            'block_norm': self.best_hog_config['block_norm'],
            'visualize': False,
            'feature_vector': True
        }
        
        model_data = {
            'model': self.best_model,
            'scaler': self.scaler,
            'hog_params': hog_params,
            'hog_config': self.best_hog_config,
            'image_size': self.image_size,
            'class_mapping': self.class_mapping,
            'class_names': self.class_names,
            'training_timestamp': self.timestamp,
            'accuracy': accuracy,
            'version': 'smart_2.0'
        }
        
        joblib.dump(model_data, model_path, compress=3)
        print(f"💾 Model saved: {model_path}")
        
        return model_path

    def save_training_report(self, metrics, hog_results, svm_results):
        """Save training report"""
        report = {
            'training_info': {
                'timestamp': self.timestamp,
                'dataset_path': str(self.dataset_path),
                'image_size': self.image_size,
                'quick_mode': self.quick_mode,
                'test_size': self.test_size
            },
            'hog_optimization': {
                'best_config': self.best_hog_config,
                'all_results': hog_results
            },
            'svm_optimization': svm_results,
            'final_performance': {
                'accuracy': float(metrics['accuracy']),
                'auc_score': float(metrics['auc_score']) if metrics['auc_score'] else None,
                'confusion_matrix': metrics['confusion_matrix'].tolist()
            }
        }
        
        # Save JSON report
        report_path = self.output_dir / f'training_report_{self.timestamp}.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save text summary
        summary_path = self.output_dir / f'training_summary_{self.timestamp}.txt'
        with open(summary_path, 'w') as f:
            f.write(f"Smart Face Anti-Spoofing Training Summary\n")
            f.write(f"========================================\n\n")
            f.write(f"Timestamp: {self.timestamp}\n")
            f.write(f"Dataset: {self.dataset_path}\n")
            f.write(f"Quick Mode: {self.quick_mode}\n\n")
            f.write(f"Best HOG Config: {self.best_hog_config['name']}\n")
            f.write(f"Best SVM Params: {svm_results.get('best_params', 'N/A')}\n\n")
            f.write(f"Final Accuracy: {metrics['accuracy']:.4f}\n")
            if metrics['auc_score']:
                f.write(f"AUC Score: {metrics['auc_score']:.4f}\n")
        
        print(f"📋 Report saved: {report_path}")
        print(f"📄 Summary saved: {summary_path}")
        
        return report_path

    def run_training_pipeline(self):
        """Run the complete smart training pipeline"""
        print("🚀 STARTING SMART TRAINING PIPELINE")
        print("=" * 50)
        
        pipeline_start = time.time()
        
        # Step 1: Load dataset
        print("\n📂 STEP 1: LOADING DATASET")
        images, labels = self.load_dataset()
        
        if len(images) == 0:
            print("❌ No data loaded!")
            return None
        
        # Step 2: HOG optimization
        print("\n🔬 STEP 2: HOG OPTIMIZATION")
        hog_results = self.test_hog_configurations(images, labels)
        
        # Step 3: Feature extraction
        print(f"\n🎯 STEP 3: FEATURE EXTRACTION")
        print(f"Using {self.best_hog_config['name']} HOG configuration...")
        
        features = []
        for img in tqdm(images, desc="Extracting features"):
            feat = self.extract_hog_features(img, self.best_hog_config)
            features.append(feat)
        features = np.array(features)
        
        print(f"✅ Extracted {features.shape[1]} features per sample")
        
        # Step 4: Dataset splitting
        print("\n🔄 STEP 4: DATASET SPLITTING")
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=self.test_size, 
            random_state=self.random_state, stratify=labels
        )
        
        print(f"  Training: {len(X_train)} samples")
        print(f"  Testing: {len(X_test)} samples")
        
        # Step 5: SVM optimization
        print("\n🔧 STEP 5: SVM OPTIMIZATION")
        svm_results = self.optimize_svm_parameters(X_train, y_train)
        
        # Step 6: Final evaluation
        print("\n📊 STEP 6: FINAL EVALUATION")
        metrics = self.evaluate_model(X_test, y_test)
        
        if metrics is None:
            print("❌ Evaluation failed!")
            return None
        
        # Step 7: Save everything
        print("\n💾 STEP 7: SAVING RESULTS")
        model_path = self.save_model(metrics['accuracy'])
        report_path = self.save_training_report(metrics, hog_results, svm_results)
        
        total_time = time.time() - pipeline_start
        
        print(f"\n🎉 TRAINING COMPLETED!")
        print("=" * 50)
        print(f"⏱️  Total time: {total_time:.2f}s ({total_time/60:.1f} min)")
        print(f"🎯 Final accuracy: {metrics['accuracy']:.4f}")
        print(f"💾 Model saved: {model_path}")
        print(f"📋 Report saved: {report_path}")
        print("=" * 50)
        
        return {
            'metrics': metrics,
            'model_path': model_path,
            'report_path': report_path,
            'training_time': total_time
        }


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Smart Face Anti-Spoofing Training with Progress Monitoring')
    parser.add_argument('--dataset', type=str, 
                       default='Face Anti-Spoofing.v4i.yolov11',
                       help='Path to the dataset folder')
    parser.add_argument('--image_size', type=int, nargs=2, default=[64, 64],
                       help='Target image size (width height)')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Proportion of dataset for testing')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random state for reproducibility')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode: faster training with minimal parameters')
    parser.add_argument('--output_dir', type=str, default='models',
                       help='Directory to save models and results')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = SmartFaceAntiSpoofingTrainer(
        dataset_path=args.dataset,
        image_size=tuple(args.image_size),
        test_size=args.test_size,
        random_state=args.random_state,
        quick_mode=args.quick,
        output_dir=args.output_dir
    )
    
    # Run training
    print(f"\n🎯 Training Mode: {'Quick' if args.quick else 'Comprehensive'}")
    print(f"📊 This will find the best HOG + SVM combination for face anti-spoofing")
    print(f"⏱️  Progress updates will be shown every 30 seconds during long operations")
    
    results = trainer.run_training_pipeline()
    
    if results:
        print(f"\n✨ SUCCESS!")
        print(f"   🎯 Best accuracy: {results['metrics']['accuracy']:.4f}")
        print(f"   💾 Model: {results['model_path']}")
        print(f"   📋 Report: {results['report_path']}")
        print(f"\n🚀 You can now use the model for inference!")
        print(f"   Example: python inference_enhanced.py --model {results['model_path']} --image test_image.jpg")
    else:
        print("\n❌ Training failed!")


if __name__ == "__main__":
    main()
