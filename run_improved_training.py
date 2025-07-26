#!/usr/bin/env python3
"""
Run improved anti-spoofing training
"""

import subprocess
import sys
from pathlib import Path

def main():
    print("🚀 Starting Improved Anti-Spoofing Training")
    print("=" * 50)
    
    # Check if required files exist
    yolo_model = Path("face_detection_model/yolov5s-face.onnx")
    dataset = Path("Face Anti-Spoofing.v4i.yolov11")
    
    if not yolo_model.exists():
        print(f"❌ YOLO model not found: {yolo_model}")
        return
    
    if not dataset.exists():
        print(f"❌ Dataset not found: {dataset}")
        return
    
    print("✅ Required files found")
    
    # Run improved training
    cmd = [
        sys.executable, "improved_training.py",
        "--dataset", str(dataset),
        "--yolo-model", str(yolo_model),
        "--image-size", "64", "64"
    ]
    
    print(f"🏃 Running: {' '.join(cmd)}")
    print("\n🔧 Improvements in this version:")
    print("   • Lower confidence threshold (0.3 vs 0.5)")
    print("   • Data augmentation (flip, brightness, rotation)")
    print("   • Dataset balancing (upsample minority class)")
    print("   • Better HOG parameters (12 orientations, 6x6 cells)")
    print("   • Comprehensive SVM parameter search")
    print("   • Combined train+validation for larger dataset")
    print("   • Better face detection validation")
    
    try:
        subprocess.run(cmd, check=True)
        print("\n🎉 Improved training completed!")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with error code: {e.returncode}")
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")

if __name__ == "__main__":
    main()
