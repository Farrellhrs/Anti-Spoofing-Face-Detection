@echo off
echo 🚀 Testing Live Webcam Detection with Improved Model
echo ==================================================

echo 📋 Using improved model with 93.59%% accuracy
echo 🎥 Starting webcam detection...

conda run --live-stream --name myenv python live_detection_v2.py --mode webcam --antispoofing_model "anti_spoofing_model\improved_model_20250726_143343_acc_0.9359.pkl" --save_video

echo ✅ Detection completed!
pause
