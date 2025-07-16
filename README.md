
# NeonPoseMirror 🔮🧍‍♀️🧍‍♂️

**NeonPoseMirror** is a real-time AI-based 3D clone visualizer that mimics your body movements using webcam input. It leverages [MediaPipe Pose](https://developers.google.com/mediapipe) for landmark tracking and displays your mirrored pose in both 2D and 3D using OpenCV and Matplotlib — with a neon glow effect and animated trails.

## 🚀 Features
- Real-time full-body pose detection via webcam
- 2D neon effect overlay with joint trails
- 3D rotating clone visualization using Matplotlib
- Gradient pulse and scanning animation
- Fast, lightweight and GPU-free (CPU-based)

## 📷 Demo
> Include a screenshot or short GIF here. Save inside `assets/` folder.

## 🛠 Requirements
- Python **3.10.0**
- Webcam (laptop camera or USB)
- Installed libraries:
  - `mediapipe`
  - `opencv-python`
  - `numpy`
  - `matplotlib`

## 🔧 Installation
```bash
git clone https://github.com/yourusername/NeonPoseMirror.git
cd NeonPoseMirror
python -m pip install -r requirements.txt
```

## ▶️ Run the App
```bash
python main.py
```

## 🧠 How It Works
- MediaPipe detects 33 body keypoints.
- Your pose is mirrored horizontally and visualized in:
  - 2D (OpenCV window) with animated glowing joints and trails
  - 3D (Matplotlib plot) in a rotating pose

## ❓ Troubleshooting
- Make sure your full body is visible to the camera.
- Use model_complexity=2 for better accuracy.

## 📜 License
MIT License.
