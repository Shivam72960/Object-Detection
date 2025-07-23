# YOLOv3 Object Detection Web App

A modern web application for object detection using YOLOv3 with a glassmorphism UI design.

![Demo](https://via.placeholder.com/600x300?text=YOLOv3+Object+Detection)

## ✨ Features

- **Modern UI** - Glassmorphism design with smooth animations
- **Object Detection** - Powered by YOLOv3 deep learning model
- **Real-time Processing** - Upload images and get instant results
- **Responsive Design** - Works on desktop and mobile devices

## 🚀 Quick Start

1. **Install dependencies:**
```bash
pip install opencv-python-headless flask numpy requests
```

2. **Download YOLOv3 model files:**
```bash
mkdir yolov3
curl -L "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov3.weights" -o yolov3/yolov3.weights
curl "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg" -o yolov3/yolov3.cfg
curl "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names" -o yolov3/coco.names
```

3. **Run the application:**
```bash
python app.py
```

4. **Open browser:** `http://localhost:5000`

## 📁 Project Structure

```
├── app.py              # Flask backend
├── index.html          # Modern frontend
├── yolov3/            # Model files
├── uploads/           # User images
└── processed/         # Results
```

## 🎯 Usage

1. Upload an image using the file picker
2. Click "Detect Objects" 
3. View results with bounding boxes and detection list

## 🔌 API

**POST** `/detect`
- Upload image file
- Returns JSON with original/processed images and detections

## 🛠️ Tech Stack

- **Backend:** Flask, OpenCV, NumPy
- **Frontend:** HTML5, CSS3, JavaScript
- **AI Model:** YOLOv3
- **Design:** Glassmorphism, CSS Grid, Animations

## 📄 License

MIT License - feel free to use and modify!

---

⭐ **Star this repo if you find it useful!**