import os
import cv2
import numpy as np
from flask import Flask, request, render_template, send_file, jsonify
import requests
app = Flask(__name__)
# Directory for uploaded images and processed images
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
YOLO_FOLDER = 'yolov3'
# Create folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(YOLO_FOLDER, exist_ok=True)
# Download YOLOv3 files if not present
def download_yolo_files():
    yolo_weights_url = "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov3.weights"
    cfg_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg"
    coco_names_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
    weights_path = os.path.join(YOLO_FOLDER, "yolov3.weights")
    cfg_path = os.path.join(YOLO_FOLDER, "yolov3.cfg")
    names_path = os.path.join(YOLO_FOLDER, "coco.names")
    if not os.path.exists(weights_path):
        print("Downloading weights...")
        download_file(yolo_weights_url, weights_path)
    if not os.path.exists(cfg_path):
        print("Downloading config...")
        download_file(cfg_url, cfg_path)
    if not os.path.exists(names_path):
        print("Downloading class names...")
        download_file(coco_names_url, names_path)
def download_file(url, filename):
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers, stream=True)
    response.raise_for_status()
    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
# Call the function to download YOLO files when the app starts
download_yolo_files()
# Load the YOLO model and class names
weights_path = os.path.join(YOLO_FOLDER, "yolov3.weights")
cfg_path = os.path.join(YOLO_FOLDER, "yolov3.cfg")
names_path = os.path.join(YOLO_FOLDER, "coco.names")
# Load the class names
with open(names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]
# Load the YOLO model
net = cv2.dnn.readNet(weights_path, cfg_path)
layer_names = net.getLayerNames()
try:
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
except:
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/detect', methods=['POST'])
def detect_objects():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify(error="No file part"), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify(error="No selected file"), 400
    # Save the uploaded file
    filename = file.filename
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)
    # Read the image
    image = cv2.imread(file_path)
    if image is None:
        return jsonify(error="Could not read image"), 400
    height, width, channels = image.shape
    # Prepare the image blob
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    # Process the detections
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    # Apply non-max suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    # Draw the detections and collect explanations
    explanations = []
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = classes[class_ids[i]]
            confidence = confidences[i]
            color = (0, 255, 0)
            cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
            cv2.putText(image, f"{label} {confidence:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            explanations.append(f"{label} ({confidence:.2f})")
    # Save the processed image
    processed_filename = f"processed_{filename}"
    processed_path = os.path.join(PROCESSED_FOLDER, processed_filename)
    cv2.imwrite(processed_path, image)
    # Return the result
    return jsonify(
        original_image=filename,
        processed_image=processed_filename,
        detections=explanations
    )
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_file(os.path.join(UPLOAD_FOLDER, filename))
@app.route('/processed/<filename>')
def processed_file(filename):
    return send_file(os.path.join(PROCESSED_FOLDER, filename))
if __name__ == '__main__':
    app.run(debug=True)