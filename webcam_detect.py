
Object Detection
Real-Time Object Detection with YOLOv5 using Python
Project Overview:
In this project, we will implement a real-time object detection system using YOLOv5 (You Only Look Once Version 5). YOLOv5 is a state-of-the-art deep learning model known for its fast and accurate object detection capabilities. We will use pre-trained models to detect common objects in images, videos, or from a live webcam feed.

Python 3.x
Libraries: OpenCV, PyTorch, Matplotlib, Numpy
YOLOv5 Model Files (can be downloaded from the official repository)
Requirements:
Explanation:
The code begins with importing essential libraries such as torch for using the YOLOv5 model, and opencv-python for image processing and display.
It loads a pre-trained YOLOv5 model (yolov5s) using PyTorch Hub. YOLOv5s is a smaller, faster version, suitable for real-time detection.
The detect_objects function performs inference on the given image and extracts the detected labels and coordinates.
The plot_boxes function draws bounding boxes around detected objects and labels them with the object name.
The real_time_detection function captures the video feed from the webcam and performs object detection on each frame in real-time. The frame is displayed with bounding boxes and object labels.
It runs the real-time object detection loop and exits when 'q' is pressed.
Speed: Highly optimized for real-time object detection.
Accuracy: Capable of detecting multiple objects with high precision.
Ease of Use: Pre-trained models are readily available.
Security Systems: Real-time monitoring to detect intrusions or unusual activity.
Autonomous Vehicles: Detecting objects like pedestrians, vehicles, traffic signs, etc.
Retail Analytics: Analyzing customer behavior in stores.
Applications:
Advantages of YOLOv5:
Running the Code:
Real-Time Detection:
Bounding Box Plotting:
Object Detection Function:
Load YOLOv5 Model:
Setup and Imports:
This project is a practical and exciting way to get started with deep learning, computer vision, and real-time applications using Python and YOLOv5.


# Project Title: Real-Time Object Detection with YOLOv5 using Python

# 1. Setup and Imports
# Install the necessary libraries (uncomment if needed)

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 2. Load YOLOv5 Model
# Load the pre-trained YOLOv5 model from the PyTorch Hub
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# 3. Function for Object Detection
def detect_objects(image):
    # Perform inference on the image using the YOLOv5 model
    results = model(image)
    # Extract detected objects
    labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    return labels, cord

# 4. Define a function to draw bounding boxes
def plot_boxes(results, frame):
    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]

    for i in range(n):
        row = cord[i]
        if row[4] >= 0.3:  # Confidence threshold
            x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(row[3] * y_shape)
            label = int(labels[i])
            bgr = (0, 255, 0)
            # Draw the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
            # Put label text
            cv2.putText(frame, model.names[label], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 2)
    return frame

# 5. Real-Time Detection using Webcam
def real_time_detection():
    cap = cv2.VideoCapture(0)  # Open webcam
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Object detection
        results = detect_objects(frame)
        frame = plot_boxes(results, frame)

        # Display the frame
        cv2.imshow("YOLOv5 Real-Time Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# 6. Running Real-Time Detection
print("Press 'q' to exit the real-time detection window.")
real_time_detection()

     
/usr/local/lib/python3.10/dist-packages/torch/hub.py:330: UserWarning: You are about to download and run code from an untrusted repository. In a future release, this won't be allowed. To add the repository to your trusted list, change the command to {calling_fn}(..., trust_repo=False) and a command prompt will appear asking for an explicit confirmation of trust, or load(..., trust_repo=True), which will assume that the prompt is to be answered with 'yes'. You can also use load(..., trust_repo='check') which will only prompt for confirmation if the repo is not already trusted. This will eventually be the default behaviour
  warnings.warn(
Downloading: "https://github.com/ultralytics/yolov5/zipball/master" to /root/.cache/torch/hub/master.zip
Creating new Ultralytics Settings v0.0.6 file ✅ 
View Ultralytics Settings with 'yolo settings' or at '/root/.config/Ultralytics/settings.json'
Update Settings with 'yolo settings key=value', i.e. 'yolo settings runs_dir=path/to/dir'. For help see https://docs.ultralytics.com/quickstart/#ultralytics-settings.
YOLOv5 🚀 2024-11-10 Python-3.10.12 torch-2.5.0+cu121 CPU

Downloading https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt to yolov5s.pt...
100%|██████████| 14.1M/14.1M [00:00<00:00, 64.0MB/s]

Fusing layers... 
YOLOv5s summary: 213 layers, 7225885 parameters, 0 gradients, 16.4 GFLOPs
Adding AutoShape... 
Press 'q' to exit the real-time detection window.


     
