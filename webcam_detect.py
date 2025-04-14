
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
if not cap.isOpened():
    print("Failed to open")
    exit()
    while True:
        ret, frame = cap.read()
        if not ret:
            print ("Fail to get frame")
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

     


     
