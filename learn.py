from picamera2 import Picamera2
import cv2
import torch
import time

# Initialize Picamera2
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'BGR888', "size": (320, 240)}))
picam2.start()
time.sleep(1)  # Let the camera warm up

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Object class names
classNames = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
    'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'handbag', 'tie',
    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'N/A', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush' ]

while True:
    # Capture image
    img = picam2.capture_array()

    # Inference
    results = model(img, size=320)
    print("Number of Detections:", len(results.pred[0]))

    for r in results.pred[0]:
        x1, y1, x2, y2 = int(r[0]), int(r[1]), int(r[2]), int(r[3])
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

        confidence = round(r[4].item(), 2)
        cls = int(r[5].item())
        class_name = classNames[cls]

        print(f"Class: {class_name}, Confidence: {confidence}")
        cv2.putText(img, class_name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Camera', img)
    key = cv2.waitKey(1)

    if key == ord('q'):
        break
    elif key == ord('c'):
        cv2.destroyAllWindows()
        break

cv2.destroyAllWindows()
