import cv2
import torch

# Start webcam
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Set webcam resolution
width, height = 640, 480  # Adjust to a supported webcam resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Object classes
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
    success, img = cap.read()

    if not success:
        print("Error: Failed to read a frame from the webcam :{cap.get(cv2.CAP_PROP_POS_FRAMES)}")
        break

    # Perform inference using YOLOv5
    results = model(img, size=width)

    # Debugging: print the results to see if there are any detections
    print("Number of Detections:", len(results.pred[0]))

    # Coordinates
    for r in results.pred[0]:
        # Bounding box
        x1, y1, x2, y2 = int(r[0]), int(r[1]), int(r[2]), int(r[3])
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

        # Confidence
        confidence = r[4].item()
        confidence = round(confidence, 2)
        print("Confidence --->", confidence)

        # Class name
        cls = int(r[5].item())
        print("Class name -->", classNames[cls])

        # Object details
        org = (x1, y1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color = (255, 0, 0)
        thickness = 2

        cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

    cv2.imshow('Webcam', img)

    key = cv2.waitKey(1)
    if key == ord('q'):
        # Press 'q' to quit the loop and close the webcam
        break
    elif key == ord('c'):
        # Press 'c' to close the webcam without quitting the script
        cap.release()
        cv2.destroyAllWindows()
        break

cap.release()
cv2.destroyAllWindows()
