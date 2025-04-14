from picamera2 import Picamera2
import cv2
import time

# Initialize Picamera2
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'BGR888', "size": (320, 240)}))
picam2.start()
time.sleep(1)  # Let the camera warm up

# Load YOLOv3-Tiny model using OpenCV DNN
net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Load class labels from coco.names file
with open("coco.names", "r") as f:
    classNames = [line.strip() for line in f.readlines()]

# Function to get bounding box and class label
def process_frame(frame):
    # Get the height and width of the frame
    height, width = frame.shape[:2]

    # Convert the frame to a blob to feed into the network
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (320, 320), swapRB=True, crop=False)
    net.setInput(blob)
    
    # Get the output layer names
    output_layer_names = net.getUnconnectedOutLayersNames()
    
    # Perform inference and get output
    outputs = net.forward(output_layer_names)

    # Process the results (detections)
    boxes, confidences, class_ids = [], [], []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = scores.argmax()
            confidence = scores[class_id]
            if confidence > 0.5:  # Confidence threshold
                center_x, center_y, width, height = map(int, detection[:4] * [width, height, width, height])
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)
                boxes.append([x, y, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    return boxes, confidences, class_ids

while True:
    # Capture image from the camera
    frame = picam2.capture_array()

    # Process the frame for object detection
    boxes, confidences, class_ids = process_frame(frame)

    # Apply Non-Maximum Suppression (NMS)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes and class labels
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = classNames[class_ids[i]]
            confidence = confidences[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}: {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with detections
    cv2.imshow('Camera', frame)

    # Press 'q' to exit, 'c' to close
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('c'):
        cv2.destroyAllWindows()
        break

cv2.destroyAllWindows()
