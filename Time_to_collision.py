from picamera2 import Picamera2
import cv2
import torch
import time
import serial

# Initialize serial connection to Arduino
arduino = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
time.sleep(2)  # Let Arduino reset

# Initialize Picamera2
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'BGR888', "size": (224, 160)}))  # Reduced resolution
picam2.start()
time.sleep(1)

# Load YOLOv5 model (you can also try 'yolov5n' for better speed)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Object class names
classNames = [...]  # (You can keep your full list here)

# Detection duration tracker
detection_start = None
total_detection_time = 0

while True:
    img = picam2.capture_array()

    # Measure inference time
    start_time = time.time()
    results = model(img, size=224)  # Match input size to camera resolution
    inference_time = time.time() - start_time
    print(f"Inference time: {inference_time:.3f} seconds")

    detections = results.pred[0]
    detected = False

    for r in detections:
        x1, y1, x2, y2 = int(r[0]), int(r[1]), int(r[2]), int(r[3])
        confidence = round(r[4].item(), 2)
        cls = int(r[5].item())
        class_name = classNames[cls]

        # Draw detection
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(img, f"{class_name} {confidence}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Target detection condition
        if class_name == 'person' and confidence > 0.5:
            detected = True

    # Track detection duration
    if detected:
        arduino.write(b'1')
        if detection_start is None:
            detection_start = time.time()
        else:
            duration = time.time() - detection_start
            print(f"Person detected for {duration:.2f} seconds")
    else:
        arduino.write(b'0')
        if detection_start is not None:
            total_detection_time += time.time() - detection_start
            print(f"Detection ended. Total time: {total_detection_time:.2f} seconds")
            detection_start = None

    # Show camera
    cv2.imshow('Camera', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Final detection time output
if detection_start:
    total_detection_time += time.time() - detection_start
print(f"Final total person detection time: {total_detection_time:.2f} seconds")

cv2.destroyAllWindows()
arduino.close()
