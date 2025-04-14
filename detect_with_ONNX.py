from picamera2 import Picamera2
import onnxruntime as ort
import numpy as np
import cv2
import time

# Initialize camera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'BGR888', "size": (320, 320)}))
picam2.start()
time.sleep(1)

# Load ONNX model
session = ort.InferenceSession("yolov5s.onnx")
input_name = session.get_inputs()[0].name

def preprocess(img):
    img = cv2.resize(img, (320, 320))
    img = img[:, :, ::-1]  # BGR to RGB
    img = np.transpose(img, (2, 0, 1)) / 255.0  # Normalize
    img = np.expand_dims(img, axis=0).astype(np.float32)
    return img

while True:
    frame = picam2.capture_array()
    input_tensor = preprocess(frame)

    # Inference
    start = time.time()
    outputs = session.run(None, {input_name: input_tensor})
    print("Inference time:", round((time.time() - start) * 1000), "ms")

    # Parse output (simplified – use real NMS for production)
    preds = outputs[0][0]
    for det in preds:
        x1, y1, x2, y2, conf, cls = det[:6]
        if conf > 0.4:
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            label = f"{int(cls)}: {round(float(conf), 2)}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    cv2.imshow("ONNX YOLOv5", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
