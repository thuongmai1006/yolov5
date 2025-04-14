import time 
from picamera2 import Picamera2

picam2=Picamera2()
print(picam2)
picam2.preview_configuration.main.size=(1280,1280)
picam2.preview_configuration.main.format="RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()
time.sleep(2)
picam2.capture_file("image.jpg")
picam2.stop()
print("Image captured")
