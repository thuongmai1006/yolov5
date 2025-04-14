from picamera2 import Picamera2
import torch 
import cv2 
import time 
picam2=Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format":'BGR888',"size":(640,480)}))
picam2.start()
time.sleep(2)
while True: 
	frame=picam2.capture_array()
	cv2.imshow("Camera", frame)
	if cv2.waitKey(1) & 0xFF ==ord('q'):
		break
model =torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
	print ("Error: cannot open camera")
	exit()
#while True: 
#	ret, frame=cap.read()
#	if not ret:
#		print ("can't read frame")
#		break
#		result =model(frame)
#		
#		annotated_frame=result.render()[0]
#		cv2.imshow('YOLO detection',annotated_frame)
#		if cv2.waitKey(1)& 0XFF ==ord('q'):
#			break
#cap.release()
cv2.destroyAllWindows()
		
