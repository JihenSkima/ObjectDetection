from ultralytics import YOLO
import cv2

#load YOLO model 
model = YOLO("yolo8n.py")

#load video 
video_path = "/TrafficCars.mp4"
cap = cv2.videoCapture(video_path)

ret = True

#read frames 
while ret : 
    ret , frame = cap.read()
    
    #object detection and tracking 
    obj = model.track(frame,persist=True)
    #plotting results 
    frame_ = obj[0].plot()
    
    #Visualization
    cv2.imshow("frame", frame_,)
    if cv2.waitKey(25) == 27 :
        break
            