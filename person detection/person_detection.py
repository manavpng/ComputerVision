import numpy as np
import cv2
import time

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,1280)

backgroundobject = cv2.createBackgroundSubtractorMOG2(detectShadows = False)
kernel = np.ones((5,5),np.uint8)
noiseth = 1000

while(1):
    _,frame1 = cap.read()
    frame1 = cv2.flip(frame1,1)
    
    time.sleep(0.05)
    
    _,frame2 = cap.read()
    frame2 = cv2.flip(frame2,1)
    
    frame = backgroundobject.apply(frame1)
    
    mask = cv2.erode(frame,kernel,iterations = 1)
    mask = cv2.dilate(mask,kernel,iterations = 2)
    
    _,contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    if contours and cv2.contourArea(max(contours,key = cv2.contourArea)) > noiseth:
        c = max(contours,key = cv2.contourArea)
        
        x,y,w,h = cv2.boundingRect(c)
        
        cv2.rectangle(frame2,(x,y),(x+w,y+h),(0,0,255),2)
    
    cv2.imshow('image',frame2)
    
    k = cv2.waitKey(1)
    if k==27:
        break
        
cv2.destroyAllWindows()
cap.release()