import cv2
import numpy as np
import time

load_from_disk = True
if load_from_disk:
    penval = np.load('penval.npy')

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

kernel = np.ones((5,5),np.uint8)

cv2.namedWindow('image',cv2.WINDOW_NORMAL)

canvas = None

x1 = 0
y1 = 0

#threshold for noise
noiseth = 500

#This threshold determines the amount of disruption in the background.
background_threshold = 600

#Threshold for the wiper, the size of the contour must be bigger than for us to clear the canvas
wiper_thresh = 20000

clear = False

pen_img = cv2.resize(cv2.imread('pen.png',1), (50, 50))
eraser_img = cv2.resize(cv2.imread('eraser.png',1), (50, 50))

backgroundobject = cv2.createBackgroundSubtractorMOG2(detectShadows = False)

switch = 'Pen'

last_switch = time.time()

while(1):
    _,frame = cap.read()
    frame = cv2.flip(frame,1)
    
    if canvas is None:
        canvas = np.zeros_like(frame)
    
    top_left = frame[0:50,0:50]
    fgmask = backgroundobject.apply(top_left)
    
    switch_thresh = np.sum(fgmask==255)
    
    #if disruption is greater than background threshold and there has been some time after the previous switch then we can change the marker type
    if switch_thresh>background_threshold and (time.time()-last_switch) > 1:
        last_switch = time.time()
        
        if switch == 'Pen':
            switch = 'Eraser'
        else:
            switch = 'Pen'
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    if load_from_disk:
        lower_range = penval[0]
        upper_range = penval[1]
    else:
        lower_range = np.array([26,80,147])
        upper_range = np.array([81,255,255])
    
    
    mask = cv2.inRange(hsv,lower_range,upper_range)
    
    mask = cv2.erode(mask,kernel,iterations = 1)
    mask = cv2.dilate(mask,kernel,iterations = 2)
    
    #Find contour in frame
    _,contours = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    #make sure there is a contour present and also make sure its size is bigger than noise threshold
    if contours and cv2.contourArea(max(contours,key = cv2.contourArea)) > noiseth:
        c = max(contours,key = cv2.contourArea)
        
        x2,y2,w,h = cv2.boundingRect(c)
        
        area = cv2.contourArea(c)
        
        if x1==0 and y1==0:
            x1 = x2
            y1 = y2
        else:
            if switch == 'Pen':
                #Draw
                canvas = cv2.line(canvas,(x1,y1),(x2,y2),[255,0,0],5)
            
            else:
                #Erase
                cv2.circle(canvas,(x2,y2),20,(0,0,0),-1)
        x1 = x2
        y1 = y2
        
        if area > wiper_thresh:
            cv2.putText(canvas,'Pen is too close, clearing canvas', (100,200),cv2.FONT_HERSHEY_SIMPLEX,2, (0,0,255), 5, cv2.LINE_AA)
            clear = True
    else:
        x1 = 0
        y1 = 0
        
    #merge the canvas and frame.
    frame = cv2.add(frame,canvas)
    
    #Switch the images depending upon what the marker status is
    if switch!='Pen':
        cv2.circle(frame, (x1, y1), 20, (255,255,255), -1)
        frame[0:50,0:50] = eraser_img
    else:
        frame[0: 50, 0: 50] = pen_img
    
    cv2.imshow('Trackbars',cv2.resize(frame,None,fx = 1,fy = 1))
    k = cv2.waitKey(1)
    if k == 27:
        break
    
    if clear ==True:
        time.sleep(1)
        canvas = None
        
        clear = False
        
cv2.destroyAllWindows()
cap.release()