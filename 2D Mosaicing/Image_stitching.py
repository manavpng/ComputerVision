import cv2
import numpy as np
import matplotlib.pyplot as plt
from random import randrange
from tkinter import filedialog as fd

# Opens the Video file
filename = fd.askopenfilename()
cap= cv2.VideoCapture(filename)
i=1
images = []
rate = int(input('Please select the frame rate='))
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    if i%rate == 0:
#         frame = np.rot90(frame,3)
        frame = cv2.rotate(frame,cv2.ROTATE_90_CLOCKWISE)
        images.append(frame)
#         cv2.imwrite('C:/Users/MANVENDRA/Desktop/study material/Second Semester/Computer vision/Project/frames/'+str(i)+'.jpg',frame)
    i+=1
    
def mosaicing(img_,img):
    img1 = cv2.cvtColor(img_,cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SURF_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=5)

    # Apply ratio test
    good = []
    for m in matches:
        if m[0].distance < 0.5*m[1].distance:
            good.append(m)
    matches = np.asarray(good)
    if len(matches[:,0]) >= 4:
        src = np.float32([ kp1[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
        dst = np.float32([ kp2[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
        H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    else:
        raise AssertionError('Can’t find enough keypoints.')
    dst = cv2.warpPerspective(img_,H,(img.shape[1] + img_.shape[1], img.shape[0]))
    dst[0:img.shape[0], 0:img.shape[1]] = img
    for k in range(dst.shape[1]):
        if (dst[:,k]).any() == np.zeros((dst.shape[0],1,3)).any():
            dst = dst[:,0:k,:]
            break
    cv2.imwrite('output.jpg',dst)
    return dst

x = images[0]
for i in range(1,len(images)):
    x = mosaicing(images[i],x)

