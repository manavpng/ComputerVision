import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def order_points(pts):
    rect = np.zeros((4, 2), dtype = "float32")
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


x1 = np.empty((2,4)) 
from tkinter import *
from tkinter import messagebox
from PIL import ImageTk, Image
from tkinter import filedialog

def openfn():  #Function to get the file name(location)
    filename = filedialog.askopenfilename(title='open')
    return filename   
image = mpimg.imread(openfn())
x2 = np.zeros((x1.shape))
print("x2=")
print(x2)
p = [] 
for k in range(0,4):
    print(k)
    plt.imshow(image)
    p1 = plt.ginput(n=1,show_clicks = True)
    i1 = [i[0] for i in p1]
    i2 = [i[1] for i in p1]
    p +=[i1[0], i2[0]]
    p = np.asarray(p)
    x2[0,k]= i1[0]
    x2[1,k]= i2[0]
print("Selected coordinates are=")

x2=np.transpose(x2)
print(x2)
warped = four_point_transform(image, x2)

j = Image.fromarray(warped, mode='RGB')
j.save('Rectified_Image.jpg')
cv2.imshow("Rectified Image", warped)
cv2.waitKey(0)