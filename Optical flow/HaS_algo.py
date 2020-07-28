import numpy as np 
import cv2
import math
from tqdm import tqdm
from tkinter import *
from tkinter import messagebox
from PIL import ImageTk, Image
from tkinter import filedialog
import matplotlib.pyplot as plt

#Creating the test Images

im1 = cv2.rectangle(np.zeros((100,100)), (25,25), (50,50), (255,0,0), -1)
im2 = cv2.rectangle(np.zeros((100,100)), (25,27), (50,52), (255,0,0), -1)

#Initializing u and v frames
u = np.zeros(im1.shape)
v = np.zeros(im1.shape)
uav = 0
vav = 0

#value of lamda (weight)
lam = 4

#Creating masks to be used 
m1 = np.array([[-1,1],[-1,1]])
m2 = np.array([[-1,1],[1,1]])
m3 = np.array([[-1,-1],[-1,-1]])
m4 = np.array([[1,1],[1,1]])

#calculation of fx, fy and ft
fx = (cv2.filter2D(im1,-1,m1) + cv2.filter2D(im2,-1,m1))*0.5
fy = (cv2.filter2D(im1,-1,m2) + cv2.filter2D(im2,-1,m2))*0.5
ft = (cv2.filter2D(im1,-1,m3) + cv2.filter2D(im2,-1,m4))

#Main for loop
mask = np.array([[1/12,1/6,1/12],[1/6,-1,1/6],[1/12,1/6,1/12]])   #Laplacian mask

for _ in range(40):
    for i in range(1,im1.shape[0]-2):
        for j in range(1,im1.shape[1]-2):
            uav = np.sum(np.multiply(u[i-1:i+2,j-1:j+2],mask))
            vav = np.sum(np.multiply(v[i-1:i+2,j-1:j+2],mask))
            
            P   = fx[i,j]*uav + fy[i,j]*vav + ft[i,j] 
            D   = fx[i,j]**2 + fy[i,j]**2 +lam
            
            u[i,j] = uav - fx[i,j] * (P/D)
            v[i,j] = vav - fy[i,j] * (P/D)

figure=plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
plt.imshow(im1,cmap = 'gray')
plt.title('Image 1')
plt.subplot(2,2,2)
plt.imshow(im2,cmap = 'gray')
plt.title('Image 2')
plt.subplot(2,2,3)
plt.imshow(u,cmap = 'gray')
plt.title('u image')
plt.subplot(2,2,4)
plt.imshow(v,cmap = 'gray')
plt.title('v image')