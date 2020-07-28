#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
from scipy import signal
from PIL import Image
import PIL
import itertools
from tqdm import tqdm
from tkinter import *
from tkinter import messagebox
from PIL import ImageTk, Image
from tkinter import filedialog

def openfn():  #Function to get the file name(location)
    filename = filedialog.askopenfilename(title='open')
    return filename

a = cv2.imread(openfn())
x = a
if len(a.shape) == 3:
    a = cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
    b = a.shape
else:
    b = a.shape



n = 3
n1 = math.ceil(n/2)
a = np.double(a)

#For edges 
hpf = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
lpf = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])

c = 0
h = 0
d = np.zeros((b[0]-2*n1,b[1]-2*n1))
g = np.zeros((b[0]-2*n1,b[1]-2*n1))

kh = float(input('Enter the value of Kh = '))
T  = int(input('Enter the value of threshold = ')) 


#For calculation of Ix
for i in tqdm(range(n1,b[0]-n1-2)):
    for j in range(n1,b[1]-n1-2):
        c = sum(np.multiply(a[i-n1:i+n1-1,j-n1:j+n1-1],lpf).reshape(n**2))
                
        d[i,j] = c
        c = 0
        
Ix = d

#For calculation of Iy
for i in tqdm(range(n1,b[0]-n1-2)):
    for j in range(n1,b[1]-n1-2):
        h = sum(np.multiply(a[i-n1:i+n1-1,j-n1:j+n1-1],hpf).reshape(n**2))
                
        g[i,j] = h
        h = 0

Iy = g



#Some extra calculations for matrix
Ix2 = np.multiply(Ix,Ix)
Iy2 = np.multiply(Iy,Iy)
Ixy = np.multiply(Ix,Iy)

b1 = Ix.shape


m = 5
m1 = math.ceil(m/2)

#Blank canvas for different features
cor = np.zeros((b[0]-2*m1,b[1]-2*m1))
Edg = np.zeros((b[0]-2*m1,b[1]-2*m1))
Fla = np.zeros((b[0]-2*m1,b[1]-2*m1))
R   = np.zeros((b[0]-2*m1,b[1]-2*m1))

Ix2c = np.array([0]*(m**2))
Iy2c = np.array([0]*(m**2))
Ixyc = np.array([0]*(m**2))


for i in tqdm(range(m1,b[0]-m1-3)):
    for j in range(m1,b[1]-m1-3):
        p = 0
        Ix2c = Ix2[i-m1:i+m1-1,j-m1:j+m1-1].reshape((m)**2)
        Iy2c = Iy2[i-m1:i+m1-1,j-m1:j+m1-1].reshape((m)**2)
        Ixyc = Ixy[i-m1:i+m1-1,j-m1:j+m1-1].reshape((m)**2)
        H = np.multiply((1/(m**2)),[[sum(Ix2c),sum(Ixyc)],[sum(Ixyc),sum(Iy2c)]])
                
        R[i,j] = np.linalg.det(H) - kh*(np.trace(H))**2
                
        #Conditions
        if R[i,j] > T:
            cor[i,j] = 1
        elif R[i,j] < -T*0.5:
            Edg[i,j] = 1
        else:
            Fla[i,j] = 1
                    
        p+=1

print('-----------------------------------------------------------')
cv2.imshow('Original', np.uint8(x))
cv2.imshow('Corners', cor)
cv2.imshow('Edges', Edg)
cv2.imshow('Flat regions', Fla)

cv2.waitKey(0)
cv2.destroyAllWindows()


# In[10]:




