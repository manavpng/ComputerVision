import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import math
from tqdm import tqdm

a1  = np.mean(np.array(Image.open("scene1.row3.col2.ppm")),2)
a = np.mean(np.array(Image.open("scene1.row3.col3.ppm")),2)

n   = 5
ser = 100
b   = a.shape
out = np.zeros((a.shape[0],a.shape[1]))
for i in tqdm(range(a.shape[0]-n)):
    for j in range(a.shape[1]-n):
        sub = []
        box1 = a[i:i+n,j:j+n]
        for j1 in range(j,min(j+ser,b[1]-n)):
            box2 = a1[i:i+n,j1:j1+n]
            diff = (box2-box1).reshape(1,n**2)
            sub.append(np.sum(abs(diff)))
        if len(sub)!=0:
            temp     = np.argmin(np.array(sub)) 
            out[i,j] = np.argmin(np.array(sub))
        else:
            out[i,j] = temp
plt.figure()
plt.imshow(np.multiply(out/ser,255),cmap = 'gray')
