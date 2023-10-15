import cv2
import matplotlib.pyplot as plt
import numpy as np
from srad import *

#from numpy import float16
# import os

#1\Read the image
Image = cv2.imread('3.bmp',1)
image = cv2.cvtColor(Image,cv2.COLOR_BGR2GRAY)
img   = np.array(image, np.float)

img_after = srad(img,Iterations = 200)
plt.subplot(1,2,1)
plt.imshow(img,cmap = 'gray')
plt.subplot(1,2,2)
plt.imshow(img_after,cmap = 'gray')
plt.show()



























# def Gradient_r(img):
    # grx = np.zeros(img.shape,dtype = np.float)
    # gry = np.zeros(img.shape,dtype = np.float)
    # for i in range(img.shape[0]):
        # if i < (img.shape[0]-1):
            # gry[i] = img[i+1]-img[i]
        # else:
            # gry[i] = 0
    # for i in range(img.shape[1]):
        # if i < (img.shape[1]-1):
            # grx[:,i] = img[:,i+1]-img[:,i]
        # else:
            # grx[:,i] = 0
    # return gry,grx

# def Gradient_l(img):
    # glx = np.zeros(img.shape,dtype = np.float)
    # gly = np.zeros(img.shape,dtype = np.float)
    # for i in range(img.shape[0]):
        # if i == 0:
            # gly[i] = 0
        # else:
            # gly[i] = img[i]-img[i-1]
    # for i in range(img.shape[1]):
        # if i == 0:
            # glx[:,i] = 0;
        # else:
            # glx[:,i] = img[:,i]-img[:,i-1]
    # return gly,glx        
