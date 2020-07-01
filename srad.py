import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math

def Gradient(img):
    """
    Gradients of one image with symmetric boundary conditons
    
    Parameters
    -------
    img ； ndarray
    
    Returns
    ------
    grx : ndarry
        one-order froward  difference in the direction of column(axis = 1)
    gry : ndarry
        one-order froward  difference in the direction of row   (axis = 0)
    glx : ndarry
        one-order backward difference in the direction of column(axis = 1)
    gly : ndarry
        one-order backward difference in the direction of row   (axis = 0)
    grc : ndarry
        self-defined difference function    
    """
    #img(i,j-1)
    img_right = np.roll(img,1,axis = 1)
    img_right[:,0] = img[:,0]
    #img(i,j+1)
    img_left  = np.roll(img,-1,axis = 1)
    img_left[:,-1] = img[:,-1]
    #img(i+1,j)
    img_up = np.roll(img,-1,axis = 0)
    img_up[-1] = img[-1]
    #img(i-1,j)
    img_down = np.roll(img,1,axis = 0)
    img_down[0] = img[0]
    
    #img(i,j+1) - img(i,j)
    grx = img_left - img 
    #img(i+1,j) - img(i,j)
    gry = img_up - img
    #img(i,j)  - img(i,j-1)
    glx = img - img_right 
    #img(i,j)   - img(i-1,j)
    gly = img - img_down   
    #img(i,j+1) + img(i+1,j)+ img(i,j-1) +img(i-1,j)  - 4*I(i,j)
    grc = grx+gry-glx-gly  
    return grx,gry,glx,gly,grc

def qq(img):
    """
    Instantaneous coefficient of variation: q(x,y,t)
    
    Parameters
    ------
    img: ndarray
    
    Returns
    ------
    q : ndarray
        The formula is as follows:
        q(x, y ; t)=\sqrt{\frac{(1 / 2)(|\nabla I| / I)^{2}
        -\left(1 / 4^{2}\right)\left(\nabla^{2} I / I\right)^{2}}
        {\left[1+(1 / 4)\left(\nabla^{2} I / I\right)\right]^{2}}}
    """
    grx,gry,glx,gly,grc = Gradient(img)
    q_1 = (grx**2+gry**2+glx**2+gly**2)**0.5/(img+1e-06)
    q_2 = grc /(img+1e-06)
    q   = ((1/2*q_1**2 - 1/16*q_2**2) / ((1+1/4*q_2)**2)+1e-06)**0.5
    
    return q
    
def srad(img,k = 30,m = 0.5,q_0 = 1,rho = 1,delta_t = 0.05,Iterations = 400):
    """
    speckle reducing anistropic diffusion
    
    Parameter
    ------
    img: ndarray
    
    k:  number
        attenuation coefficient
    m； number
        control rate of homogeneous area
    q_0: number
        the threshold of intial speckle noise
    rho: number
    delta_t:number
        timespace
    Iteration: number
        the number of iterations
    
    Returns
    img: ndarray 
        the image after being filtered by srad
    """
    for i in range(0,Iterations):
        grx,gry,glx,gly,grc = Gradient(img)
    
        # compute the diffusion coefficient
        q  = qq(img)
        q_t = q_0*math.exp(-rho*i*delta_t)
        cq = np.pi/2 - np.arctan(k*(q**2 - m*q_t**2))
        
        # cq(i+1,j)
        cq_up = np.roll(cq,-1,axis = 0)
        cq_up[-1] = cq[-1]
        # cq(i,j+1)
        cq_left = np.roll(cq,-1,axis = 1)
        cq_left[:,-1] = cq[:,-1]
        
        Div = cq_up*gry - cq*gly + cq_left*grx-cq*glx
        img = img + 1/4*delta_t*Div
    return img

 # 2020 06 22