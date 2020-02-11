import cv2
import numpy as np
from tqdm import tqdm

def find_key_points(img, coords, radius=5, color=(255,0,0)):
    img_ = img.copy()
    for point in coords:
        x,y = point[0],point[1]
        img_ = cv2.circle(img_, (x,y), radius, color, -1)
    
    return img_

def max_neighbour(img,i,j):
    dict_ = {}
    for x in [i-1,i,i+1]:
        for y in [j-1,j,j+1]:
            dict_[np.sum(img[x,y])] = img[x,y]
    m = max(dict_.keys())
    
    return dict_[m]

def manual_interpolation(img):
    img_ = img.copy()
    h,w = img_.shape[0],img_.shape[1]
    for y in tqdm(range(1,h-1)):
        for x in range(1,w-1):
            if np.sum(img_[y,x]) == 0:
                img_[y,x] = max_neighbour(img,y,x)
    img_[0] = img_[1]
    img_[-1] = img_[-2]
    img_[:,0] = img_[:,1]
    img_[:,-1] = img_[:,-2]

    return img_
