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


def get_line(coords):
    assert np.shape(coords)[0] == 2
    x1 = np.hstack((coords[0],1))
    x2 = np.hstack((coords[1],1))
    l = np.cross(x1,x2)
    l = np.divide(l,l[-1])
    return l

def get_intersection(l1,l2):
    x = np.cross(l1,l2)
    x = np.divide(x,x[-1])
   
    return x

def find_l_infinity(coords):
    assert np.shape(coords)[0] == 4
    l1,l2 = get_line(coords[:2]), get_line(coords[2:])
    p1 = np.cross(l1,l2)
    p1 = np.divide(p1,p1[-1])
    l3,l4 = get_line(coords[::2]), get_line(coords[1::2])
    p2 = np.cross(l3,l4)
    p2 = np.divide(p2,p2[-1])
    l_inf = np.cross(p1,p2)
    l_inf = np.divide(l_inf,l_inf[-1])
   
    return p1,p2,l_inf

def get_rectified_image(img,H):
    r_img = np.zeros_like(img)
    h, w, _ = r_img.shape
    for y in range(h):
        for x in range(w):
            x_new, y_new, z_new = H.dot([x,y,1])
            x_new, y_new = int(np.around(x_new/z_new)),int(np.around(y_new/z_new))
            if 0 <= x_new < w and 0 <= y_new < h:
                r_img[y_new,x_new,:] = img[y,x,:]

    img_n = manual_interpolation(r_img)
    return img_n


def get_shifted_rectified_image(img,H,x_off=0,y_off=0):
    r_img = np.zeros_like(img)
    h, w, _ = r_img.shape
    for y in range(h):
        for x in range(w):
            x_new, y_new, z_new = H.dot([x,y,1])
            x_new, y_new = int(np.around(x_new/z_new)),int(np.around(y_new/z_new))
            print(x_new,y_new)
            x_new = x_new + int(x_off)
            y_new = y_new + int(y_off)
            print('n:',x_new,y_new)
            if 0 <= x_new < w and 0 <= y_new < h:
                r_img[y_new,x_new,:] = img[y,x,:]

    img_n = manual_interpolation(r_img)
    return img_n