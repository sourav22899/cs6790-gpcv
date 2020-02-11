import cv2
from glob import glob
import numpy as np
import matplotlib
from tqdm import tqdm

from config import *
from utils import find_key_points, manual_interpolation

files = sorted(glob(OUTPUT_FILE_PATH + '/*'))
COORDS, ACTUAL_COORDS = np.asarray(COORDS,dtype=np.int32), np.asarray(ACTUAL_COORDS,dtype=np.int32)

for k, file in enumerate(files):
    img = cv2.imread(file)
    coords = COORDS[k]
    actual_coords = ACTUAL_COORDS[k]
    img_ = find_key_points(img, coords)
    cv2.imshow('image',img_)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    A = np.zeros((8,9))
    for i, point in enumerate(coords):
        x_,y_ = point[0],point[1]
        x,y = actual_coords[i][0],actual_coords[i][1]
        A[2*i,:3] = [-x,-y,-1]
        A[2*i,-3:] = [x*x_,y*x_,x_]
        A[2*i+1,3:6] = [-x,-y,-1]
        A[2*i+1,-3:] = [x*y_,y*y_,y_]
        
    _,_,vh = np.linalg.svd(A)
    h = vh.T[:,-1].reshape(3,3)
    h_inv = np.linalg.pinv(h)
    h_inv = h_inv/h_inv[-1,-1]

    r_img = np.zeros_like(img)
    h, w, _ = r_img.shape
    for y in range(h):
        for x in range(w):
            x_new, y_new, z_new = h_inv.dot([x,y,1])
            x_new, y_new = int(np.around(x_new/z_new)),int(np.around(y_new/z_new))
            if 0 <= x_new < w and 0 <= y_new < h:
                r_img[y_new,x_new,:] = img[y,x,:]
    
    cv2.imwrite('rect1_i.jpg',r_img)
    cv2.imshow('rect_img',r_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    img_n = manual_interpolation(r_img)

    fname = 'rect' + str(k+1) + '.jpg'
    cv2.imwrite(fname,img_n)
    cv2.imshow('rect_img',img_n)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
