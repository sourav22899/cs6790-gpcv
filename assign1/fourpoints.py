import cv2
from glob import glob
import numpy as np
import matplotlib
from tqdm import tqdm

from config import *
files = sorted(glob(OUTPUT_FILE_PATH + '/*'))

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

COORDS = np.asarray([[[200,99],[390,55],[283,269],[504,196]],\
                    [[109,212],[280,150],[332,290],[533,266]],\
                    [[112,109],[294,78],[174,293],[384,222]],\
                    [[284,108],[374,117],[266,183],[369,193]],\
                    [[213,130],[392,76],[291,368],[372,302]],\
                    [[357,40],[584,116],[331,387],[568,320]]])

ACTUAL_COORDS = np.asarray([[[200,100],[400,100],[200,300],[400,300]],\
                            [[200,100],[400,100],[200,300],[400,300]],\
                            [[160,80],[320,80],[160,240],[320,240]],\
                            [[300,50],[400,50],[300,200],[400,200]],\
                            [[150,100],[450,100],[250,325],[350,325]],\
                            [[200,150],[550,150],[200,400],[550,400]]])


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
