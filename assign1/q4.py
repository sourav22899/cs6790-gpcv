import cv2
from glob import glob
import numpy as np
from tqdm import tqdm
np.set_printoptions(suppress=True, formatter={'float_kind':'{:f}'.format})

from config import *
from utils import *

files = sorted(glob(OUTPUT_FILE_PATH + '/*'))
COORDS = np.asarray(COORDS,dtype=np.int32)
AFFINE_COORDS_2 = np.asarray(AFFINE_COORDS_2, dtype=np.int32)

for k, file in enumerate(files):
    img = cv2.imread(file)
    coords = COORDS[k]
    _,_,l_inf = find_l_infinity(coords)
    h = np.eye(3)
    if k in [1,4,5]:
        h = h*0.5
    h[-1] = l_inf
    img_n = get_rectified_image(img,h)

    aff_coords = AFFINE_COORDS_2[k]
    img_ = find_key_points(img_n, aff_coords)
    cv2.imshow('rect_img',img_)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    A = np.zeros((5,6))

    for i,points in enumerate(aff_coords):
        x,y = points[0]/600.0,points[1]/400.0
        A[i] = np.asarray([x**2,x*y,y**2,x,y,1])
    
    _,_,vh = np.linalg.svd(A)
    x = vh.T[:,-1]
    c = np.asarray([[x[0],0.5*x[1],0.5*x[3]],\
                    [0.5*x[1],x[2],0.5*x[4]],\
                    [0.5*x[3],0.5*x[4],x[5]]])
    
    # print(c)
    print('this should be positive:',c[0,0]*c[1,1] - c[0,1]*c[1,0])
    s = np.asarray([[c[1,1],-c[0,1]],[-c[1,0],c[0,0]]])
    u,d,_ = np.linalg.svd(s)
    K = np.matmul(u,np.diag(np.sqrt(d)))
    H = np.eye(3,3)
    H[0:2,0:2] = K

    print(H)
    H_inv = np.linalg.pinv(H)
    H_inv = np.divide(H_inv,H_inv[-1,-1])
    y_off = -int(img.shape[0]//2) if k > 2 else 0
    img_n = get_shifted_rectified_image(img_n,H_inv,x_off=0,y_off=y_off)

    fname = 'q4' + str(k+1) + '.jpg'
    img_n = cv2.flip(img_n, 1)
    cv2.imwrite(fname,img_n)
    cv2.imshow('rect_img',img_n)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if k == 0:
        break