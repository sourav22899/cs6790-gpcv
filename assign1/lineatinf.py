import cv2
from glob import glob
import numpy as np
from tqdm import tqdm
np.set_printoptions(suppress=True, formatter={'float_kind':'{:f}'.format})

from config import *
from utils import *

files = sorted(glob(OUTPUT_FILE_PATH + '/*'))
COORDS, ACTUAL_COORDS = np.asarray(COORDS,dtype=np.int32), np.asarray(ACTUAL_COORDS,dtype=np.int32)
AFFINE_COORDS = np.asarray(AFFINE_COORDS, dtype=np.int32)

for k, file in enumerate(files):
    img = cv2.imread(file)
    coords = COORDS[k]
    _,_,l_inf = find_l_infinity(coords)
    h = np.eye(3)
    if k in [1,4,5]:
        h = h*0.5
    h[-1] = l_inf
    img_n = get_rectified_image(img,h)

    aff_coords = AFFINE_COORDS[k]
    img_ = find_key_points(img_n, aff_coords)
    cv2.imshow('rect_img',img_)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    aff_coords = aff_coords.flatten().reshape(4,2,-1)
    p_lines = []
    for i in range(aff_coords.shape[0]):
        p_lines.append(get_line(aff_coords[i]))

    A = np.zeros((2,3))
    for i in range(A.shape[0]):
        l1,l2 = p_lines[2*i],p_lines[2*i+1]
        A[i] = np.asarray([l1[0]*l2[0],l1[0]*l2[1]+l1[1]*l2[0],l1[1]*l2[1]])
    
    _,_,vh = np.linalg.svd(A)
    x = vh.T[:,-1]
    s = np.asarray([[x[0],x[1]],[x[1],x[2]]])
    u,d,_ = np.linalg.svd(s)
    K = np.matmul(u,np.diag(np.sqrt(d)))
    H = np.eye(3,3)
    H[0:2,0:2] = K

    H_inv = np.linalg.pinv(H)
    H_inv = np.divide(H_inv,H_inv[-1,-1])
    y_off = -int(img.shape[0]//2) if k > 2 else 0
    img_n = get_shifted_rectified_image(img_n,H_inv,x_off=img.shape[1],y_off=y_off)

    fname = 'q2' + str(k+1) + '.jpg'
    img_n = cv2.flip(img_n, 1)
    cv2.imwrite(fname,img_n)
    cv2.imshow('rect_img',img_n)
    cv2.waitKey(0)
    cv2.destroyAllWindows()