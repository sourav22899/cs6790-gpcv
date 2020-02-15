import cv2
from glob import glob
import numpy as np
from tqdm import tqdm
np.set_printoptions(suppress=True, formatter={'float_kind':'{:f}'.format})

from config import *
from utils import *

files = sorted(glob(OUTPUT_FILE_PATH + '/*'))
PERP_COORDS = np.asarray(PERP_COORDS,dtype=np.int32)

for k, file in enumerate(files):
    img = cv2.imread(file)
    coords = PERP_COORDS[k]
    img_ = find_key_points(img, coords)
    cv2.imshow('image',img_)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    p_coords = coords.flatten().reshape(10,2,-1)
    print(p_coords)
    p_lines = []
    for i in range(p_coords.shape[0]):
        p_lines.append(get_line(p_coords[i]))
    
    for i in p_lines:
        print(i)

    A = np.zeros((5,6))
    for i in range(A.shape[0]):
        l1,l2 = p_lines[2*i],p_lines[2*i+1]
        A[i] = np.asarray([l1[0]*l2[0],0.5*(l1[0]*l2[1]+l1[1]*l2[0]),\
                           l1[1]*l2[1],0.5*(l1[0]*l2[2]+l1[2]*l2[0]),\
                           0.5*(l1[1]*l2[2]+l1[2]*l2[1]),l1[2]*l2[2]])

    _,_,vh = np.linalg.svd(A)
    x = vh.T[:,-1]
    print('x:',x)
    s = np.asarray([[x[0],0.5*x[1],0.5*x[3]],\
                    [0.5*x[1],x[2],0.5*x[4]],\
                    [0.5*x[3],0.5*x[4],x[5]]])
    print('s:')
    print(s)
    u,d,_ = np.linalg.svd(s)
    d = np.sqrt(d); d[-1] = 1
    H = np.matmul(u,np.diag(d))
    print('root_d:',d)
    H_inv = np.linalg.pinv(H)
    H_inv = np.divide(H_inv,H_inv[-1,-1])
    print('H_inv:')
    print(H_inv)

    img_n = get_shifted_rectified_image(img,H_inv,x_off=img.shape[1],y_off=img.shape[0])
    fname = 'q3' + str(k+1) + '.jpg'
    cv2.imwrite(fname,img_n)
    cv2.imshow('rect_img',img_n)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if k == 0:
        break