import cv2
import numpy as np
from glob import glob
np.set_printoptions(suppress=True, formatter={'float_kind':'{:f}'.format})

from config import *
from utils import *

files = sorted(glob('*.jpg'))
COORDS = np.asarray(COORDS,dtype=np.int32)

A = np.zeros((5,6))
for k,file in enumerate(files):
    if k < 5:
        coords = COORDS[k]
        p1,p2,l_inf = find_l_infinity(coords)
        print(p1,p2,l_inf)
        A[k] = np.asarray([p1[0]*p2[0],p1[1]*p2[1],1,p1[0]*p2[1]+p1[1]*p2[0],p1[0]+p2[0],p1[1]+p2[1]])
    else:
        break

_,_,vh = np.linalg.svd(A)
x = vh.T[:,-1]
w = np.asarray([[x[0],x[3],x[4]],[x[3],x[1],x[5]],[x[4],x[5],x[2]]])
kkt = np.linalg.pinv(w)
kkt = np.divide(kkt,kkt[-1,-1])
print(kkt)

def build_k_matrix(kkt,resize_f=8):
    k = np.zeros((3,3))
    k[:,-1] = kkt[:,-1]
    k[1,1] = np.sqrt(kkt[1,1]-kkt[1,2]**2)
    k[0,1] = (kkt[0,1] - kkt[0,2]*kkt[1,2])/k[1,1]
    k[0,0] = np.sqrt(kkt[0,0]-k[0,1]**2-kkt[0,2]**2)
    k = k*resize_f;k[-1,-1] = 1
    return k

k = build_k_matrix(kkt)
print(k)