import cv2
from glob import glob
import numpy as np
from tqdm import tqdm
np.set_printoptions(suppress=True,
   formatter={'float_kind':'{:f}'.format})

from config import *
from utils import *

files = sorted(glob(OUTPUT_FILE_PATH + '/*'))
COORDS, ACTUAL_COORDS = np.asarray(COORDS,dtype=np.int32), np.asarray(ACTUAL_COORDS,dtype=np.int32)
# AFFINE_COORDS = np.asarray(AFFINE_COORDS, dtype=np.int32)

for k, file in enumerate(files):
    img = cv2.imread(file)
    coords = COORDS[k]
    _,_,l_inf = find_l_infinity(coords)
    h = np.eye(3)
    if k in [1,4,5]:
        h = h*0.5
    h[-1] = l_inf
    img_n = get_rectified_image(img,h)

    cv2.imshow('rect_img',img_n)
    cv2.waitKey(0)
    cv2.destroyAllWindows()