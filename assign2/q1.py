import cv2
import numpy as np
from glob import glob
np.set_printoptions(suppress=True, formatter={'float_kind':'{:f}'.format})

from config import *
from utils import *

files = sorted(glob(INPUT_FILE_PATH + '/*'))

for k,file in enumerate(files):
    img = cv2.imread(file)
    resize_f = 2
    if img.shape[0] > 1000:
        resize_f = 8
    height_, width_ = int(img.shape[0]/resize_f), int(img.shape[1]/resize_f)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    res = cv2.resize(img, (width_,height_), interpolation=cv2.INTER_CUBIC)
    cv2.imshow('image', res)
    print(res.shape)
    filename = 'image' + str(k+1) + '.jpg'
    cv2.imwrite(filename, res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    break