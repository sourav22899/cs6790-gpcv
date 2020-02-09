import cv2
import numpy as np
from glob import glob

from config import *

files = sorted(glob(INPUT_FILE_PATH + '/*'))

for i,file in enumerate(files):
    img = cv2.imread(file)
    print(img.shape)
    res = cv2.resize(img, (IM_RESIZE_WIDTH,IM_RESIZE_HEIGHT), interpolation=cv2.INTER_CUBIC)
    cv2.imshow('image', res)
    filename = OUTPUT_FILE_PATH + '/image' + str(i+1) + '.jpg'
    cv2.imwrite(filename, res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    