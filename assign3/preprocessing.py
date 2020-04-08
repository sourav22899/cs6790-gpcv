import cv2
import numpy as np
from glob import glob

IM_RESIZE_WIDTH = 600
IM_RESIZE_HEIGHT = 400

files = sorted(glob('/home/sourav/Semester VI/CS 6790/Images_assign_3/*'))
print(files)
for i,file in enumerate(files):
    img = cv2.imread(file)
    print(img.shape)
    res = cv2.resize(img, (IM_RESIZE_WIDTH,IM_RESIZE_HEIGHT), interpolation=cv2.INTER_CUBIC)
    cv2.imshow('image', res)
    filename = 'image' + str(i+1) + '.jpg'
    cv2.imwrite(filename, res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    