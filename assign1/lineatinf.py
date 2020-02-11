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
                    [[213,130],[392,76],[224,338],[390,220]],\
                    [[357,40],[584,116],[331,387],[568,320]]])

ACTUAL_COORDS = np.asarray([[[200,100],[400,100],[200,300],[400,300]],\
                            [[200,100],[400,100],[200,300],[400,300]],\
                            [[160,80],[320,80],[160,240],[320,240]],\
                            [[300,50],[400,50],[300,200],[400,200]],\
                            [[175,100],[425,100],[175,270],[425,270]],\
                            [[200,150],[550,150],[200,400],[550,400]]])
