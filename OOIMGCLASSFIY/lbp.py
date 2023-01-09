import cv2 as cv
import numpy as np
import skimage.feature
def lbp_basic(img):
    img1 = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    img_ku = skimage.feature.local_binary_pattern(img1,8,1.0,method='default')
    img_ku = img_ku.astype(np.uint8)

    return img_ku
