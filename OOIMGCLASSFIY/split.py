import numpy as np
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import slic, watershed, quickshift, felzenszwalb
from skimage.segmentation import mark_boundaries

# 分水岭
def WatershedSeg(img):
    gradient = sobel(rgb2gray(img))
    segments = watershed(gradient, markers=255, compactness=0.001) - 1
    result = mark_boundaries(img, segments)
    segments, objects= toobjects(img, segments)
    return segments, result, objects

# 超像素 SLIC
def SlicSeg(img):
    segments = slic(img, n_segments=400, compactness=30, sigma=1)
    result = mark_boundaries(img, segments)
    segments, objects= toobjects(img, segments)
    return segments, result, objects

# 快速位移
def QuickSeg(img):
    segments = quickshift(img, kernel_size=3, max_dist=100, ratio=0.5)
    result = mark_boundaries(img, segments)
    segments, objects= toobjects(img, segments)
    return segments, result, objects

# 基于菲尔森茨瓦布高效图
def FelzenSeg(img):
    segments = felzenszwalb(img, scale=100, sigma=0.5, min_size=150)
    result = mark_boundaries(img, segments)
    segments, objects= toobjects(img, segments)
    return segments, result, objects

# 对象生成
def toobjects(img, segments):
    objects = []
    for obj in np.unique(segments):
        obj_value = []
        for height, width in np.argwhere(segments == obj):
            obj_value.append(img[height, width, :])
        objects.append(np.array(obj_value))
    return segments, objects


