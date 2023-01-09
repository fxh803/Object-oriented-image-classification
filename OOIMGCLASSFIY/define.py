import numpy as np
np.seterr(divide='ignore',invalid='ignore')
import lbp
from collections import Counter
# 灰度均值
def gray_mean(obj):
    avg = []
    for i in range(len(obj)):
        avg.append(obj[i].mean(axis=0))
    return avg


# 灰度标准差
def gray_standard_deviation(obj):
    sd = []
    for i in range(len(obj)):
        sd.append(obj[i].mean(axis=0))
    return sd


# 灰度最大值
def gray_max(obj):
    max = []
    for i in range(len(obj)):
        max.append(obj[i].mean(axis=0))
    return max


# 灰度最小值
def gray_min(obj):
    min = []
    for i in range(len(obj)):
        min.append(obj[i].mean(axis=0))
    return min


# 纹理
def texture(img, segments, n_segments):
    basic_arr = lbp.lbp_basic(img) 
    tex_histogram = np.zeros((n_segments, 256))
    for i in range(len(img)):
        for j in range(len(img[0])):
            tex_histogram[segments[i][j]][basic_arr[i][j]] += 1
    return tex_histogram


# 形状
def area(segments):
    areas = []
    n_segments = len(np.unique(segments))
    count = Counter(segments.flatten())
    for i in range(n_segments):
        areas.append([float(count[i])])
    return areas


# 归一化
def normalization(feature, n_segments):
    feature = np.array(feature)
    for i in range(n_segments):
        feature[i] = (feature[i] - feature.min(axis=0)) / (feature.max(axis=0) - feature.min(axis=0))
    feature[np.isnan(feature)] = 0
    return feature

