'''
    Metrics calculator module of crip.

    https://github.com/z0gSh1u/crip
'''

import numpy as np
from skimage.metrics import structural_similarity


def calcMAPE(x, y):
    return np.mean(np.abs(x - y) / y) * 100


def calcSSIM(x, y):
    ssim = structural_similarity(x, y, data_range=1)
    return ssim


def calcRMSE(x, y):
    sq = (x - y)**2
    mean = sq.mean()
    return np.sqrt(mean)
