'''
    Metrics module of crip.

    https://github.com/SEU-CT-Recon/crip
'''

import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

from .utils import cripWarning, cripAssert, is3D, isOfSameShape
from ._typing import *


def computeMAPE(x, y, eps=1e-6) -> float:
    ''' Compute the Mean Absolution Percentage Error (%) where `x` is the prediction
        and `y` is the ground truth.
    '''
    return np.mean(np.abs(x - y) / (y + eps)) * 100


def computePSNR(x: TwoOrThreeD, y: TwoOrThreeD, range_=1) -> float:
    ''' Compute the Peak Signal Noise Ratio (PSNR) (dB) between `x` and `y`.
    '''
    cripWarning(range_ in [1, 255],
                '`range_` for PSNR is usually 1 or 255. Be sure your behavior follows your intention.')
    cripAssert(isOfSameShape(x, y), 'Input images must have the same shape.')

    return peak_signal_noise_ratio(x, y, data_range=range_)


def computeSSIM(x: TwoOrThreeD, y: TwoOrThreeD, range_=1) -> float:
    ''' Compute the Structural Similarity (SSIM) between `x` and `y`.
    '''
    cripWarning(range_ in [1, 255],
                '`range_` for SSIM is usually 1 or 255. Be sure your behavior follows your intention.')
    cripAssert(isOfSameShape(x, y), 'Input images must have the same shape.')

    return structural_similarity(x, y, data_range=range_, channel_axis=0 if is3D(x) else None)


def computeRMSE(x: TwoOrThreeD, y: TwoOrThreeD) -> float:
    ''' Compute the Root Mean Squared Error (RMSE) between `x` and `y`.
    '''
    sq = (x - y)**2

    return np.sqrt(sq.mean())


class AverageMeter():
    ''' Computes and stores the average and current value in `avg` and `val` respectively.
    '''

    def __init__(self):
        self.val = 0  # current value
        self.avg = 0  # average value
        self.sum = 0  # sum of all values
        self.count = 0  # number of values

    def update(self, val):
        self.val = val
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count
