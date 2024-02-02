'''
    Metrics calculator module of crip.

    https://github.com/SEU-CT-Recon/crip
'''

import numpy as np
from .utils import cripWarning, cripAssert, is3D, isOfSameShape
from ._typing import *
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

__all__ = ['calcMAPE', 'calcPSNR', 'calcSSIM', 'calcRMSE']


def calcMAPE(x, y, eps=1e-6):
    '''
        Compute the Mean Absolution Percentage Error (%) where `x` is the prediction
        and `y` is the ground truth.
    '''
    return np.mean(np.abs(x - y) / (y + eps)) * 100


def calcPSNR(x: TwoOrThreeD, y: TwoOrThreeD, range_=1):
    '''
        Compute the Peak Signal Noise Ratio (PSNR) (dB).
    '''
    cripWarning(
        range_ == 1 or range_ == 255,
        'Common `range_` for PSNR computation is 1 or 255. Make sure you know what you are doing follows your intention.'
    )
    psnr = peak_signal_noise_ratio(x, y, data_range=range_)
    return psnr


def calcSSIM(x: TwoOrThreeD, y: TwoOrThreeD, range_=1):
    '''
        Compute the Structural Similarity (SSIM).
    '''
    cripWarning(
        range_ == 1 or range_ == 255,
        'Common `range_` for SSIM computation is 1 or 255. Make sure you know what you are doing follows your intention.'
    )
    cripAssert(isOfSameShape(x, y), 'Input images must have the same shape.')

    ssim = structural_similarity(x, y, data_range=range_, channel_axis=0 if is3D(x) else None)

    return ssim


def calcRMSE(x, y):
    '''
        Compute the Root Mean Squared Error.
    '''
    sq = (x - y)**2
    return np.sqrt(sq.mean())


class AverageMeter():

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.val = val
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count

    def getValue(self):
        return self.avg
