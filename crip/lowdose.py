'''
    Low Dose CT (LDCT) module of crip.

    https://github.com/z0gSh1u/crip
'''

__all__ = ['injectGaussianNoise', 'injectPoissonNoise', 'totalVariation']

import numpy as np
from .shared import *
from ._typing import *
from .utils import *


@ConvertListNDArray
def injectGaussianNoise(projections: TwoOrThreeD, sigma: float, mu: float = 0) -> TwoOrThreeD:
    '''
        Inject Gaussian noise which obeys distribution `N(\\mu, \\sigma^2)`.
    '''
    cripAssert(is2or3D(projections), '`projections` should be 2D or 3D.')
    cripAssert(sigma > 0, 'sigma should > 0.')

    injectOne = lambda img: (np.random.randn(*img.shape) * sigma + mu) + img

    if is3D(projections):
        res = np.array(list(map(injectOne, projections)))
    else:
        res = injectOne(projections)

    return res


@ConvertListNDArray
def injectPoissonNoise(projections: TwoOrThreeD, rescale: Or[int, float] = 1) -> TwoOrThreeD:
    '''
        Inject Poisson noise which obeys distribution `P(\\lambda)` where \\lambda is the ground-truth quanta in `projections`.
        `projections` must have int type whose value stands for the photon quanta in some way. Floating projections
        should be manually properly rescaled ahead and scale back as you need since Poisson Distribution only deals with
        positive integers.
    '''
    cripAssert(is2or3D(projections), '`projections` should be 2D or 3D.')
    cripAssert(np.min(projections >= 0), '`projections` should not contain negative values.')
    cripWarning(isIntType(projections), '`projections` should have int dtype. It will be floored after rescaling.')

    img = projections * rescale
    img = np.random.poisson(img.astype(np.uint32)).astype(DefaultFloatDType)
    img[img <= 0] = 1
    img /= rescale

    return img


@ConvertListNDArray
def totalVariation(img: TwoOrThreeD) -> TwoOrThreeD:
    '''
        Computes the Total Variation (TV) of image or images.
    '''
    cripAssert(is2or3D(img), 'img should be 2 or 3D.')

    vX = img[..., :, 1:] - img[..., :, :-1]
    vY = img[..., 1:, :] - img[..., :-1, :]
    tv = np.sum(np.abs(vX) + np.abs(vY), axis=(-2, -1))

    return tv
