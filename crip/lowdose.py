'''
    Low Dose CT (LDCT) module of crip.

    https://github.com/z0gSh1u/crip
'''

__all__ = ['limitAngle', 'limitView', 'injectGaussianNoise', 'injectPoissonNoise', 'totalVariation']

import numpy as np
from .shared import *
from ._typing import *
from .utils import *


def limitAngle(projections: ThreeD, total: float, start: float, dst: float):
    cripAssert(False, 'Unimplemented.')


def limitView(projections: ThreeD, ratio: float):
    cripAssert(False, 'Unimplemented.')


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
def injectPoissonNoise(projections: TwoOrThreeD, nPhoton: int) -> TwoOrThreeD:
    '''
        Inject Poisson noise which obeys distribution `P(\\lambda)`.
        `nPhoton` is the number of photon hitting per detector element.
    '''
    cripAssert(is2or3D(projections), '`projections` should be 2D or 3D.')

    def injectOne(img):
        I0 = np.max(img)
        cripAssert(I0 != 0, 'The maximum of img is 0.')
        proj = nPhoton * np.exp(-img / I0)
        proj = np.random.poisson(proj)
        proj = -np.log(img / nPhoton) * I0
        return proj

    if is3D(projections):
        res = np.array(list(map(injectOne, projections)))
    else:
        res = injectOne(projections)

    return res


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
