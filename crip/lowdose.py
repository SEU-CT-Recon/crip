'''
    Low Dose CT module of crip.

    https://github.com/SEU-CT-Recon/crip
'''

import numpy as np
from .shared import *
from ._typing import *
from .utils import *


@ConvertListNDArray
def injectGaussianNoise(projections: TwoOrThreeD, sigma: float, mu: float = 0) -> TwoOrThreeD:
    ''' Inject additive Gaussian noise ~ `N(mu, sigma^2)` where `sigma` is the standard deviation and `mu` is the mean.
    '''
    cripAssert(is2or3D(projections), f'`projections` should be 2 or 3-D, but got {projections.ndim}-D.')
    cripAssert(sigma > 0, 'sigma should be greater than 0.')

    _inject1 = lambda img: (np.random.randn(*img.shape) * sigma + mu) + img
    if is3D(projections):
        res = np.array(list(map(_inject1, projections)))
    else:
        res = _inject1(projections)

    return res


@ConvertListNDArray
def injectPoissonNoise(
    projections: TwoOrThreeD,
    type_: str = 'postlog',
    nPhoton: int = 1e5,
) -> TwoOrThreeD:
    ''' Inject Poisson noise ~ `P(lambda)` where `lambda` is the ground-truth quanta deduced from arguments.
        `type_` [postlog or raw] gives the content type of `projections`, usually you use
        postlog as input and get postlog as output. `nPhoton` is the photon count for each pixel.
    '''
    cripAssert(type_ in ['postlog', 'raw'], f'Invalid type_: {type_}.')
    cripAssert(is2or3D(projections), '`projections` should be 2D or 3D.')

    img = projections
    if type_ == 'postlog':
        img = np.exp(-img)

    img = nPhoton * img  # N0 exp(-\sum \mu L), i.e., ground-truth quanta
    img = np.random.poisson(img.astype(np.uint32)).astype(DefaultFloatDType)  # noisy quanta
    img[img <= 0] = 1
    img /= nPhoton  # cancel the rescaling from N0

    if type_ == 'postlog':
        img = -np.log(img)

    return img


@ConvertListNDArray
def totalVariation(img: TwoOrThreeD) -> TwoOrThreeD:
    ''' Computes the Total Variation (TV) of images.
    '''
    cripAssert(is2or3D(img), 'img should be 2 or 3D.')

    vX = img[..., :, 1:] - img[..., :, :-1]
    vY = img[..., 1:, :] - img[..., :-1, :]
    tv = np.sum(np.abs(vX) + np.abs(vY), axis=(-2, -1))

    return tv


def nps2D(roi: TwoOrThreeD, pixelSize: float, detrend='individual', n: Or[int, None] = None):
    ''' Compute the noise power spectrum (NPS) of a 2D square ROI using `n`-dots DFT.
        Recommended to provide multiple realizations of that ROI.
        `pixelSize` is the pixel size of reconstructed image ([mm] recommended).
    '''
    cripWarning(is3D(roi), "It's highly recommended to...")

    roi = as3D(roi)
    h, w = getHnW(roi)
    cripAssert(h == w, 'Width and height of ROI should be the same.')
    cripAssert(detrend in ['individual', 'mutual'], f'Invalid detrend method: {detrend}.')
    dots = n or nextPow2(h)

    if detrend == 'individual':
        detrended = roi - np.mean(roi)

    dft = np.fft.fftshift(np.fft.fft2(deTrend, n=dots))
    mod2 = np.real(dft * np.conj(dft))

    return mod2 * pixelSize * pixelSize / (h * w)


def nps2DRadialAvg(roi: TwoOrThreeD, pixelSize: float, n: Or[int, None] = None):
    ''' Compute the radially averaged noise power spectrum (NPS) of a 2D region of interest (ROI).
        It's recommended that you provide multiple samples (realizations) of the ROI.
    '''
    nps = nps2D(roi, pixelSize, n)
    n = nps.shape[0]

    x, y = np.meshgrid(np.arange(nps.shape[1]), np.arange(nps.shape[0]))
    R = np.sqrt(x**2 + y**2)

    f = lambda r: nps[(R >= r - .5) & (R < r + .5)].mean()
    r = np.linspace(1, n, num=n)
    mean = np.vectorize(f)(r)

    return mean
