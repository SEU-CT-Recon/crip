'''
    Low Dose CT module of crip.

    https://github.com/SEU-CT-Recon/crip
'''

import numpy as np

from ._typing import *
from .shared import *
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


def nps2D(roi: TwoOrThreeD, pixelSize: float, detrend='individual', n: Or[int, None] = None) -> TwoD:
    ''' Compute the noise power spectrum (NPS) of a 2D square ROI using `n`-dots DFT.
        `pixelSize` is the pixel size of reconstructed image ([mm] recommended).
        `detrend` method can be `individual` (by mean value subtraction) or `mutual` (by foreground subtraction).
    '''
    cripAssert(detrend in ['individual', 'mutual'], f'Invalid detrend method: {detrend}.')
    cripWarning(is3D(roi), "It's highly recommended to provide multiple realizations of the ROI.")
    if detrend == 'mutual':
        cripAssert(is3D(roi), '`mutual` detrend method requires multiple realizations of the ROI.')

    roi = as3D(roi)
    h, w = getHnW(roi)
    cripAssert(h == w, 'ROI should be the square.')
    dots = n or nextPow2(h)

    # de-trend the signal
    if detrend == 'individual':
        detrended = roi - np.mean(roi, axis=0)
        s = 1
    elif detrend == 'mutual':
        detrended = roi[:-1, ...] - roi[1:, ...]
        detrended = detrended - np.mean(detrended, axis=0)
        s = 1 / 2

    dft = np.fft.fftshift(np.fft.fft2(detrended, n=dots))
    dft = np.mean(dft, axis=0)  # averaged NPS
    mod2 = np.real(dft * np.conj(dft))  # square of modulus

    return (pixelSize * pixelSize) / (h * w) * mod2 * s


def nps2DRadialAvg(nps: TwoD) -> NDArray:
    ''' Compute the radially averaged noise power spectrum (NPS) where `nps` can be the output from `nps2D`.
    '''
    cripAssert(is2D(nps), '`nps` should be 2D.')
    cripAssert(nps.shape[0] == nps.shape[1], '`nps` should be square.')

    N = nps.shape[0]
    x, y = np.meshgrid(np.arange(N), np.arange(N))
    R = np.sqrt(x**2 + y**2)

    _f = lambda r: nps[(R >= r - .5) & (R < r + .5)].mean()
    _args = np.linspace(1, N, num=N)

    return np.vectorize(_f)(_args)
