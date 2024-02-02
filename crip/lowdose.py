'''
    Low Dose CT (LDCT) module of crip.

    https://github.com/SEU-CT-Recon/crip
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
def injectPoissonNoise(projections: TwoOrThreeD,
                       type_: str = 'postlog',
                       nPhoton: Or[int, float] = 1,
                       suppressWarning=False) -> TwoOrThreeD:
    '''
        Inject Poisson noise which obeys distribution `P(\\lambda)` where \\lambda is the ground-truth quanta in `projections`.
        `projections` must have int type whose value stands for the photon quanta in some way. Floating projections
        should be manually properly rescaled ahead and scale back as you need since Poisson Distribution only deals with
        positive integers. `type_` (postlog or raw) gives the content type of `projections`, usually you should use
        postlog as input. If you input postlog, the output will also be postlog.
    '''
    cripAssert(type_ in ['postlog', 'raw'], "type_ should be 'postlog' or 'raw'.")
    img = projections
    cripAssert(is2or3D(img), '`projections` should be 2D or 3D.')

    if type_ == 'postlog':
        img = np.exp(-projections)

    img = img * nPhoton  # N0 exp(-\sum \mu L)

    cripAssert(np.min(img >= 0), '`img` should not contain negative values.')
    if not suppressWarning:
        # temporary workaround
        cripWarning(isIntType(img), '`img` should have int dtype. It will be floored after rescaling.')

    img = np.random.poisson(img.astype(np.uint32)).astype(DefaultFloatDType)
    img[img <= 0] = 1
    img /= nPhoton

    if type_ == 'postlog':
        img = -np.log(img)

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


def nps2D(roi: TwoOrThreeD, pixelSize: float, n: Or[int, None] = None):
    '''
        Compute the noise power spectrum (NPS) of a 2D region of interest (ROI).
        It's recommended that you provide multiple samples (realizations) of the ROI.
    '''
    h, w = getHW(roi)
    cripAssert(h == w, 'h == w required.')
    dots = n or nextPow2(h)

    deTrend = roi - np.mean(roi)  # order 0
    dft = np.fft.fftshift(np.fft.fft2(deTrend, n=dots))
    mod2 = np.real(dft * np.conj(dft))

    return mod2 * pixelSize * pixelSize / (h * w)


def nps2DRadialAvg(roi: TwoOrThreeD, pixelSize: float, n: Or[int, None] = None):
    '''
        Compute the radially averaged noise power spectrum (NPS) of a 2D region of interest (ROI).
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
