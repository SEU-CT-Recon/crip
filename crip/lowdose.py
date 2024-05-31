'''
    Low Dose CT module of crip.

    https://github.com/SEU-CT-Recon/crip
'''

import numpy as np

from ._typing import *
from .shared import *
from .utils import *


@ConvertListNDArray
def injectGaussianNoise(projections: TwoOrThreeD,
                        sigma: float,
                        mu: float = 0,
                        clipMinMax: Or[None, Tuple[float, float]] = None) -> TwoOrThreeD:
    ''' Inject additive Gaussian noise ~ `N(mu, sigma^2)` where `sigma` is the standard deviation and `mu` is the mean.
        use `clipMinMax = [min, max]` to clip the noisy projections.
    '''
    cripAssert(is2or3D(projections), f'`projections` should be 2 or 3-D, but got {projections.ndim}-D.')
    cripAssert(sigma > 0, 'sigma should be greater than 0.')

    _inject1 = lambda img: (np.random.randn(*img.shape) * sigma + mu) + img
    if is3D(projections):
        res = np.array(list(map(_inject1, projections)))
    else:
        res = _inject1(projections)

    if clipMinMax is not None:
        cripAssert(len(clipMinMax) == 2, 'Invalid `clipMinMax`.')
        res = np.clip(res, *clipMinMax)

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

    img = nPhoton * img  # N0 exp(-\sum \mu L), i.e., ground truth quanta
    img = np.random.poisson(img.astype(np.uint32)).astype(DefaultFloatDType)  # noisy quanta
    img[img <= 0] = 1
    img /= nPhoton  # cancel the rescaling from N0

    if type_ == 'postlog':
        img = -np.log(img)

    return img


@ConvertListNDArray
def totalVariation(img: TwoOrThreeD) -> Or[float, NDArray]:
    ''' Computes the Total Variation (TV) of images.
        For 2D image, it returns a scalar.
        For 3D image, it returns an array of TV values for each slice.
    '''
    cripAssert(is2or3D(img), 'img should be 2 or 3D.')

    vX = img[..., :, 1:] - img[..., :, :-1]
    vY = img[..., 1:, :] - img[..., :-1, :]

    axis = (-2, -1)
    tv = np.sum(np.abs(vX), axis=axis) + np.sum(np.abs(vY), axis=axis)

    return tv


def nps2D(roi: TwoOrThreeD,
          pixelSize: float,
          detrend: Or[str, None] = 'individual',
          n: Or[int, None] = None,
          fftshift: bool = True,
          normalize: Or[None, str] = None) -> TwoD:
    ''' Compute the noise power spectrum (NPS) of a 2D square ROI using DFT.
        `pixelSize` is the pixel size of reconstructed image ([mm] recommended).
        `detrend` method can be `individual` (by mean value subtraction), `mutual` (by foreground subtraction) or None.
        `normalize` method can be `sum` (by amp. sum), `max` (by amp. max) or None.
        `fftshift` is used to shift the zero frequency to the center.
        `n` is the number of dots in DFT, if not provided, it will be the next power of 2 of the ROI size.
        Usually, the ROI should be a uniform region, and multiple realizations are recommended.
        The output NPS unit is usually recognized as [a.u.], and x,y-coordinates correspond to
        physical location `coord*pixelSize`.

        [1] https://amos3.aapm.org/abstracts/pdf/99-28842-359478-110263-658667764.pdf
    '''
    cripAssert(detrend in ['individual', 'mutual', None], f'Invalid detrend method: {detrend}.')
    cripAssert(normalize in ['sum', 'max', None], f'Invalid normalize method: {normalize}.')
    cripWarning(is3D(roi), "It's highly recommended to provide multiple realizations of the ROI.")
    if detrend == 'mutual':
        cripAssert(is3D(roi), '`mutual` detrend method requires multiple realizations of the ROI.')

    roi = as3D(roi)
    h, w = getHnW(roi)
    cripAssert(h == w, 'ROI should be square.')
    dots = n or nextPow2(h)

    # de-trend the signal
    if detrend == 'individual':
        detrended = np.zeros_like(roi)
        for i in range(roi.shape[0]):
            detrended[i] = roi[i] - np.mean(roi[i])  # (DC+noise)-DC
        s = 1
    elif detrend == 'mutual':
        detrended = np.zeros((roi.shape[0] - 1, h, w))
        for i in range(1, roi.shape[0]):
            detrended[i - 1] = roi[i, ...] - roi[i - 1, ...]  # (DC+noise)-(DC+noise)
        s = 1 / 2
    elif detrend is None:
        detrended = roi.copy()
        s = 1

    dft = np.fft.fft2(detrended, s=(dots, dots))
    dft = np.mean(dft, axis=0)  # averaged NPS
    if fftshift:
        dft = np.fft.fftshift(dft)

    mod2 = np.abs(dft)**2  # square of modulus
    nps = (pixelSize * pixelSize) / (h * w) * mod2 * s
    if normalize == 'sum':
        nps /= nps.sum()
    elif normalize == 'max':
        nps /= nps.max()

    return nps


def nps2DRadAvg(nps: TwoD, fftshifted: bool = True, normalize: Or[str, None] = None) -> NDArray:
    ''' Compute the radially averaged noise power spectrum (NPS) where `nps` can be the output from 
        `nps2D` (unnormalized, fftshifted). Do not normalize the input `nps` before using this function.
        The output is a 1D array of NPS values [a.u.] and x-axis is the spatial frequency [1/[unit of pixelSize]].
    '''
    cripAssert(is2D(nps), '`nps` should be 2D.')
    cripAssert(nps.shape[0] == nps.shape[1], '`nps` should be square.')
    cripAssert(normalize in ['sum', 'max', None], f'Invalid normalize method: {normalize}.')
    if nps.max() <= 1:
        cripWarning(False, 'The input `nps` looks to be normalized already, which is not expected.')
    if not fftshifted:
        nps = np.fft.fftshift(nps)

    N = nps.shape[0]
    x, y = np.meshgrid(np.arange(N), np.arange(N))
    R = np.sqrt(x**2 + y**2)

    _f = lambda r: nps[(R >= r - .5) & (R < r + .5)].mean()
    _args = np.linspace(1, N, num=N)
    res = np.vectorize(_f)(_args)

    if normalize == 'sum':
        res /= res.sum()
    elif normalize == 'max':
        res /= res.max()

    return res
