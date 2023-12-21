'''
    Preprocess module of crip.

    https://github.com/z0gSh1u/crip
'''

__all__ = [
    'averageProjections', 'flatDarkFieldCorrection', 'projectionsToSinograms', 'sinogramsToProjections', 'padImage',
    'padSinogram', 'correctBeamHardeningPolynomial', 'injectGaussianNoise', 'injectPoissonNoise', 'binning', 'fan2para',
    'correctRingArtifactInProj'
]

import numpy as np
import warnings

warnings.simplefilter("ignore", DeprecationWarning)

from scipy.interpolate import interp2d
import cv2
from .shared import *
from ._typing import *
from .utils import *
from .lowdose import injectGaussianNoise, injectPoissonNoise


@ConvertListNDArray
def averageProjections(projections: TwoOrThreeD) -> TwoD:
    '''
        Average projections. For example, to calculate the flat field.
        Projections can be either `(views, H, W)` shaped numpy array, or
        `views * (H, W)` Python List.
    '''
    cripAssert(is3D(projections), '`projections` should be 3D array.')
    projections = asFloat(projections)

    res = projections.sum(axis=0) / projections.shape[0]

    return res


@ConvertListNDArray
def flatDarkFieldCorrection(projections: TwoOrThreeD,
                            flat: Or[TwoD, ThreeD, float, None] = None,
                            dark: Or[TwoD, ThreeD, float] = 0) -> TwoOrThreeD:
    '''
        Perform flat field (air) and dark field correction to get post-log value.
        If `flat` is 3D and `projections` is also 3D, the correction will be performed view by view.
        I.e., `- log [(X - D) / (F - D)]`. Multi projections accepted.
        If flat is None, air is estimated using the brightest pixel by default.
    '''
    sampleProjection = projections if is2D(projections) else projections[0]

    if is2D(flat):
        cripAssert(isOfSameShape(sampleProjection, flat), '`projection` and `flat` should have same shape.')
    if is3D(flat):
        cripAssert(isOfSameShape(projections, flat), '`projection` and `flat` should have same shape.')
    if is2D(dark):
        cripAssert(isOfSameShape(sampleProjection, dark), '`projection` and `dark` should have same shape.')
    if is3D(dark):
        cripAssert(isOfSameShape(projections, dark), '`projection` and `dark` should have same shape.')
    if flat is None:
        cripWarning(False, '`flat` is None. Use the maximum value instead.')
        flat = np.max(projections)

    numerator = projections - dark
    denominator = flat - dark
    cripAssert(np.min(numerator > 0), 'Some projections values are not greater than zero.')
    cripAssert(np.min(denominator > 0), 'Some flat field values are not greater than zero.')

    res = -np.log(numerator / denominator)
    res[res == np.inf] = 0
    res[res == np.nan] = 0

    return res


@ConvertListNDArray
def projectionsToSinograms(projections: ThreeD):
    '''
        Permute projections to sinograms by axes swapping `(views, h, w) -> (h, views, w)`.

        Note that the width direction is along detector channels of a row.
    '''
    cripAssert(is3D(projections), 'projections should be 3D.')

    (views, h, w) = projections.shape
    sinograms = np.zeros((h, views, w), dtype=projections.dtype)
    for i in range(views):
        sinograms[:, i, :] = projections[i, :, :]

    return sinograms


@ConvertListNDArray
def sinogramsToProjections(sinograms: ThreeD):
    '''
        Permute sinograms back to projections by axes swapping `(h, views, w) -> (views, h, w)`.

        Note that the width direction is along detector channels of a row.
    '''
    cripAssert(is3D(sinograms), 'projections should be 3D.')

    (h, views, w) = sinograms.shape
    projections = np.zeros((views, h, w), dtype=sinograms.dtype)
    for i in range(views):
        projections[i, :, :] = sinograms[:, i, :]

    return projections


@ConvertListNDArray
def padImage(img: TwoOrThreeD,
             padding: Tuple[int, int, int, int],
             mode: str = 'symmetric',
             decay: Or[str, None] = None):
    '''
        Pad the image on four directions using symmetric `padding` (Up, Right, Down, Left). \\
        `mode` determines the border value, can be `symmetric`, `edge`, `constant` (zero), `reflect`. \\
        `decay` can be None, `cosine`, `smoothCosine` to perform a decay on padded border.
    '''
    cripAssert(mode in ['symmetric', 'edge', 'constant', 'reflect'], f'Invalid mode: {mode}.')
    cripAssert(decay in [None, 'cosine', 'smoothCosine'], f'Invalid decay: {decay}.')

    decays = {
        'cosine':
        lambda ascend, dot: np.cos(np.linspace(-np.pi / 2, 0, dot) if ascend else np.linspace(0, np.pi / 2, dot)),
        'smoothCosine':
        lambda ascend, dot: 0.5 * np.cos(np.linspace(-np.pi, 0, dot)) + 0.5
        if ascend else 0.5 * np.cos(np.linspace(0, np.pi, dot)) + 0.5
    }

    h, w = getHW(img)
    nPadU, nPadR, nPadD, nPadL = padding
    padH = h + nPadU + nPadD
    padW = w + nPadL + nPadR

    def decayLR(xPad, w, nPadL, nPadR, decay):
        xPad[:, 0:nPadL] *= decay(True, nPadL)[:]
        xPad[:, w - nPadR:w] *= decay(False, nPadR)[:]
        return xPad

    def procOne(img):
        xPad = np.pad(img, ((nPadU, nPadD), (nPadL, nPadR)), mode=mode)
        if decay is not None:
            xPad = decayLR(xPad, padW, nPadL, nPadR, decays[decay])
            xPad = decayLR(xPad.T, padH, nPadU, nPadD, decays[decay])
            xPad = xPad.T
        return xPad

    if is3D(img):
        res = [procOne(img[i, ...]) for i in range(img.shape[0])]
        return np.array(res)
    else:
        return procOne(img)


@ConvertListNDArray
def padSinogram(sgms: TwoOrThreeD, padding: Or[int, Tuple[int, int]], mode='symmetric', decay='smoothCosine'):
    '''
        Pad sinograms in width direction (same line detector elements) using `mode` and `decay`\\
        with `padding` (single int, or (right, left)).

        @see padImage for parameter details.
    '''
    if isType(padding, int):
        padding = (padding, padding)

    l, r = padding

    return padImage(sgms, (0, r, 0, l), mode, decay)


@ConvertListNDArray
def correctBeamHardeningPolynomial(postlog: TwoOrThreeD, coeffs: Or[Tuple, np.poly1d], bias=True):
    '''
        Apply the polynomial (\\mu L vs. PostLog fit) on postlog to perform basic beam hardening correction.
        `coeffs` can be either `tuple` or `np.poly1d`. Set `bias=True` if your coeffs includes the bias (order 0) term.
    '''
    cripWarning(isType(coeffs, np.poly1d) and bias is False, 'When using np.poly1d as coeffs, bias is always True.')

    if isType(coeffs, Tuple):
        if bias is False:
            coeffs = np.poly1d([*coeffs, 0])

    return coeffs(postlog)


@ConvertListNDArray
def correctRingArtifactInProj(sgm: TwoOrThreeD, sigma: float, ksize: Or[int, None] = None):
    '''
        Apply the ring artifact correction method in projection domain (input postlog sinogram),
        using gaussian filter in sinogram detector direction [1].
        [1] DOI:10.13700/j.bh.1001-5965.2007.11.020.
    '''
    ksize = ksize or int(2 * np.ceil(2 * sigma) + 1)
    kernel = np.squeeze(cv2.getGaussianKernel(ksize, sigma))
    
    def procOne(sgm: TwoD):
        Pc = np.mean(sgm, axis=0)
        Rc = np.convolve(Pc, kernel, mode='same')
        Ec = Pc - Rc
        return sgm - Ec[np.newaxis, :]
        
    if is3D(sgm):
        res = np.array(list(map(procOne, sgm)))
    else:
        res = procOne(sgm)

    return res


def fan2para(sgm: TwoD, gammas, betas, d, oThetas, oLines):
    '''
        Re-order Fan-Beam sinogram to Parallel-Beam's.
        gammas: the fan angles from min to max
        betas: the rotation angles from min to max
        d: Source-Isocenter-Distance
        oThetas: output rotation angle [min, delta, max]
        oLines: output detector element physical locs [min, delta, max]
    '''
    nThetas = np.round((oThetas[2] - oThetas[0]) / oThetas[1]).astype(np.int32)
    nLines = np.round((oLines[2] - oLines[0]) / oLines[1]).astype(np.int32)

    thetas1 = np.linspace(oThetas[0], oThetas[2], nThetas).reshape((1, -1))
    R1 = np.linspace(oLines[0], oLines[2], nLines).reshape((1, -1))

    thetas = thetas1.T @ np.ones((1, nLines))
    Rs = np.ones((nThetas, 1)) @ R1

    oGammas = np.arcsin(Rs / d)
    oBetas = thetas + oGammas - np.pi / 2

    minBetas, maxBetas = np.min(betas), np.max(betas)
    inds = oBetas > maxBetas
    if len(inds) > 0:
        oBetas[inds] = oBetas[inds] - 2 * np.pi
    inds = oBetas < minBetas
    if len(inds) > 0:
        oBetas[inds] = oBetas[inds] + 2 * np.pi
    oBetas = np.minimum(oBetas, maxBetas)

    func = interp2d(gammas, betas, sgm, 'linear', fill_value=0)

    h, w = oGammas.shape
    out = np.zeros_like(oGammas)
    for i in range(h):
        for j in range(w):
            out[i, j] = func(oGammas[i, j], oBetas[i, j])

    return out