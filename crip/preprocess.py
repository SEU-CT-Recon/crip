'''
    Preprocess module of crip.

    https://github.com/SEU-CT-Recon/crip
'''

import numpy as np
import warnings
from scipy.interpolate import interpn
import cv2

from .shared import *
from ._typing import *
from .utils import *

warnings.simplefilter("ignore", DeprecationWarning)


@ConvertListNDArray
def averageProjections(projections: ThreeD) -> TwoD:
    ''' Average projections along the first axis (views) to get a single projection.
    '''
    cripAssert(is3D(projections), f'`projections` should be 3D, but got {len(projections.shape)}-D.')

    projections = asFloat(projections)
    res = projections.sum(axis=0) / projections.shape[0]

    return res


@ConvertListNDArray
def correctFlatDarkField(projections: TwoOrThreeD,
                         flat: Or[TwoD, ThreeD, float, None] = None,
                         dark: Or[TwoD, ThreeD, float] = 0,
                         fillNaN: Or[float, None] = 0,
                         fillInf: Or[float, None] = 0) -> TwoOrThreeD:
    ''' Perform flat field (air) and dark field correction to get post-log projections.
        I.e., `- log [(X - D) / (F - D)]`.
        Usually `flat` and `dark` are 2D.
        If `flat` and `projections` are both 3D, perform view by view.
        If `flat` is None, estimate it by brightest pixel each view.
    '''
    if flat is None:
        cripWarning(False, '`flat` is None. Use the maximum value of each view instead.')
        flat = np.max(projections, axis=0) * np.ones_like(projections)

    sampleProjection = projections if is2D(projections) else projections[0]

    def checkShape(haystack, needle, needleName):
        cripAssert(
            isOfSameShape(haystack, needle),
            f'`projections` and `{needleName}` should have same shape, but got {haystack.shape} and {needle.shape}.')

    checkShape(sampleProjection if is2D(flat) else projections, flat, 'flat')
    checkShape(sampleProjection if is2D(dark) else projections, dark, 'dark')

    numerator = projections - dark
    denominator = flat - dark
    cripAssert(np.min(numerator > 0), 'Some `projections` values are not greater than zero after canceling `dark`.')
    cripAssert(np.min(denominator > 0), 'Some `flat` values are not greater than zero after canceling `dark`.')

    res = -np.log(numerator / denominator)

    if fillInf is not None:
        res[res == np.inf] = fillInf
    if fillNaN is not None:
        res[res == np.nan] = fillNaN

    return res


@ConvertListNDArray
def projectionsToSinograms(projections: ThreeD):
    ''' Permute projections to sinograms by axes swapping `(views, h, w) -> (h, views, w)`.
        The width direction is along detector channels of a row.
    '''
    cripAssert(is3D(projections), 'projections should be 3D.')

    (views, h, w) = projections.shape
    sinograms = np.zeros((h, views, w), dtype=projections.dtype)
    for i in range(views):
        sinograms[:, i, :] = projections[i, :, :]

    return sinograms


@ConvertListNDArray
def sinogramsToProjections(sinograms: ThreeD):
    ''' Permute sinograms back to projections by axes swapping `(h, views, w) -> (views, h, w)`.
        The width direction is along detector channels of a row.
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

    h, w = getHnW(img)
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
def correctRingArtifactInProj(sgm: TwoOrThreeD, sigma: float, ksize: Or[int, None] = None):
    '''
        Apply the ring artifact correction method in projection domain (input postlog sinogram),
        using gaussian filter in sinogram detector direction [1].
        [1] 李俊江,胡少兴,李保磊等.CT图像环状伪影校正方法[J].北京航空航天大学学报,2007,(11):1378-1382.DOI:10.13700/j.bh.1001-5965.2007.11.020
        https://kns.cnki.net/kcms2/article/abstract?v=xBNwvqFr00JMj5WzBbZMcj9N-kBm9Pi08enuNi8ymtGWVZuwGHdLWwttkgzSivkJSBVk0CVQZxo7DgBmujqhLCaVvvBMoig5RV0B4fDLUnjGQcCPo3O4KfAo4iX4vCwZ&uniplatform=NZKPT&flag=copy
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


def fanToPara(sgm: TwoD, gammas: NDArray, betas: NDArray, sid: float, oThetas: Tuple[float],
              oLines: Tuple[float]) -> TwoD:
    '''
        Re-order a Fan-Beam sinogram to Parallel-Beam's.
        `gammas` is fan angles from min to max [RAD], computed by `arctan(elementOffcenter / SDD)` for each element.
        `betas` is system rotation angles from min to max [RAD].
        `sid` is Source-Isocenter-Distance [mm].
        `oThetas` is output rotation angle range (min, delta, max) tuple [RAD]
        `oLines` is output detector element physical locations range (min, delta, max) tuple [mm], e.g., `elementOffcenter` array
        ```
               /| <- gamma for detector element X
              / | <- SID
             /  |
        ====X============ <- detector
            ^<--^ <- offcenter
    '''
    nThetas = np.round((oThetas[2] - oThetas[0]) / oThetas[1]).astype(np.int32)
    nLines = np.round((oLines[2] - oLines[0]) / oLines[1]).astype(np.int32)

    thetas1 = np.linspace(oThetas[0], oThetas[2], nThetas).reshape((1, -1))
    R1 = np.linspace(oLines[0], oLines[2], nLines).reshape((1, -1))

    thetas = thetas1.T @ np.ones((1, nLines))
    Rs = np.ones((nThetas, 1)) @ R1

    oGammas = np.arcsin(Rs / sid)
    oBetas = thetas + oGammas - np.pi / 2

    minBetas, maxBetas = np.min(betas), np.max(betas)
    inds = oBetas > maxBetas
    if len(inds) > 0:
        oBetas[inds] = oBetas[inds] - 2 * np.pi
    inds = oBetas < minBetas
    if len(inds) > 0:
        oBetas[inds] = oBetas[inds] + 2 * np.pi
    oBetas = np.minimum(oBetas, maxBetas)

    interpolator = interpn((betas, gammas), sgm, (oBetas, oGammas), method='linear', bounds_error=False, fill_value=0)
    out = interpolator.reshape(oBetas.shape)

    return out
