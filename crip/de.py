'''
    Dual-Energy CT module of crip.

    https://github.com/z0gSh1u/crip
'''

__all__ = ['singleMatMuDecomp', 'calcAttenedSpec', 'calcPostLog', 'deDecompGetCoeff', 'deDecompProj', 'deDecompRecon']

import numpy as np

from .utils import ConvertListNDArray, cripAssert, cripWarning, is2D, isOfSameShape
from ._typing import *
from .physics import Atten, DiagEnergyRange, Spectrum, calcAttenedSpec, calcPostLog


def singleMatMuDecomp(src: Atten, base1: Atten, base2: Atten, method='coeff', energyRange=DiagEnergyRange) -> NDArray:
    '''
        Decompose single material `src`'s attenuation onto `base1` and `base2`.

        Return decomposing coefficients along energies when `method = 'coeff'`, or proportion when `method = 'prop'`.
    '''
    cripAssert(method in ['coeff', 'prop'], 'Invalid method.')

    range_ = np.array(energyRange)
    srcMu = src.mu[range_]
    baseMu1 = base1.mu[range_]
    baseMu2 = base2.mu[range_]

    M = np.array([baseMu1, baseMu2], dtype=DefaultFloatDType).T

    if method == 'prop':
        M /= srcMu
        srcMu = np.ones_like(srcMu)

    res = np.linalg.pinv(M) @ (srcMu.T)

    return res


def deDecompGetCoeff(lowSpec: Spectrum, highSpec: Spectrum, base1: Atten, len1: Or[NDArray, Iterable], base2: Atten,
                     len2: Or[NDArray, Iterable]):
    '''
        Calculate the decomposing coefficient (Order 2 with bias term) of two spectra onto two material bases.
    '''
    lenCombo = []
    postlogLow = []
    postlogHigh = []
    for i in len1:
        for j in len2:
            lenCombo.append([i, j])
            postlogLow.append(calcPostLog(lowSpec, [base1, base2], [i, j]))
            postlogHigh.append(calcPostLog(highSpec, [base1, base2], [i, j]))

    def deCalcBetaGamma(A1, A2, LComb):
        A1Square = A1.T**2
        A2Square = A2.T**2
        A1A2 = (A1 * A2).T
        A1 = A1.T
        A2 = A2.T
        Ones = np.ones((A1.T.shape[0]))
        A = np.array([A1Square, A2Square, A1A2, A1, A2, Ones]).T

        return np.linalg.pinv(A) @ LComb

    beta, gamma = deCalcBetaGamma(np.array(postlogLow), np.array(postlogHigh), np.array(lenCombo)).T

    return beta, gamma


@ConvertListNDArray
def deDecompProj(lowProj: TwoOrThreeD, highProj: TwoOrThreeD, coeff1: NDArray,
                 coeff2: NDArray) -> Tuple[TwoOrThreeD, TwoOrThreeD]:
    '''
        Perform dual-energy decompose in projection domain point-by-point using coeffs.

        Coefficients can be generated using @see `deDecompGetCoeff`.
    '''
    cripAssert(isOfSameShape(lowProj, highProj), 'Two projection sets should have same shape.')
    cripAssert(
        len(coeff1) == 6 and len(coeff2) == 6,
        'Decomposing coefficients should have length 6 (2 variable, order 2 with bias).')

    def applyPolyV2L2(coeff, A1, A2):
        return coeff[0] * A1**2 + coeff[1] * A2**2 + coeff[2] * A1 * A2 + coeff[3] * A1 + coeff[4] * A2 + coeff[5]

    return applyPolyV2L2(coeff1, lowProj, highProj), applyPolyV2L2(coeff2, lowProj, highProj)


@ConvertListNDArray
def deDecompRecon(low: TwoOrThreeD, high: TwoOrThreeD, muBase1Low: float, muBase1High: float, muBase2Low: float,
                  muBase2High: float, checkCond: bool = True):
    '''
        Perform dual-energy decompose in reconstruction domain. \\mu values can be calculated using @see `calcMu`.

        The values of input volumes should be \\mu value. The outputs are decomposing coefficients.
    '''
    cripAssert(isOfSameShape(low, high), 'Two volumes should have same shape.')
    COND_TOLERANCE = 1000

    A = np.array([[muBase1Low, muBase2Low], [muBase1High, muBase2High]])
    if checkCond:
        cripWarning(np.linalg.cond(A) <= COND_TOLERANCE, 'The material decomposition matrix possesses high condition number.')
    M = np.linalg.inv(A)

    def decompOne(low, high):
        c1 = M[0, 0] * low + M[0, 1] * high
        c2 = M[1, 0] * low + M[1, 1] * high
        return c1, c2

    if is2D(low):
        return decompOne(low, high)
    else:
        return np.array(list(map(lambda args: decompOne(*args), zip(low, high)))).transpose((1, 0, 2, 3))
