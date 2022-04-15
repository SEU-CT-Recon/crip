'''
    Dual-Energy CT module of crip.

    [NOTE] This module is still under development and cannot be imported now.

    https://github.com/z0gSh1u/crip
'''

__all__ = ['singleMatMuDecompose', 'calcAttenedSpec', 'calcPostLog', 'deDecompGetCoeff', 'deDecompProj', 'deDecompRecon']

from typing import Iterable, List
import numpy as np

from .utils import cripAssert, isType
from ._typing import DefaultFloatDType, Or, NDArray
from .physics import Atten, DiagEnergyRange, Spectrum


def singleMatMuDecompose(src: Atten,
                         basis1: Atten,
                         basis2: Atten,
                         method='coeff',
                         energyRange=DiagEnergyRange) -> NDArray:
    '''
        Decompose single material `src`'s attenuation onto `basis1` and `basis2`.

        Return decomposing coefficients when `method = 'coeff'`, or proportion when `method = 'prop'`.
    '''
    cripAssert(method in ['coeff', 'prop'], 'Invalid method.')

    range_ = np.array(energyRange)
    attenMu = src.mu[range_]
    attenMu1 = basis1.mu[range_]
    attenMu2 = basis2.mu[range_]

    M = np.array([attenMu1, attenMu2], dtype=DefaultFloatDType).T

    if method == 'prop':
        M /= attenMu
        attenMu = np.ones_like(attenMu)

    res = np.linalg.pinv(M) @ (attenMu.T)

    return res


def calcAttenedSpec(spec: Spectrum, attens: Or[Atten, List[Atten]], Ls: Or[float, List[float]]) -> Spectrum:
    '''
        Calculate the attenuated spectrum using polychromatic Beer-Lambert law. Supports multiple materials.

        I.e., `\\Omega(E) \\exp (- \\mu(E) L) \\through all E`. L in mm.
    '''
    if isType(attens, Atten):
        attens = [attens]
    if isType(Ls, float):
        Ls = [Ls]
    cripAssert(len(attens) == len(Ls), 'atten should have same length as L.')

    N = len(attens)
    omega = np.array(spec.omega, copy=True)
    for i in range(N):  # materials
        atten = attens[i]
        L = Ls[i]
        for E in DiagEnergyRange:  # energies
            omega[E] *= np.exp(-atten.mu[E] * L)

    return Spectrum(omega, spec.unit)


def calcPostLog(spec: Spectrum, atten: Or[Atten, List[Atten]], L: Or[float, List[float]]) -> float:
    '''
        Calculate post-log value after attenuation of `L` length `atten`. L in mm.
    '''
    attenSpec = calcAttenedSpec(spec, atten, L)

    return -np.log(attenSpec.sumOmega / spec.sumOmega)


def deDecompGetCoeff(lowSpec: Spectrum, highSpec: Spectrum, basis1: Atten, len1: Or[NDArray, Iterable], basis2: Atten,
                     len2: Or[NDArray, Iterable]):
    '''
        Calculate the decomposing coefficient (Order 2) of two spectra onto to material bases.
    '''
    lenCombo = []
    postlogLow = []
    postlogHigh = []
    for i in len1:
        for j in len2:
            lenCombo.append([i, j])
            postlogLow.append(calcPostLog(lowSpec, [basis1, basis2], [i, j]))
            postlogHigh.append(calcPostLog(highSpec, [basis1, basis2], [i, j]))
    lenCombo = np.array(lenCombo)
    postlogLow = np.array(postlogLow)
    postlogHigh = np.array(postlogHigh)

    def deCalcBetaGamma(A1, A2, LComb):
        A1Square = A1.T**2
        A2Square = A2.T**2
        A1A2 = (A1 * A2).T
        A1 = A1.T
        A2 = A2.T
        Ones = np.ones((A1.T.shape[0]))
        A = np.array([A1Square, A2Square, A1A2, A1, A2, Ones]).T

        return np.linalg.pinv(A) @ LComb

    beta, gamma = deCalcBetaGamma(postlogLow, postlogHigh, lenCombo)

    return beta, gamma


def deDecompProj():
    '''
        Perform dual-energy decompose in projection domain.
    '''
    cripAssert(False, 'Unimplemented.')  


def deDecompRecon():
    '''
        Perform dual-energy decompose in reconstruction domain.
    '''
    cripAssert(False, 'Unimplemented.')  
