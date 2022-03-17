'''
    Dual-Energy
'''

import numpy as np

from .typing import DefaultFloatDType
from .physics import Atten, DiagEnergyRange


def singleMatMuDecompose(src: Atten, basis1: Atten, basis2: Atten, energyRange=DiagEnergyRange) -> np.ndarray:
    '''
        Point-wise material decompose.
    '''
    range_ = np.array(energyRange)

    attenMu = src.mu[range_]
    attenMu1 = basis1.mu[range_]
    attenMu2 = basis2.mu[range_]

    M = np.array([attenMu1, attenMu2], dtype=DefaultFloatDType).T
    coef = np.linalg.pinv(M) @ (attenMu.T)
    return coef


def singleMaterialMuDecomposeRatio(material, rho, material_1, rho_1, material_2, rho_2, energyRange: tuple) -> np.array:
    '''
        Point-wise material decompose.
        All attenuation have the same weight.
    '''
    # whats this?

    attenMu1 = getBuiltInAtten(material_1, rho_1).mu[energyRange[0]:energyRange[1]]
    attenMu2 = getBuiltInAtten(material_2, rho_2).mu[energyRange[0]:energyRange[1]]
    attenMu = getBuiltInAtten(material, rho).mu[energyRange[0]:energyRange[1]]

    vector = np.array([attenMu1, attenMu2], dtype=float) / attenMu
    attenMu = attenMu / attenMu
    coef = attenMu @ vector.T @ np.linalg.inv(vector @ vector.T)
    return coef


def deDecomposeProjDomainCoeff():
    pass


def deDecomposeProjDomain():
    pass


def deDecomposeImageDomain():
    pass
