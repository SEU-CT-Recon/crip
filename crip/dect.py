'''
    Dual-Energy
'''

import numpy as np


def muDecomposeSingleMaterial(srcAtten, basis1Atten, basis2Atten):
    srcMu = srcAtten.getArray().T
    mu1 = basis1Atten.getArray().T
    mu2 = basis2Atten.getArray().T

    M = np.array([mu1, mu2]).T
    MPinv = np.linalg.pinv(M)
    return MPinv * srcMu # a1, a2

def deDecomposeProjDomainCoeff():
    pass

def deDecomposeProjDomain():
    pass




def deDecomposeImageDomain():
    pass




