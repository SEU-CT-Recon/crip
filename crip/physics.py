'''
    Physics module of crip.

    by z0gSh1u @ https://github.com/z0gSh1u/crip
'''

# Attenuation Coefficient of Water, Liquid
AttenWaterText = '''
https://physics.nist.gov/PhysRefData/XrayMassCoef/tab4.html
Water, Liquid
ASCII format
_________________________________

  Energy        μ/ρ        μen/ρ 
   (MeV)      (cm2/g)     (cm2/g)
_________________________________

1.00000E-03  4.078E+03  4.065E+03 
1.50000E-03  1.376E+03  1.372E+03 
2.00000E-03  6.173E+02  6.152E+02 
3.00000E-03  1.929E+02  1.917E+02 
4.00000E-03  8.278E+01  8.191E+01 
5.00000E-03  4.258E+01  4.188E+01 
6.00000E-03  2.464E+01  2.405E+01 
8.00000E-03  1.037E+01  9.915E+00 
1.00000E-02  5.329E+00  4.944E+00 
1.50000E-02  1.673E+00  1.374E+00 
2.00000E-02  8.096E-01  5.503E-01 
3.00000E-02  3.756E-01  1.557E-01 
4.00000E-02  2.683E-01  6.947E-02 
5.00000E-02  2.269E-01  4.223E-02 
6.00000E-02  2.059E-01  3.190E-02 
8.00000E-02  1.837E-01  2.597E-02 
1.00000E-01  1.707E-01  2.546E-02 
1.50000E-01  1.505E-01  2.764E-02 
2.00000E-01  1.370E-01  2.967E-02 
3.00000E-01  1.186E-01  3.192E-02 
4.00000E-01  1.061E-01  3.279E-02 
5.00000E-01  9.687E-02  3.299E-02 
6.00000E-01  8.956E-02  3.284E-02 
8.00000E-01  7.865E-02  3.206E-02 
1.00000E+00  7.072E-02  3.103E-02 
1.25000E+00  6.323E-02  2.965E-02 
1.50000E+00  5.754E-02  2.833E-02 
2.00000E+00  4.942E-02  2.608E-02 
3.00000E+00  3.969E-02  2.281E-02 
4.00000E+00  3.403E-02  2.066E-02 
5.00000E+00  3.031E-02  1.915E-02 
6.00000E+00  2.770E-02  1.806E-02 
8.00000E+00  2.429E-02  1.658E-02 
1.00000E+01  2.219E-02  1.566E-02 
1.50000E+01  1.941E-02  1.441E-02 
2.00000E+01  1.813E-02  1.382E-02 
'''
RhoWater = 1.0  # g/cm^3
DiagnosticEnergyRange = (10, 150)
DiagEnergyLow = 0
DiagEnergyHigh = 150
DiagEnergyRange = (DiagEnergyLow, DiagEnergyHigh + 1)  # [low, high)
DiagEnergyLen = DiagEnergyHigh - DiagEnergyLow + 1

import numpy as np
import re

from .utils import cripAssert, isInt, isList


class MaterialAtten:
    def __init__(self, name, array, rho) -> None:
        cripAssert(rho > 0, '`rho` should > 0.')
        if isList(array):
            array = np.array(array)
        array = array.squeeze()

        cripAssert(len(array) == DiagEnergyLen, '`array` should have same length as energy range.')

        self.name = name
        self.array = array
        self.rho = rho
    
    



def readAtten(content, rho):
    """
        Read NIST ASCII format attenuation coefficient list. `rho` in (g/cm^3). \\
        Returns energy (MeV) and corresponding \mu value (mm^-1).
    """
    # Ignore all lines starts with non-digit.
    content = list(map(lambda x: x.strip(), content.replace('\r\n', '\n').split('\n')))
    content = list(filter(lambda y: len(y) > 0 and str.isdigit(y[0]), content))

    def procAttenLine(line: str):
        energy, muDivRho, _ = tuple(map(float, re.split(r'\s+', line)))
        return energy, muDivRho

    content = np.array(list(map(procAttenLine, content)), dtype=float)
    attenEnergy, attenMuDivRho = content.T
    attenEnergy = attenEnergy * 1000  # to keV, 1 MeV = 1000 keV
    attenMu = attenMuDivRho * rho * 0.1  # to mu mm^-1

    return attenEnergy, attenMu


def readSpectrum(content, unit='keV'):
    """
        Read spectrum list whose unit of energy is keV. Return energies in keV and omega. \\
        Set `unit` to `eV` if the unit in spectrum is `eV`.
    """
    # Ignore all lines starts with non-digit.
    content = list(
        filter(lambda y: len(y) > 0 and str.isdigit(y[0]),
               list(map(lambda x: x.strip(),
                        content.replace('\r\n', '\n').split('\n')))))

    def procSpectrumLine(line: str):
        energy, omega = tuple(map(float, re.split(r'\s+', line)))
        return energy, omega

    content = np.array(list(map(procSpectrumLine, content)), dtype=float)
    spectrumEnergy, spectrumOmega = content.T

    if unit == 'eV':
        spectrumEnergy /= 1000  # to keV

    return spectrumEnergy, spectrumOmega


def calcMuWater(spectrum, unit='keV', rhoWater=RhoWater):
    """
        Calculate \mu value (mm^-1) of water according to a specified spectrum \\
        (@see `readSpectrum` for specturm format) considering only energies in diagonstic
        range (10 ~ 150 keV).
    """
    # Read attenuation.
    attenWaterEnergy, attenWaterMu = readAtten(AttenWaterText, RhoWater)

    # We only consider energies in diagnostic range.
    attenWaterInDiag = np.intersect1d(np.argwhere(attenWaterEnergy >= DiagnosticEnergyRange[0]),
                                      np.argwhere(attenWaterEnergy <= DiagnosticEnergyRange[1]))
    attenWaterEnergy = attenWaterEnergy[attenWaterInDiag]
    attenWaterMu = attenWaterMu[attenWaterInDiag]

    # Perform log-domain interpolation.
    attenWaterEnergyDense = np.arange(DiagnosticEnergyRange[0], DiagnosticEnergyRange[1] + 1, 1)
    attenWaterMu = np.interp(np.log(attenWaterEnergyDense), np.log(attenWaterEnergy), np.log(attenWaterMu))
    attenWaterEnergy = attenWaterEnergyDense
    attenWaterMu = np.exp(attenWaterMu)

    # Read spectrum.
    spectrumEnergy, spectrumOmega = readSpectrum(spectrum, unit)

    # Intergrate along energies.
    sumOmegaMuWater = 0.0
    sumOmega = 0.0
    for idx, E in enumerate(spectrumEnergy):
        E = int(E)
        # We only consider energies in diagnostic range.
        if E >= DiagnosticEnergyRange[0] and E <= DiagnosticEnergyRange[1]:
            sumOmegaMuWater += spectrumOmega[idx] * attenWaterMu[np.argwhere(attenWaterEnergy == E)]
            sumOmega += spectrumOmega[idx]

    muWaterRef = sumOmegaMuWater / sumOmega  # mm^-1
    return muWaterRef.squeeze().squeeze()
