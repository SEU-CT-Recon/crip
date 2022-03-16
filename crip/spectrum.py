# class ReadSpectrum:
import numpy as np
import re
from crip.io import *


class Spectrum:
    def __init__(self, spectrum, unit='keV'):
        self.spectrum = spectrum
        self.unit = unit
        self.spectrumEnergy = np.zeros(150, dtype=float)
        self.spectrumOmega = np.zeros(150, dtype=float)
        self.read()

    def read(self):
        """
            Read spectrum list whose unit of energy is keV. Return energies in keV and omega. \\
            Set `unit` to `eV` if the unit in spectrum is `eV`.
            Return energy range [1, 149], len = 149
        """
        # Ignore all lines starts with non-digit.
        with open(self.spectrum, 'r') as f:
            content = f.read()
            content = list(
                filter(lambda y: len(y) > 0 and str.isdigit(y[0]),
                       list(map(lambda x: x.strip(),
                                content.replace('\r\n', '\n').split('\n')))))

            def procSpectrumLine(line: str):
                energy, omega = tuple(map(float, re.split(r'\s+', line)))
                return energy, omega

            content = np.array(list(map(procSpectrumLine, content)), dtype=float)
            spectrumEnergy, spectrumOmega = content.T

        if self.unit == 'eV':
            spectrumEnergy /= 1000  # to keV

        self.startVoltage, self.cutOffVoltage = int(spectrumEnergy[0]), int(spectrumEnergy[-1])
        self.spectrumOmega[self.startVoltage: self.cutOffVoltage] = spectrumOmega[:-1]  # drop -1

        return self.spectrumOmega[1:]


def getAtten(material: str, rho: float, energy: np.array) -> np.array:
    """
        Read NIST or ICRP110 ASCII format attenuation coefficient list. `rho` in (g/cm^3). \\
        Return \mu value (mm^-1).
    """
    from os import path
    path = path.join(path.dirname(path.abspath(__file__)), '_atten')
    assert getFileList(material, path, '.txt') != [], 'Material not found!'

    with open(*getFileList(material, path, '.txt'), 'r') as f:
        content = f.read()
    content = list(map(lambda x: x.strip(), content.replace('\r\n', '\n').split('\n')))
    content = list(filter(lambda y: len(y) > 0 and str.isdigit(y[0]), content))

    def procAttenLine(line: str):
        parameter = tuple(map(float, re.split(r'\s+', line)))
        return parameter[0], parameter[1]  # attenEnergy, attenMuDivRho

    content = np.array(list(map(procAttenLine, content)), dtype=float)
    attenEnergy, attenMuDivRho = content.T
    attenEnergy = attenEnergy * 1000  # to keV, 1 MeV = 1000 keV

    # Perform log-domain interpolation.
    attenInterpMuDivRho = np.interp(np.log(energy), np.log(attenEnergy), np.log(attenMuDivRho))
    attenNewMu = 0.1 * rho * np.exp(attenInterpMuDivRho)  # value (mm^-1)

    return attenNewMu


def spectrumAtten(material: str, rho: float, spectrumOmega: np.array, E=1) -> float:
    """
        Generate material attention reference under specific spectrum.
        E=1: Flat Panel Detector
        E=np.linspace(1, 149, 149): Photon Counting Detector
        Return \mu value (mm^-1).
    """
    attenMu = getAtten(material, rho, np.linspace(1, 149, 149))
    return sum(spectrumOmega * E * attenMu) / sum(spectrumOmega * E)


def decompose(material, rho, material_1, rho_1, material_2, rho_2, energy: np.array) -> np.array:
    """
        Point-wise material decompose.
    """
    attenMu1 = getAtten(material_1, rho_1, energy)
    attenMu2 = getAtten(material_2, rho_2, energy)
    attenMu = getAtten(material, rho, energy)

    vector = np.array([attenMu1, attenMu2], dtype=float)
    coef = attenMu @ vector.T @ np.linalg.inv(vector @ vector.T)
    return coef


def decomposeRatio(material, rho, material_1, rho_1, material_2, rho_2, energy: np.array) -> np.array:
    """
        Point-wise material decompose.
        All attenuation have the same weight.
    """
    attenMu1 = getAtten(material_1, rho_1, energy)
    attenMu2 = getAtten(material_2, rho_2, energy)
    attenMu = getAtten(material, rho, energy)

    vector = np.array([attenMu1, attenMu2], dtype=float) / attenMu
    attenMu = attenMu / attenMu
    coef = attenMu @ vector.T @ np.linalg.inv(vector @ vector.T)
    return coef