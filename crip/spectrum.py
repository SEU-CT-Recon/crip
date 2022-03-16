# class ReadSpectrum:
import numpy as np
import re
import crip._atten


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
    path = path.join(path.dirname(path.abspath(__file__)), f'_atten/{material}_atten.txt')

    with open(path, 'r') as f:
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