'''
    Physics module of crip.

    by z0gSh1u @ https://github.com/z0gSh1u/crip
'''

RhoWater = 1.0  # g/cm^3
DiagnosticEnergyRange = (10, 150)
DiagEnergyLow = 0
DiagEnergyHigh = 150
DiagEnergyRange = (DiagEnergyLow, DiagEnergyHigh + 1)  # [low, high)
DiagEnergyLen = DiagEnergyHigh - DiagEnergyLow + 1

import json
import numpy as np
import re
import os
import os.path as path

from .typing import DefaultFloatDType, VoltageUnit, checkVoltageUnit
from .utils import cripAssert, getChildFolder, inRange, isInt, isList


class Spectrum:
    '''
        Read spectrum text and parse it.
        
        Refer to the document for spectrum text format. @see https://github.com/z0gSh1u/crip

        Get \omega of certain energy (keV):
        ```py
            omega = spec.spectrum[voltage]
        ```
    '''
    def __init__(self, spectrumContent: str, unit: VoltageUnit = 'keV'):
        self.content = spectrumContent

        cripAssert(checkVoltageUnit(unit), f'Invalid unit: {unit}')
        self.unit = unit

        self.spectrum = np.zeros(DiagEnergyLen, dtype=DefaultFloatDType)
        self._read()

        self.sumOmega = np.sum(self.spectrum)

    def _read(self):
        # split content into list, and ignore all lines starting with non-digit
        self.content = list(
            filter(lambda y: len(y) > 0 and str.isdigit(y[0]),
                   list(map(lambda x: x.strip(),
                            self.content.replace('\r\n', '\n').split('\n')))))

        def procSpectrumLine(line: str):
            tup = tuple(map(float, re.split(r'\s+', line)))
            cripAssert(len(tup) == 2, f'Invalid line in spectrum: \n{line}\n')
            return tup  # (energy, omega)

        # parse the spectrum text
        self.content = np.array(list(map(procSpectrumLine, self.content)), dtype=DefaultFloatDType)
        spectrumEnergy, spectrumOmega = self.content.T

        # to keV
        if self.unit == 'eV':
            spectrumEnergy /= 1000

        startVoltage, cutOffVoltage = int(spectrumEnergy[0]), int(spectrumEnergy[-1])
        cripAssert(inRange(startVoltage, DiagEnergyRange), '`startVoltage` is out of `DiagEnergyRange`.')
        cripAssert(inRange(cutOffVoltage, DiagEnergyRange), '`cutOffVoltage` is out of `DiagEnergyRange`.')

        self.spectrum[startVoltage:cutOffVoltage] = spectrumOmega[:]


def getAttenList():
    _attenListPath = path.join(getChildFolder('_atten'), './_attenList.json')
    with open(_attenListPath, 'r') as fp:
        _attenList = json.load(fp)

    return _attenList


def getBuiltInAttenText(materialName: str, ICRP=False):
    _attenList = getAttenList()
    _attenPath = getChildFolder('_atten')
    _attenFile = '{}{}.txt'.format(materialName, '_ICRP' if ICRP else '')
    _attenFilePath = path.join(_attenPath, f'./{_attenList[materialName]}', f'./{_attenFile}')
    cripAssert(path.exists(_attenFilePath), f'Atten file {_attenFile} does not exist.')

    with open(_attenFilePath, 'r') as fp:
        content = fp.read()
    return content


def getAtten(material: str, rho: float, energy: np.array) -> np.array:
    """
        Read NIST or ICRP110 ASCII format attenuation coefficient list. `rho` in (g/cm^3). \\
        Return \mu value (mm^-1).
    """
    def getFileList(material, folder, extension):
        file_list = []
        for dir_path, dir_names, file_names in os.walk(folder):
            for file in file_names:
                file_material, file_type = os.path.splitext(file)
                if material == file_material and file_type == extension:
                    file_fullname = os.path.join(dir_path, file)
                    file_list.append(file_fullname)
        return file_list

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
