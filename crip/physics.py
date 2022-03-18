'''
    Physics module of crip.

    https://github.com/z0gSh1u/crip
'''

__all__ = ['Spectrum', 'Atten', 'calcMu']

import json
import re
import numpy as np
from os import path
from typing import Callable

from .typing import BuiltInAttenEnergyUnit, DefaultEnergyUnit, DefaultFloatDType, DefaultMuUnit, Or
from .utils import cvtEnergyUnit, cvtMuUnit, inArray, cripAssert, getChildFolder, inRange, isNumber, isType, readFileText

## Constants ##

RhoWater = 1.0  # g/cm^3
DiagEnergyLow = 0
DiagEnergyHigh = 150
DiagEnergyRange = range(DiagEnergyLow, DiagEnergyHigh + 1)  # [low, high)
DiagEnergyLen = DiagEnergyHigh - DiagEnergyLow + 1


class Spectrum:
    '''
        Parse spectrum text as `Spectrum` class object.
        
        Refer to the document for spectrum text format. @see https://github.com/z0gSh1u/crip

        Get \omega of certain energy (keV):
        ```py
            omega = spec.spectrum[E]
        ```
    '''
    def __init__(self, specText: Or[str, None], unit='keV'):
        self.specText = specText

        cripAssert(inArray(unit, ['MeV', 'keV', 'eV']), f'Invalid unit: {unit}')
        self.unit = unit

        self.spectrum = np.zeros(DiagEnergyLen, dtype=DefaultFloatDType)
        if specText is not None:
            self._read()

        self.sumOmega = np.sum(self.spectrum)

    def _read(self):
        # split content into list, and ignore all lines starting with non-digit
        content = list(
            filter(lambda y: len(y) > 0 and str.isdigit(y[0]),
                   list(map(lambda x: x.strip(),
                            self.specText.replace('\r\n', '\n').split('\n')))))

        def procSpectrumLine(line: str):
            tup = tuple(map(float, re.split(r'\s+', line)))
            cripAssert(len(tup) == 2, f'Invalid line in spectrum: \n{line}\n')
            return tup  # (energy, omega)

        # parse the spectrum text
        content = np.array(list(map(procSpectrumLine, content)), dtype=DefaultFloatDType)
        specEnergy, specOmega = content.T

        # to keV
        specEnergy = cvtEnergyUnit(specEnergy, self.unit, DefaultEnergyUnit)

        startEnergy, cutOffEnergy = int(specEnergy[0]), int(specEnergy[-1])
        cripAssert(inRange(startEnergy, DiagEnergyRange), '`startEnergy` is out of `DiagEnergyRange`.')
        cripAssert(inRange(cutOffEnergy, DiagEnergyRange), '`cutOffEnergy` is out of `DiagEnergyRange`.')

        self.spectrum[startEnergy:cutOffEnergy + 1] = specOmega[:]

    @staticmethod
    def fromOmegaArray(omega: np.ndarray, unit='keV'):
        cripAssert(len(omega) == DiagEnergyLen, 'omega array should have same length as DiagEnergyLen.')

        spec = Spectrum(None, unit)
        spec.spectrum = omega
        spec.sumOmega = np.sum(spec.spectrum)

        return spec


class Atten:
    '''
        Parse atten text as `Atten` class object. Interpolation is performed to fill `DiagEnergyRange`.

        Refer to the document for atten text format (NIST ASCII or ICRP). @see https://github.com/z0gSh1u/crip

        \\rho: g/cm^3.

        Get \mu of certain energy (keV):
        ```py
            mu = atten.mu[E]
        ```
    '''
    def __init__(self, attenText: str, rho: float, energyUnit='MeV') -> None:
        cripAssert(rho > 0, '`rho` should > 0.')

        self.attenText = attenText
        self.rho = rho
        self.energyUnit = energyUnit

        self.mu = np.zeros(DiagEnergyLen, dtype=DefaultFloatDType)
        self._read()

    def _read(self):
        # split attenText into list, and ignore all lines starting with non-digit
        content = list(
            filter(lambda y: len(y) > 0 and str.isdigit(y[0]),
                   list(map(lambda x: x.strip(),
                            self.attenText.replace('\r\n', '\n').split('\n')))))

        def procAttenLine(line: str):
            tup = tuple(map(float, re.split(r'\s+', line)))
            cripAssert(inArray(len(tup), [2, 3]), f'Invalid line in attenText: {line}')
            energy, muDivRho = tuple(map(float, re.split(r'\s+', line)))[0:2]
            return energy, muDivRho

        content = np.array(list(map(procAttenLine, content)), dtype=DefaultFloatDType)

        energy, muDivRho = content.T
        energy = cvtEnergyUnit(energy, self.energyUnit, DefaultEnergyUnit)  # keV

        # perform log-domain interpolation to fill in `DiagEnergyRange`
        interpEnergy = np.log(DiagEnergyRange[1:])  # avoid log(0)
        interpMuDivRho = np.interp(interpEnergy, np.log(energy), np.log(muDivRho))

        # now we have mu for every energy in `DiagEnergyRange`.
        mu = np.exp(interpMuDivRho) * self.rho  # cm-1
        mu = cvtMuUnit(mu, 'cm-1', DefaultMuUnit)  # mm-1
        mu = np.insert(mu, 0, 0, axis=0)  # prepend energy = 0
        self.mu = mu

    @staticmethod
    def builtInAttenList():
        '''
            Get the built-in atten list file content.

            Returns `{materialName: materialType}`
        '''
        _attenListPath = path.join(getChildFolder('_atten'), './_attenList.json')
        _attenList = readFileText(_attenListPath)

        return json.loads(_attenList)

    @staticmethod
    def builtInAttenText(materialName: str, dataSource='NIST'):
        '''
            Get the built-in atten file content of `materialName`.

            Available data sources: `NIST`, `ICRP`. Call `getBuiltInAttenList` to get the material list.
        '''
        cripAssert(inArray(dataSource, ['NIST', 'ICRP']), f'Invalid dataSource: {dataSource}')
        dataSourcePostfix = {'NIST': '', 'ICRP': '_ICRP'}[dataSource]

        _attenList = Atten.builtInAttenList()
        _attenPath = getChildFolder('_atten')
        _attenFile = '{}{}.txt'.format(materialName, dataSourcePostfix)
        _attenFilePath = path.join(_attenPath, f'./{_attenList[materialName]}', f'./{_attenFile}')
        cripAssert(path.exists(_attenFilePath), f'Atten file {_attenFile} does not exist.')

        content = readFileText(_attenFilePath)
        return content

    @staticmethod
    def fromBuiltIn(materialName: str, rho: float, dataSource='NIST'):
        '''
            Get the built-in atten object.

            Available data sources: `NIST`, `ICRP`.       
            
            \\rho: g/cm^3.
        '''
        return Atten(Atten.builtInAttenText(materialName, dataSource), rho, BuiltInAttenEnergyUnit)


def calcMu(atten: Atten, spec: Spectrum, energyConversion: Or[str, float, int, Callable]) -> float:
    '''
        Calculate the \mu value (mm-1) of certain atten under a specific spectrum.

        `energyConversion` determines the energy conversion efficiency of the detector.
            - "PCD" (Photon Counting), "EID" (Energy Integrating)
            - a constant value
            - a callback function (callable) that takes energy in keV and returns the factor
    '''
    mus = atten.mu
    eff = None

    if isType(energyConversion, str):
        cripAssert(inArray(energyConversion, ['PCD', 'EID']), f'Invalid `energyConversion`: {energyConversion}')
        eff = {'PCD': 1, 'EID': np.array(DiagEnergyRange)}[energyConversion]

    elif isNumber(energyConversion):
        eff = energyConversion

    elif isType(energyConversion, Callable):
        eff = np.array(list(map(energyConversion, list(DiagEnergyRange)))).squeeze()

    else:
        cripAssert(False, 'Invalid `energyConversion`.')

    return np.sum(spec.spectrum * eff * mus) / np.sum(spec.spectrum * eff)
