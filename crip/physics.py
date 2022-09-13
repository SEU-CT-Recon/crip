'''
    Physics module of crip.

    https://github.com/z0gSh1u/crip
'''

__all__ = [
    'Spectrum', 'Atten', 'calcMu', 'RhoWater', 'DiagEnergyLow', 'DiagEnergyHigh', 'DiagEnergyRange', 'DiagEnergyLen',
    'getClassicDensity', 'forwardProjectWithSpectrum'
]

import json
import re
import numpy as np
from os import path

from ._typing import *
from .utils import cvtEnergyUnit, cvtMuUnit, inArray, cripAssert, getChildFolder, inRange, isNumber, isOfSameShape, isType, readFileText

## Constants ##

RhoWater = 1.0  # g/cm^3
DiagEnergyLow = 0  # keV
DiagEnergyHigh = 150  # keV
DiagEnergyRange = range(DiagEnergyLow, DiagEnergyHigh + 1)  # [low, high)
DiagEnergyLen = DiagEnergyHigh - DiagEnergyLow + 1


class Spectrum:
    '''
        Construct Spectrum object with omega array of every energy.

        Get \\omega of certain energy (keV):
        ```py
            omega = spec.omega[E]
        ```
    '''
    def __init__(self, omega: np.ndarray, unit='keV'):
        cripAssert(len(omega) == DiagEnergyLen, 'omega array should have same length as DiagEnergyLen.')
        cripAssert(inArray(unit, ['MeV', 'keV', 'eV']), f'Invalid unit: {unit}')

        self.unit = unit
        self.omega = np.array(omega)
        self.update()

    def update(self):
        self.sumOmega = np.sum(self.omega)

    @staticmethod
    def fromText(specText: str, unit='keV'):
        '''
            Parse spectrum text as `Spectrum` class object.
            
            Refer to the document for spectrum text format. @see https://github.com/z0gSh1u/crip            
        '''
        cripAssert(inArray(unit, ['MeV', 'keV', 'eV']), f'Invalid unit: {unit}')

        omega = np.zeros(DiagEnergyLen, dtype=DefaultFloatDType)

        # split content into list, and ignore all lines starting with non-digit
        content = list(
            filter(lambda y: len(y) > 0 and str.isdigit(y[0]),
                   list(map(lambda x: x.strip(),
                            specText.replace('\r\n', '\n').split('\n')))))

        def procSpecLine(line: str):
            tup = tuple(map(float, re.split(r'\s+', line)))
            cripAssert(len(tup) == 2, f'Invalid line in spectrum: \n{line}\n')
            return tup  # (energy, omega)

        # parse the spectrum text
        content = np.array(list(map(procSpecLine, content)), dtype=DefaultFloatDType)
        specEnergy, specOmega = content.T
        specOmega[specOmega < 0] = 0

        # to keV
        specEnergy = cvtEnergyUnit(specEnergy, unit, DefaultEnergyUnit)

        startEnergy, cutOffEnergy = int(specEnergy[0]), int(specEnergy[-1])
        cripAssert(inRange(startEnergy, DiagEnergyRange), '`startEnergy` is out of `DiagEnergyRange`.')
        cripAssert(inRange(cutOffEnergy, DiagEnergyRange), '`cutOffEnergy` is out of `DiagEnergyRange`.')
        cripAssert(cutOffEnergy + 1 - startEnergy == len(specOmega),
                   'The spectrum is not continous by 1 keV from start to cutoff.')

        omega[startEnergy:cutOffEnergy + 1] = specOmega[:]

        return Spectrum(omega, unit)

    @staticmethod
    def fromFile(path: str, unit='keV'):
        with open(path, 'r') as fp:
            spec = fp.read()

        return Spectrum.fromText(spec, unit)


class Atten:
    '''
        Parse atten text as `Atten` class object. Interpolation is performed to fill `DiagEnergyRange`.

        Refer to the document for atten text format (NIST ASCII or ICRP). @see https://github.com/z0gSh1u/crip

        \\rho: g/cm^3.

        Get \\mu of certain energy (keV):
        ```py
            mu = atten.mu[E]
        ```
    '''
    def __init__(self, attenText: str, rho: float, energyUnit='MeV') -> None:
        cripAssert(rho > 0, '`rho` should > 0.')

        self.attenText = attenText
        self.energyUnit = energyUnit
        self.rho = rho

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
    def builtInAttenList() -> Dict:
        '''
            Get the built-in atten list file content.

            Returns `{materialName: materialType}`
        '''
        _attenListPath = path.join(getChildFolder('_atten'), './_attenList.json')
        _attenList = readFileText(_attenListPath)

        return json.loads(_attenList)

    @staticmethod
    def _builtInAttenText(materialName: str, dataSource='NIST'):
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
    def fromBuiltIn(materialName: str, rho: Or[float, None] = None, dataSource='NIST'):
        '''
            Get the built-in atten object.

            Available data sources: `NIST`, `ICRP`.       
            
            \\rho: g/cm^3.
        '''
        if rho is None:
            rho = getClassicDensity(materialName, dataSource)

        return Atten(Atten._builtInAttenText(materialName, dataSource), rho, BuiltInAttenEnergyUnit)


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

    return np.sum(spec.omega * eff * mus) / np.sum(spec.omega * eff)


def getClassicDensity(materialName: str, dataSource='NIST'):
    '''
        Get the classic value of density of a specified material (g/cm^3) from built-in dataset.
    '''
    cripAssert(inArray(dataSource, ['NIST', 'ICRP']), f'Invalid dataSource: {dataSource}')

    _classicRhoPath = path.join(getChildFolder('_atten'), './_classicRho.json')
    _classicRho = readFileText(_classicRhoPath)
    rhoObject = json.loads(_classicRho)

    dataSourcePostfix = {'NIST': '', 'ICRP': '_ICRP'}[dataSource]
    key = '{}{}'.format(materialName, dataSourcePostfix)
    cripAssert(key in rhoObject, f'Record not found: {key}')

    return rhoObject[key]


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


def forwardProjectWithSpectrum(lengths: List[TwoD],
                               materials: List[Atten],
                               spec: Spectrum,
                               energyConversion: str,
                               fastSkip: bool = False,
                               flat: float = None):
    '''
        Perform forward projection using `spec`. `lengths` is a list of corresponding length [mm] images 
        (projection or sinogram) of `materials`. Set `lengths` and `materials` to empty lists to compute the flat field.
        This function would simulate attenuation and Beam Hardening but no scatter.
    '''
    cripAssert(len(lengths) == len(materials), 'Lengths and materials should correspond.')
    cripAssert(all([isOfSameShape(lengths[0], x) for x in lengths]), 'Lengths map should have same shape.')
    cripAssert(energyConversion in ['PCD', 'EID'], 'Invalid energyConversion.')

    efficiency = 1 if energyConversion == 'PCD' else np.array(DiagEnergyRange)

    if (len(lengths) == 0) or (fastSkip and (all([np.sum(x) == 0 for x in lengths]))):
        ones = np.ones_like(lengths[0], dtype=DefaultFloatDType) if len(lengths) > 0 else 1
        if flat is not None:
            return flat * ones
        else:
            effectiveOmega = spec.omega * efficiency
            return np.sum(effectiveOmega) * ones

    resultShape = lengths[0].shape
    # a[h, w] = [vector of attenuation in that energy bin]
    attenuations = np.zeros((*resultShape, DiagEnergyLen), dtype=DefaultFloatDType)
    for length, material in zip(lengths, materials):
        attenuations += np.outer(length, material.mu).reshape((*resultShape, DiagEnergyLen))

    attened = spec.omega * np.exp(-attenuations) * efficiency  # the attenuated image

    return np.sum(attened, axis=-1)
