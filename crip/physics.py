'''
    Physics module of crip.

    https://github.com/z0gSh1u/crip
'''

__all__ = [
    'Spectrum', 'Atten', 'Material', 'calcMu', 'DiagEnergyLow', 'DiagEnergyHigh', 'DiagEnergyRange', 'DiagEnergyLen',
    'forwardProjectWithSpectrum', 'brewPowderSolution', 'calcContrastHU', 'getCommonDensity', 'EnergyConversion'
]

import json
import re
import numpy as np
from os import path
import enum

from ._typing import *
from .utils import cvtEnergyUnit, cvtMuUnit, inArray, cripAssert, getChildFolder, inRange, isNumber, isOfSameShape, isType, readFileText, cvtConcentrationUnit
from .io import listDirectory
from .postprocess import muToHU

## Constants ##

DiagEnergyLow = 0  # keV
DiagEnergyHigh = 150  # keV
DiagEnergyRange = range(DiagEnergyLow, DiagEnergyHigh + 1)  # Diagonstic energy range, [low, high)
DiagEnergyLen = DiagEnergyHigh - DiagEnergyLow + 1
AttenAliases = {
    'Gold': 'Au',
    'Carbon': 'C',
    'Copper': 'Cu',
    'Iodine': 'I',
    'H2O': 'Water',
    'Gadolinium': 'Gd',
}


class EnergyConversion(enum.Enum):
    EID = 'EID'  # Energy-Integrating Detector
    PCD = 'PCD'  # Photon-Counting Detector


def getCommonDensity(materialName: str):
    '''
        Get the common value of density of a specified material (g/cm^3) from built-in dataset.
    '''
    rhoObject = json.loads(readFileText(path.join(getChildFolder('_atten'), './_classicRho.json')))

    if materialName in AttenAliases:
        materialName = AttenAliases[materialName]
    cripAssert(materialName in rhoObject, f'Material not found in density dataset: {materialName}')

    return rhoObject[materialName]


class Spectrum:
    '''
        Construct Spectrum object with omega array of every energy.

        Get \\omega of certain energy (keV):
        ```py
            omega = spec.omega[E]
        ```
    '''

    def __init__(self, omega: np.ndarray, unit='keV'):
        cripAssert(
            len(omega) == DiagEnergyLen,
            f'omega array should have same length as DiagEnergyLen: got {len(omega)} expect {DiagEnergyLen}.')
        cripAssert(inArray(unit, ['MeV', 'keV', 'eV']), f'Invalid unit: {unit}')

        self.unit = unit
        self.omega = np.array(omega, dtype=np.float32)
        self.sumOmega = np.sum(self.omega)

    def isMonochromatic(self):
        at = -1
        for i in DiagEnergyRange:
            if self.omega[i] > 0:
                if at != -1:
                    return False, None
                at = i

        return True, at

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
        cripAssert(inRange(startEnergy, DiagEnergyRange),
                   f'startEnergy is out of DiagEnergyRange: {DiagEnergyRange} keV')
        cripAssert(inRange(cutOffEnergy, DiagEnergyRange),
                   f'cutOffEnergy is out of DiagEnergyRange: {DiagEnergyRange} keV')
        cripAssert(cutOffEnergy + 1 - startEnergy == len(specOmega),
                   'The spectrum is not continous by 1 keV from start to cutoff.')

        omega[startEnergy:cutOffEnergy + 1] = specOmega[:]

        return Spectrum(omega, unit)

    @staticmethod
    def fromFile(path: str, unit='keV'):
        '''
            Construct a Spectrum object from spectrum file (first column is energy while second is omega).
        '''
        spec = readFileText(path)

        return Spectrum.fromText(spec, unit)

    @staticmethod
    def monochromatic(at: int, unit='keV', omega=10**5):
        '''
            Construct a monochromatic spectrum.
        '''
        text = '{} {}\n{} -1'.format(str(at), str(omega), str(at + 1))

        return Spectrum.fromText(text, unit)


class Atten:
    '''
        Parse atten text as `Atten` class object. Interpolation is performed to fill `DiagEnergyRange`.
        Refer to the document for atten text format (NIST ASCII). The density is in g/cm^3.

        Get \\mu (mm-1) of certain energy (keV):
        ```py
            mu = atten.mu[E]
        ```
    '''

    def __init__(self, muArray: NDArray, density: Or[None, float] = None) -> None:
        cripAssert(len(muArray) == DiagEnergyLen, f'muArray should have length of {DiagEnergyLen} energy bins')
        self.mu = muArray
        self.rho = density
        self.attenText = ''
        self.energyUnit = 'keV'

    @staticmethod
    def builtInAttenList() -> List:
        '''
            Get all built-in materials.
        '''
        attenListPath = path.join(getChildFolder('_atten'), './data')
        attenList = list(map(lambda x: x.replace('.txt', ''), listDirectory(attenListPath, style='filename')))
        attenList.extend(AttenAliases.keys())

        return attenList

    @staticmethod
    def _builtInAttenText(materialName: str):
        '''
            Get the built-in atten file content of `materialName` from NIST data source.
        '''
        if materialName in AttenAliases:
            materialName = AttenAliases[materialName]

        attenFilePath = path.join(getChildFolder('_atten'), f'./data/{materialName}.txt')
        cripAssert(path.exists(attenFilePath), f'Atten file for {materialName} does not exist.')
        content = readFileText(attenFilePath)

        return content

    @staticmethod
    def fromBuiltIn(materialName: str, density: Or[float, None] = None):
        '''
            Get the built-in atten object.
            Call `builtInAttenList` to get available materials.
            The density is in g/cm^3.
        '''
        if materialName in AttenAliases:
            materialName = AttenAliases[materialName]

        if density is None:
            density = getCommonDensity(materialName)

        return Atten.fromText(Atten._builtInAttenText(materialName), density, BuiltInAttenEnergyUnit)

    @staticmethod
    def fromText(attenText: str, density: float, energyUnit='MeV'):
        cripAssert(density > 0, '`density` should > 0.')
        rho = density

        # split attenText into list, and ignore all lines starting with non-digit
        content = list(
            filter(lambda y: len(y) > 0 and str.isdigit(y[0]),
                   list(map(lambda x: x.strip(),
                            attenText.replace('\r\n', '\n').split('\n')))))

        def procAttenLine(line: str):
            tup = tuple(map(float, re.split(r'\s+', line)))
            cripAssert(inArray(len(tup), [2, 3]), f'Invalid line in attenText: {line}')
            energy, muDivRho = tuple(map(float, re.split(r'\s+', line)))[0:2]
            return energy, muDivRho

        content = np.array(list(map(procAttenLine, content)), dtype=DefaultFloatDType)

        energy, muDivRho = content.T
        energy = cvtEnergyUnit(energy, energyUnit, DefaultEnergyUnit)  # keV

        # perform log-domain interpolation to fill in `DiagEnergyRange`
        interpEnergy = np.log(DiagEnergyRange[1:])  # avoid log(0)
        interpMuDivRho = np.interp(interpEnergy, np.log(energy), np.log(muDivRho))

        # now we have mu for every energy in `DiagEnergyRange`.
        mu = np.exp(interpMuDivRho) * rho  # cm-1
        mu = cvtMuUnit(mu, 'cm-1', DefaultMuUnit)  # mm-1
        mu = np.insert(mu, 0, 0, axis=0)  # prepend energy = 0

        return Atten(mu, rho)

    @staticmethod
    def fromMuArray(muArray: NDArray, rho: Or[float, None] = None):
        return Atten(muArray, rho)

    @staticmethod
    def fromFile(path: str, rho: float, energyUnit='MeV'):
        '''
            Construct a new material from file where first column is energy while second
            is \\mu / \\rho.
        '''
        atten = readFileText(path)

        return Atten.fromText(atten, rho, energyUnit)


Material = Atten
WaterAtten = Atten.fromBuiltIn('Water')


def calcMu(atten: Atten, spec: Spectrum, energyConversion: str) -> float:
    '''
        Calculate the LAC \mu value (mm-1) of certain atten under a specific spectrum.
        energyConversion determines the energy conversion efficiency of the detector,
        can be "PCD" (Photon Counting), "EID" (Energy Integrating)
    '''
    cripAssert(inArray(energyConversion, ['PCD', 'EID']), f'Invalid energyConversion: {energyConversion}')

    mus = atten.mu
    eff = {'PCD': 1, 'EID': np.array(DiagEnergyRange)}[energyConversion]

    return np.sum(spec.omega * eff * mus) / np.sum(spec.omega * eff)


def calcAttenedSpec(spec: Spectrum, attens: Or[Atten, List[Atten]], Ls: Or[float, List[float]]) -> Spectrum:
    '''
        Calculate the attenuated spectrum using polychromatic Beer-Lambert law. Supports multiple materials.

        I.e., `\\Omega(E) \\exp (- \\mu(E) L) through all E`. L in mm.
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


def forwardProjectWithSpectrum(lengths: List[TwoD], materials: List[Atten], spec: Spectrum, energyConversion: str):
    '''
        Perform forward projection using `spec`. `lengths` is a list of corresponding length [mm] images 
        (projection or sinogram) of `materials`. 
        This function would simulate attenuation and Beam Hardening but no scatter.
        
        Set `lengths` and `materials` to empty lists to compute the flat field.

        It's highly recommended to use a monochronmatic spectrum to accelerate if you simulate a lot.
    '''
    cripAssert(len(lengths) == len(materials), 'Lengths and materials should correspond.')
    cripAssert(all([isOfSameShape(lengths[0], x) for x in lengths]), 'Lengths map should have same shape.')
    cripAssert(energyConversion in ['PCD', 'EID'], 'Invalid energyConversion.')

    efficiency = 1 if energyConversion == 'PCD' else np.array(DiagEnergyRange)

    if len(lengths) == 0:
        # compute the flat field
        ones = np.ones_like(lengths[0], dtype=DefaultFloatDType) if len(lengths) > 0 else 1
        effectiveOmega = spec.omega * efficiency

        return np.sum(effectiveOmega) * ones

    resultShape = lengths[0].shape

    # speed up when it's monochromatic
    isMono, monoAt = spec.isMonochromatic()
    if isMono:
        attenuations = 0.0
        for length, material in zip(lengths, materials):
            attenuations += length * material.mu[monoAt]

        attened = spec.omega[monoAt] * np.exp(-attenuations) * (1 if energyConversion == 'PCD' else monoAt)  # the attenuated image

        return attened
    else:
        # a[h, w] = [vector of attenuation in that energy bin]
        attenuations = np.zeros((*resultShape, DiagEnergyLen), dtype=DefaultFloatDType)
        for length, material in zip(lengths, materials):
            attenuations += np.outer(length, material.mu).reshape((*resultShape, DiagEnergyLen))

        attened = spec.omega * np.exp(-attenuations) * efficiency  # the attenuated image

    return np.sum(attened, axis=-1)


def brewPowderSolution(solute: Atten,
                       solvent: Atten,
                       concentration: float,
                       concentrationUnit='mg/mL',
                       rhoSolution: Or[float, None] = None) -> Atten:
    '''
        Generate the Atten of powder solution with certain concentration (mg/mL by default).
    '''
    cripAssert(inArray(concentrationUnit, ['mg/mL', 'g/mL']), f'Invalid concentration unit: {concentrationUnit}')

    concentration = cvtConcentrationUnit(concentration, concentrationUnit, 'g/mL')
    mu = solvent.mu + (solute.mu / solute.rho) * concentration
    atten = Atten.fromMuArray(mu, rhoSolution)

    return atten


def calcContrastHU(contrast: Atten, spec: Spectrum, energyConversion: str, base: Atten = WaterAtten):
    '''
        Calculate HU difference resulted by contrast.
    '''
    cripAssert(energyConversion in ['EID', 'PCD'], 'Invalid energyConversion.')

    _calcMu = lambda atten: calcMu(atten, spec, energyConversion)

    muWater = _calcMu(WaterAtten)
    if base is not WaterAtten:
        muBase = _calcMu(base)
    else:
        muBase = muWater
    muContrast = _calcMu(contrast)

    contrastHU = muToHU(muContrast, muWater) - muToHU(muBase, muWater)

    return contrastHU
