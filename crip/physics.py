'''
    Physics module of crip.

    https://github.com/SEU-CT-Recon/crip
'''

import json
import re
import numpy as np
from os import path
import periodictable as pt
import enum

from ._typing import *
from .utils import *
from .io import listDirectory
from .postprocess import muToHU

## Constants ##

DiagEnergyLow = 0  # [keV] Lowest energy of diagnostic X-ray
DiagEnergyHigh = 150  # [keV] Highest energy of diagnostic X-ray
DiagEnergyRange = range(DiagEnergyLow, DiagEnergyHigh + 1)  # Diagonstic energy range, [0, 150] keV
DiagEnergyLen = DiagEnergyHigh - DiagEnergyLow + 1
NA = 6.02214076e23  # Avogadro's number

# Alias for built-in materials
AttenAliases = {
    'Gold': 'Au',
    'Carbon': 'C',
    'Copper': 'Cu',
    'Iodine': 'I',
    'H2O': 'Water',
    'Gadolinium': 'Gd',
}


class EnergyConversion(enum.Enum):
    ''' Energy conversion efficiency marker of the detector.
    '''
    EID = 'EID'  # Energy-Integrating Detector
    PCD = 'PCD'  # Photon-Counting Detector


def getCommonDensity(materialName: str):
    ''' Get the common density of a material [g/cm^3] from built-in dataset.
    '''
    db = json.loads(readFileText(path.join(getAsset('atten'), 'density.json')))

    if materialName in AttenAliases:
        materialName = AttenAliases[materialName]
    cripAssert(materialName in db, f'Material not found in density dataset: {materialName}')

    return db[materialName]


class Spectrum:

    def __init__(self, omega: NDArray, unit='keV'):
        ''' Construct a Spectrum object with omega array describing every energy bin in DiagEnergyRange.
            To access omega of certain energy E [keV], use `spec.omega[E]`.
        '''
        cripAssert(
            len(omega) == DiagEnergyLen,
            f'omega array should length of DiagEnergyLen ({DiagEnergyLen}), but got {len(omega)}.')
        cripAssert(unit in EnergyUnits, f'Invalid unit: {unit}.')

        self.unit = unit
        self.omega = np.array(omega, dtype=DefaultFloatDType)
        self.sumOmega = np.sum(self.omega)

    def isMonochromatic(self):
        ''' Test if the spectrum is monochromatic. If so, return True and the energy.
        '''
        at = -1
        for i in DiagEnergyRange:
            if self.omega[i] > 0:
                if at != -1:
                    return False, None
                at = i

        return True, at

    @staticmethod
    def fromText(specText: str, unit='keV'):
        ''' Parse spectrum text into `Spectrum` object.
            The text should be in the format of `<energy> <omega>`, one pair per line.
            Leading lines starting with non-digit will be ignored.
            All `<energy>` should be in ascending order and continous by 1 keV and inside `DiagEnergyRange`.
            Recommend you to end the last line with `omega=-1` to indicate the end of spectrum.
        '''
        cripAssert(unit in EnergyUnits, f'Invalid unit: {unit}.')
        omega = np.zeros(DiagEnergyLen, dtype=DefaultFloatDType)

        # split content into list, and ignore all lines starting with non-digit
        content = list(
            filter(lambda y: len(y) > 0 and str.isdigit(y[0]),
                   list(map(lambda x: x.strip(),
                            specText.replace('\r\n', '\n').split('\n')))))

        # process each line
        def proc1(line: str):
            tup = tuple(map(float, re.split(r'\s+', line)))
            cripAssert(len(tup) == 2, f'Invalid line in spectrum: {line}.')

            return tup  # (energy, omega)

        # parse the spectrum text to get provided omega array
        content = np.array(list(map(proc1, content)), dtype=DefaultFloatDType)
        specEnergy, specOmega = content.T
        specEnergy = convertEnergyUnit(specEnergy, unit, DefaultEnergyUnit)
        cripAssert(np.all(np.diff(specEnergy) == 1), 'Energy should be in ascending order and continous by 1 keV.')
        specOmega[specOmega < 0] = 0
        startEnergy, cutOffEnergy = int(specEnergy[0]), int(specEnergy[-1])
        cripAssert(startEnergy in DiagEnergyRange, f'startEnergy is out of DiagEnergyRange: {startEnergy} keV')
        cripAssert(cutOffEnergy in DiagEnergyRange, f'cutOffEnergy is out of DiagEnergyRange: {cutOffEnergy} keV')

        # fill omega array
        omega[startEnergy:cutOffEnergy + 1] = specOmega[:]

        return Spectrum(omega, unit)

    @staticmethod
    def fromFile(path: str, unit='keV'):
        ''' Construct a Spectrum object from spectrum file whose format follows `fromText`.
        '''
        return Spectrum.fromText(readFileText(path), unit)

    @staticmethod
    def monochromatic(at: int, unit='keV', omega=10**5):
        ''' Construct a monochromatic spectrum at `at`.
        '''
        text = '{} {}\n{} -1'.format(str(at), str(omega), str(at + 1))

        return Spectrum.fromText(text, unit)


class Atten:

    def __init__(self, muArray: NDArray, density: Or[None, float] = None) -> None:
        ''' Construct an Atten object with mu array describing every LAC for energy bins in DiagEnergyRange.
            The density is in [g/cm^3]. To access mu [mm-1] of certain energy E [keV], use `atten.mu[E]`.
        '''
        cripAssert(len(muArray) == DiagEnergyLen, f'muArray should have length of {DiagEnergyLen} energy bins.')
        if density is not None:
            cripAssert(density > 0, 'Density should be positive.')

        self.mu = muArray
        self.rho = density
        self.attenText = ''
        self.energyUnit = 'keV'

    @staticmethod
    def builtInAttenList() -> List[str]:
        ''' Get all built-in material names and aliases.
        '''
        attenListPath = path.join(getAsset('atten'), 'data')
        attenList = list(map(lambda x: x.replace('.txt', ''), listDirectory(attenListPath, style='filename')))
        attenList.extend(AttenAliases.keys())

        return attenList

    @staticmethod
    def builtInAttenText(material: str):
        ''' Get built-in atten file content of `material` from NIST data source.
        '''
        if material in AttenAliases:
            material = AttenAliases[material]

        attenFilePath = path.join(getAsset('atten'), f'data/{material}.txt')
        cripAssert(path.exists(attenFilePath), f'Atten file for {material} does not exist.')

        return readFileText(attenFilePath)

    @staticmethod
    def fromBuiltIn(material: str, density: Or[float, None] = None):
        ''' Get a built-in atten object for `material`.
            Call `builtInAttenList` to inspect all available materials.
            The density is in [g/cm^3], and will be automatically filled with common value if not provided.
        '''
        if material in AttenAliases:
            material = AttenAliases[material]

        if density is None:
            density = getCommonDensity(material)

        return Atten.fromText(Atten.builtInAttenText(material), density, BuiltInAttenEnergyUnit)

    @staticmethod
    def fromText(attenText: str, density: float, energyUnit='MeV'):
        ''' Parse atten text into `Atten` object. Interpolation will be performed to fill `DiagEnergyRange`.
            The text should be in the format of `<energy> <mu/density>` one pair per line,
            or `<energy> <mu/density> <mu_en/density>` where the second column will be used.
            Leading lines starting with non-digit will be ignored.
        '''
        rho = density

        # split attenText into list, and ignore all lines starting with non-digit
        content = list(
            filter(lambda y: len(y) > 0 and str.isdigit(y[0]),
                   list(map(lambda x: x.strip(),
                            attenText.replace('\r\n', '\n').split('\n')))))

        # process each line
        def proc1(line: str):
            tup = tuple(map(float, re.split(r'\s+', line)))
            cripAssert(len(tup) in [2, 3], f'Invalid line in attenText: {line}.')
            energy, muDivRho = tuple(map(float, re.split(r'\s+', line)))[0:2]

            return energy, muDivRho

        # parse the atten text to get provided mu array
        content = np.array(list(map(proc1, content)), dtype=DefaultFloatDType)
        energy, muDivRho = content.T
        energy = convertEnergyUnit(energy, energyUnit, DefaultEnergyUnit)  # keV

        # perform log-domain interpolation to fill in `DiagEnergyRange`
        interpEnergy = np.log(DiagEnergyRange[1:])  # avoid log(0)
        interpMuDivRho = np.interp(interpEnergy, np.log(energy), np.log(muDivRho))

        # now we have mu for every energy in `DiagEnergyRange`.
        mu = np.exp(interpMuDivRho) * rho  # cm-1
        mu = convertMuUnit(mu, 'cm-1', DefaultMuUnit)  # mm-1
        mu = np.insert(mu, 0, 0, axis=0)  # prepend for E=0

        return Atten.fromMuArray(mu, rho)

    @staticmethod
    def fromMuArray(muArray: NDArray, density: Or[float, None] = None):
        ''' Construct a new material from mu array for every energy in `DiagEnergyRange`.
            The density is in [g/cm^3].
        '''
        return Atten(muArray, density)

    @staticmethod
    def fromFile(path: str, density: float, energyUnit='MeV'):
        ''' Construct a new material from file where the format follows `fromText`.
        '''
        return Atten.fromText(readFileText(path), density, energyUnit)


WaterAtten = Atten.fromBuiltIn('Water')


def computeMu(atten: Atten, spec: Spectrum, energyConversion: EnergyConversion) -> float:
    ''' Calculate the LAC (mu) [mm-1] for certain Atten under a Spectrum.
        `energyConversion` determines the energy conversion efficiency of the detector.
    '''
    mus = atten.mu
    eff = {
        EnergyConversion.PCD: 1,
        EnergyConversion.EID: np.array(DiagEnergyRange),
    }[energyConversion]

    return np.sum(spec.omega * eff * mus) / np.sum(spec.omega * eff)


def computeAttenedSpectrum(spec: Spectrum, attens: List[Atten], Ls: List[float]) -> Spectrum:
    ''' Calculate the spectrum after attenuation through `attens` with thickness `Ls` [mm]
        using Beer-Lambert law.
    '''
    cripAssert(len(attens) == len(Ls), '`attens` should be paired with `Ls`.')

    N = len(attens)
    omega = np.array(spec.omega, copy=True)
    for i in range(N):
        atten, L = attens[i], Ls[i]
        for E in DiagEnergyRange:  # energies
            omega[E] *= np.exp(-atten.mu[E] * L)

    return Spectrum(omega, spec.unit)


def normalizeSpectrum(spec: Spectrum):
    ''' Normalize a Spectrum.
    '''
    return Spectrum(spec.omega / spec.sumOmega, spec.unit)


def forwardProjectWithSpectrum(lengths: List[TwoD], materials: List[Atten], spec: Spectrum, energyConversion: str):
    ''' Perform forward projection using `spec`. `lengths` is a list of corresponding length [mm] images 
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

        attened = spec.omega[monoAt] * np.exp(-attenuations) * (1 if energyConversion == 'PCD' else monoAt
                                                               )  # the attenuated image

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
    ''' Generate the `Atten` of ideal powder solution with certain concentration.
    '''
    cripAssert(concentrationUnit in ConcentrationUnits, f'Invalid concentration unit: {concentrationUnit}.')

    concentration = convertConcentrationUnit(concentration, concentrationUnit, 'g/mL')
    mu = solvent.mu + (solute.mu / solute.rho) * concentration
    atten = Atten.fromMuArray(mu, rhoSolution)

    return atten


def computeContrastHU(contrast: Atten,
                      spec: Spectrum,
                      energyConversion: EnergyConversion,
                      base: Atten = WaterAtten) -> float:
    ''' Compute the HU difference caused by contrast.
    '''
    _calcMu = lambda atten: computeMu(atten, spec, energyConversion)

    muWater = _calcMu(WaterAtten)
    muContrast = _calcMu(contrast)
    if base is not WaterAtten:
        muBase = _calcMu(base)
    else:
        muBase = muWater

    return muToHU(muContrast, muWater) - muToHU(muBase, muWater)


def computePathLength(thickness: float, sod: float, detH: int, detW: int, detElementSize: float) -> NDArray:
    ''' Compute the ray peneration pathlength for each detector element using a cuboid object with `thickness`.
        `sod` is the Source-Object Distance. `(detH, detW)` is the detector size [pixel].
        `detElementSize` is the element size of detector.
        All length units are recommended to be [mm]. Currently no object offset can be assumed.
    '''
    detCenter = (detW / 2, detH / 2)
    r, c = np.meshgrid(np.arange(detH), np.arange(detW))
    coords = np.array((r.flatten(), c.flatten()), dtype=np.float32).T

    offcenter = coords - np.array(detCenter)
    offcenter = np.abs(offcenter) * detElementSize
    offcenterDist = np.sqrt(np.sum(offcenter**2, axis=1))

    theta = np.arctan(offcenterDist / sod)
    rayIntersect = thickness / np.cos(theta)
    rayIntersect = rayIntersect.reshape((detW, detH)).T

    return rayIntersect


def atomsFromMolecule(molecule: str) -> Dict[str, int]:
    ''' Parse the molecule string to get the atoms and their counts.
        e.g. `'H2 O1' -> {'H': 2, 'O': 1}`
    '''
    atoms = {}
    for part in molecule.split(' '):
        count = ''.join(filter(str.isdigit, part))
        element = ''.join(filter(str.isalpha, part))
        atoms[element] = int(count)

    return atoms


def zeffTheoretical(molecule: str, m=2.94) -> float:
    ''' Compute the theoretical effective atomic number (Zeff) of a molecule using the power law with parameter `m`.
        `molecule` is parsed by `atomsFromMolecule`.
        [1] https://en.wikipedia.org/wiki/Effective_atomic_number_(compounds_and_mixtures)
    '''
    atoms = atomsFromMolecule(molecule)
    totalElectrons = 0
    for atom in atoms:
        totalElectrons += atoms[atom] * pt.elements.symbol(atom).number

    sumUnderSqrt = 0
    for atom in atoms:
        Z = pt.elements.symbol(atom).number  # atomic number
        f = atoms[atom] * Z / totalElectrons  # fraction
        sumUnderSqrt += f * (Z**m)

    return sumUnderSqrt**(1 / m)


def zeffExperimental(a1: float, a2: float, rhoe1: float, rhoe2: float, zeff1: float, zeff2: float, m=2.94):
    ''' Compute the experimental effective atomic number (Zeff) from Dual-Energy Two-Material decomposition
        using the power law with parameter `m`. `(a1, a2)` are material decomposition coefficients.
        `(rhoe1, rhoe2)` are electron densities of material bases.
        `(zeff1, zeff2)` are effective atomic numbers of material bases.
    '''
    n1 = a1 * rhoe1 * zeff1**m
    n2 = a2 * rhoe2 * zeff2**m
    d1 = a1 * rhoe1
    d2 = a2 * rhoe2

    return ((n1 + n2) / (d1 + d2))**(1 / m)
