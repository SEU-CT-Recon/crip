'''
    This example uses crip to calculate \mu_water for a specified spectrum.

    by z0gSh1u @ https://github.com/z0gSh1u/crip
'''

from crip.physics import calcMu, Atten, Spectrum, getCommonDensity

# A sample spectrum file.
SpectrumFile = '''
48000  100
49000  50
50000  -1
'''

WaterDensity = getCommonDensity('Water')
print(f'rho_water = {WaterDensity} g/cm^3')

spec = Spectrum.fromText(SpectrumFile, 'eV')
atten = Atten.fromBuiltIn('Water', WaterDensity)
mu = calcMu(atten, spec, 'PCD')
print(f'mu = {mu} mm-1.')
