'''
    This example uses crip to calculate \mu_water for a specified spectrum.

    by z0gSh1u @ https://github.com/z0gSh1u/crip
'''

from crip.physics import calcMu, Atten, Spectrum

# A sample spectrum file.
SpectrumFile = '''
Energy (eV)    Omega
5000  6.629919e-195
50000  -1
'''.strip()

spec = Spectrum.fromText(SpectrumFile, 'eV')
atten = Atten.fromBuiltIn('Water', 1.0)
mu = calcMu(atten, spec, 'PCD')
print(f'\\mu = {mu} mm-1.')
