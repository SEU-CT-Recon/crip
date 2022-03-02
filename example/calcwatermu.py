'''
    This example uses crip to calculate \mu_water for a specified spectrum.

    by z0gSh1u @ https://github.com/z0gSh1u/crip
'''

from . import _importcrip

# A sample spectrum file.
SpectrumFile = '''
Energy (eV)    Omega
50000 	 1.10036e+06
51000 	 1.10796e+06
52000 	 1.1069e+06
53000 	 1.10553e+06
54000 	 1.0986e+06
55000 	 1.09109e+06
56000 	 1.07444e+06
57000 	 1.05732e+06
58000 	 1.42527e+06
59000 	 1.7007e+06
60000 	 989532
'''.strip()

from crip.physics import calcMuWater
# Since energy unit is eV in our spectrum, we should pass parameter 'eV'.
print('\\mu_water = {:.6f} mm^-1'.format(calcMuWater(SpectrumFile, 'eV')))
