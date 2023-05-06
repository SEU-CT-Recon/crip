from crip.physics import calcMu, Spectrum, Material

gd = Material.fromBuiltIn('Gd')

spec70 = Spectrum.fromText('70 100000\n71 -1')
spec60 = Spectrum.fromText('60 100000\n61 -1')
spec50 = Spectrum.fromText('50 100000\n51 -1')
spec40 = Spectrum.fromText('40 100000\n41 -1')

mu70 = calcMu(gd, spec70, 'PCD') * 10
mu60 = calcMu(gd, spec60, 'PCD') * 10
mu50 = calcMu(gd, spec50, 'PCD') * 10
mu40 = calcMu(gd, spec40, 'PCD') * 10

print('{:.4f}'.format(mu70))
print('{:.4f}'.format(mu60))
print('{:.4f}'.format(mu50))
print('{:.4f}'.format(mu40))
