from crip.physics import *
from crip.io import imreadRaw
from matplotlib import pyplot as plt
import numpy as np

# Spectrum
# Spectrum类中存放了能量omega(ndarray:151,按照dict的键值对配对的，不存在的keV部分为0)，以及所有能量总和sumOmega(float)
# Spectrum.fromFile(spec_path,"keV")用来读取能谱信息，具体的能谱格式可参照spec_keV_space_energy.txt。
spec_path = r"../Data/PhysicsData/spec_keV_space_energy.txt"
spec = Spectrum.fromFile(spec_path,"keV")

# SpectrumFile = '''
# 0  100
# 1  90
# 2  80
# 3 -1
# '''
# spec = Spectrum.fromText(SpectrumFile,unit="keV")

# getClassicDensity() 参数均要大写
WaterDensity = getClassicDensity("Water")

# Atten.fromBuiltIn('material')获取0-150keV下的atten且atten的单位为mm-1
atten = Atten.fromBuiltIn('Water')

# calcMu()计算μ_water_ref,它的单位为mm-1
mu = calcMu(atten,spec,"PCD")


AlMap = imreadRaw(r"../Data/PhysicsData/sgm_al.raw", 800, 976, nSlice=1)
WaterMap = imreadRaw(r"../Data/PhysicsData/sgm_water.raw", 800, 976, nSlice=1)

AlAtten = Atten.fromBuiltIn("Al")
WaterAtten = Atten.fromBuiltIn("Water")

plt.imshow(AlMap, cmap="gray")
plt.show()

plt.imshow(WaterMap, cmap="gray")
plt.show()

# flat其实就是计算空气的值，即所有能量的累加
flat = forwardProjectWithSpectrum([], [], spec, 'EID')
sgmNew = []

sgmNew.append(
    forwardProjectWithSpectrum([AlMap, WaterMap], [AlAtten, WaterAtten], spec, 'EID', fastSkip=True, flat=flat))
    # forwardProjectWithSpectrum([alm], [AlAtten], spec, 'EID', fastSkip=True, flat=flat))

plt.imshow(sgmNew[0], cmap="gray")
plt.show()

postlog = -np.log(sgmNew[0]/flat)

plt.imshow(postlog, cmap="gray")
plt.show()
