from crip.postprocess import *
from crip.io import imreadRaw
from crip.physics import Atten,Spectrum,calcMu
from matplotlib import pyplot as plt

spec_path = "../Data/PhysicsData/spec_keV_space_energy.txt"
spec = Spectrum.fromFile(spec_path,"keV")
atten = Atten.fromBuiltIn("Water")
muWater = calcMu(atten,spec,"PCD")

rec_80_mu = imreadRaw("../Data/DeData/rec_80.raw", 512, 512, nSlice=1)

rec_80_mu = fovCrop(rec_80_mu,256,0)

plt.imshow(rec_80_mu, cmap="gray")
plt.show()

rec_80_HU = muToHU(rec_80_mu,muWater)

rec_80_mu_2 = huToMu(rec_80_HU,muWater)

# huNoRescale函数就是对HU图像做加1000的操作
rec_80_mu_rescale = huNoRescale(rec_80_HU)

print(1+1)