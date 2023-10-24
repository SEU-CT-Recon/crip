from crip.io import listDirectory, imreadRaw,imwriteRaw,imreadDicom,imwriteTiff
from crip.physics import Atten, Spectrum, calcMu
from crip.de import *
from crip.postprocess import fovCrop
from crip.postprocess import huNoRescale
from matplotlib import pyplot as plt
import numpy as np

# Decompose in the projection domain.

LowSpecPath = r"Data\PhysicsData\spec_80.txt"
HighSpecPath = r"Data\PhysicsData\spec_140.txt"

LowSpec = Spectrum.fromFile(LowSpecPath, 'keV')
HighSpec = Spectrum.fromFile(HighSpecPath, 'keV')

# Base1 = Atten.fromBuiltIn('PMMA')
# Base2 = Atten.fromBuiltIn('Al')
Base1 = Atten.fromBuiltIn("Water")
Base2 = Atten.fromBuiltIn("Bone")
#
#
# LowEProj = imreadRaw("Data\DeData\sgm_80.raw", 1000, 1000, nSlice=1)
# HighEProj = imreadRaw("Data\DeData\sgm_140.raw", 1000, 1000, nSlice=1)
#

# Base1Range = range(0, 20 + 2, 2) # The length range [mm] to fit \mu.
#
# Base2Range = range(0, 5 + 1, 1)
#
# # calcAttenedSpec函数计算预硬化后的能谱，len(atten) == len(length) atten有几个，那么length就要传几个
# # afterSpec = calcAttenedSpec(LowSpec,Base1,0.2)
# # calcPostLog函数计算postlog，len(atten) == len(length),如果长度不一那就通过遍历来实现
# # postlog = calcPostLog(LowSpec,Base1,np.arange(0,31))
#
# Alpha, Beta = deDecompGetCoeff(LowSpec, HighSpec, Base1, Base1Range, Base2, Base2Range)
# p_Decomp1, p_Decomp2 = deDecompProj(LowEProj, HighEProj, Alpha, Beta)
#
# imwriteTiff(p_Decomp1,"./p_Decomp1.tif")
# imwriteTiff(p_Decomp2,"./p_Decomp2.tif")
#
# sgm_pmma = imreadRaw(r"Data\DeData\sgm_water.raw",1000,1000,nSlice=1)
# sgm_al = imreadRaw(r"Data\DeData\sgm_al.raw",1000,1000,nSlice=1)
#
# plt.imshow(p_Decomp1, cmap="gray")
# plt.show()
#
# plt.imshow(p_Decomp2, cmap="gray")
# plt.show()

# Decompose in the image domain.

lowAveEnergy = np.sum(LowSpec.omega * np.arange(0,150+1,1)) / LowSpec.sumOmega   # lowAveEnergy = 43.383
highAveEnergy = np.sum(HighSpec.omega * np.arange(0,150+1,1)) / HighSpec.sumOmega   # highAveEnergy = 68.266

mu1Low = Base1.mu[43]
mu2Low = Base2.mu[43]
mu1High = Base1.mu[68]
mu2High = Base2.mu[68]

lowMuWater = calcMu(Base1,LowSpec,"EID")
highMuWater = calcMu(Base1,HighSpec,"EID")

Hu1Low = (mu1Low-lowMuWater )/lowMuWater *1000+1000
Hu2Low = (mu2Low-lowMuWater )/lowMuWater *1000+1000
Hu1High = (mu1High-highMuWater)/highMuWater*1000+1000
Hu2High = (mu2High-highMuWater)/highMuWater*1000+1000

# mu1Lo = calcMu(Base1,LowSpec,"EID")
# mu2Lo = calcMu(Base2,LowSpec,"EID")
# mu1Hig = calcMu(Base1,HighSpec,"EID")
# mu2Hig = calcMu(Base2,HighSpec,"EID")

real_100 = imreadDicom(
    r"../Data/DeData/LI_XIANG.CT.ABDOMEN_11_ABDOMEN_DE_BT_(ADULT).0010.0001.2017.08.30.09.57.48.421875.782907008.IMA")
real_140 = imreadDicom(
    r"../Data/DeData/LI_XIANG.CT.ABDOMEN_11_ABDOMEN_DE_BT_(ADULT).0011.0001.2017.08.30.09.57.48.421875.782927274.IMA")
real_Decomp1, real_Decomp2 = deDecompRecon(real_100,real_140,Hu1Low,Hu1High,Hu2Low,Hu2High)

real_water = fovCrop(real_Decomp1, 256, 0)
real_bone = fovCrop(real_Decomp2, 256, 0)
imwriteTiff(real_water, '../Data/DeData/real_water.tif')
imwriteTiff(real_bone, '../Data/DeData/real_bone.tif')

lowSlice = imreadRaw("../Data/DeData/rec_80.raw", 512, 512)
highSlice = imreadRaw("../Data/DeData/rec_140.raw", 512, 512)
I_Decomp1, I_Decomp2 = deDecompRecon(lowSlice,highSlice,mu1Low,mu1High,mu2Low,mu2High)

# imwriteTiff(I_Decomp1,"./I_Decomp1_mu.tif")
# imwriteTiff(I_Decomp2,"./I_Decomp2_mu.tif")

plt.imshow(I_Decomp1, cmap="gray")
plt.show()

plt.imshow(I_Decomp2, cmap="gray")
plt.show()

print(1+1)