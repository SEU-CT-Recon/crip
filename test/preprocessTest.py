from crip.preprocess import *
from crip.io import imreadRaw,imwriteRaw
from matplotlib import pyplot as plt

sgm = imreadRaw("../Data/LowdoseData/sgm_high_gt.raw", 800, 976, nSlice=1)

sgm_gauss = injectGaussianNoise(sgm,0.0001)
imwriteRaw(sgm_gauss,"./sgm_ga.raw")

sgm_poisson = injectPoissonNoise(sgm,'postlog',10000)

imwriteRaw(sgm_poisson,"./sgm_po.raw")
error = sgm - sgm_poisson
imwriteRaw(error,"./error.raw")
plt.imshow(sgm, cmap="gray")
plt.show()

plt.imshow(error, cmap="gray")
plt.show()