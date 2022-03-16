'''
    This example uses crip to perform truncation artifact correction on projections
    using existing projection and FDK reconstruction algorithm.

    by CandleHouse @ https://github.com/z0gSh1u/crip
'''
import matplotlib.pyplot as plt
from crip.preprocess import *
from crip.io import *
from crip.postprocess import *
from crip.mangoct import *


# related directory and file path
proj_folder = ''
sgm_pad_path = ''
mgfbp_exe = ''
config_path = ''
rec_img_path = ''

# 1. combine all raw image under the directory and convert to sinogram
h, w = 237, 324
projs = []
for file in listDirectory(proj_folder, sort='nat', joinFolder=True, reverse=True):
    projs.append(imreadRaw(file, h, w))
sgm = projectionsToSinograms(stackImages(projs))

# 2. padding all sinogram and save
pad_sgm = []
for slice in range(sgm.shape[0]):
    pad_sgm.append(padSinogram(sgm[slice], padding=20, smootherDecay=True))
imwriteRaw(stackImages(pad_sgm), sgm_pad_path)

# 3. image reconstruction
Mgfbp(mgfbp_exe).exec(config_path)

# 4. calculate radius reference and plot
rec_img = np.fromfile(rec_img_path, dtype=np.float32).reshape(60, 512, 512)
r_reference = fovCropRadiusReference(SOD=750, SDD=1250, detectorWidth=399.168, reconPixelSize=0.5)
x, y = drawCircle(rec_img[0], r=r_reference)
plt.imshow(rec_img[0], cmap='gray')
plt.plot(x, y, 'b-.'), plt.legend(['crop reference']), plt.show()

# 5. crop and save
crop_rec_img = cropCircleFOV(rec_img, r_reference-2)
imwriteTiff(crop_rec_img, './rec/crop_f_abdomen.tif')