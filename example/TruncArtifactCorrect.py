'''
    This example uses crip to perform truncation artifact correction on projections
    using existing projection and FDK reconstruction algorithm.

    by CandleHouse @ https://github.com/SEU-CT-Recon/crip
'''

import numpy as np
import matplotlib.pyplot as plt
from crip.preprocess import *
from crip.io import *
from crip.postprocess import *

# related directory and file path
proj_folder = ''
sgm_pad_path = ''
config_path = ''
rec_img_path = ''

# 1. combine all raw image under the directory and convert to sinogram
h, w = 237, 324
projs = [imreadRaw(file, h, w) for file in listDirectory(proj_folder, sort='nat', reverse=True, style='fullpath')]
sgm = projectionsToSinograms(projs)

# 2. padding all sinogram and save
pad_sgm = []
for slice in range(sgm.shape[0]):
    pad_sgm.append(padSinogram(sgm[slice], padding=20, smootherDecay=True))
imwriteRaw(pad_sgm, sgm_pad_path)

# 3. image reconstruction
# Use your own recon tool.

# 4. calculate radius reference and plot
rec_img = np.fromfile(rec_img_path, dtype=np.float32).reshape(60, 512, 512)
r_reference = fovCropRadius(SOD=750, SDD=1250, detWidth=399.168, reconPixSize=0.5)
x, y = drawCircle(rec_img[0], radius=r_reference)
plt.imshow(rec_img[0], cmap='gray')
plt.plot(x, y, 'b-.'), plt.legend(['crop reference']), plt.show()

# 5. crop and save
crop_rec_img = fovCrop(rec_img, r_reference - 2)
imwriteTiff(crop_rec_img, './rec/crop_f_abdomen.tif')