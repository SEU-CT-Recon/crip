'''
    This example uses crip to perform truncation artifact correction on projections
    using existing projection and FDK reconstruction algorithm.

    by CandleHouse @ https://github.com/z0gSh1u/crip
'''
import matplotlib.pyplot as plt
import numpy as np
from crip.preprocess import *
from crip.io import *
from crip.utils import *
from crip.postprocess import *


# related directory and file path
proj_folder = ''
sgm_pad_path = ''
config_path = ''
rec_img_path = ''

# 1. combine all raw image under the directory and convert to sinogram
h, w = 237, 324
combine_rawImage = combineRawImageUnderDirectory(proj_folder, h, w, reverse=True)
sgm = projectionsToSinograms(combine_rawImage)

# 2. padding all sinogram and save
pad_sgm = []
for slice in range(sgm.shape[0]):
    pad_sgm.append(padSinogram(sgm[slice], padding=20, smootherDecay=True))
imwriteRaw(np.array(pad_sgm), sgm_pad_path)

# 3. image reconstruction
mgfbp(config_path)

# 4. calculate radius and plot
r_reference = fovCropRadiusReference(SOD=750, SDD=1250, detector_width=399.168, rec_pixel_width=0.5)
rec_img = np.fromfile(rec_img_path, dtype=np.float32).reshape(60, 512, 512)
plt.imshow(rec_img[0], cmap='gray')
drawCircle(rec_img[0], r=r_reference)
plt.show()

# 5. crop and save
crop_rec_img = cropCircleFOV(rec_img, r_reference-2)
imwriteTiff(crop_rec_img, './rec/crop_f_abdomen.tif')
