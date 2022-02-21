'''
    This example uses crip to perform air correction on projections
    using existing air projection.

    by z0gSh1u @ https://github.com/z0gSh1u/crip
'''

import _importcrip

import os
import numpy as np
from crip.io import imreadTiff, imwriteTiff, imreadRaw
from crip.preprocess import flatDarkFieldCorrection

# Read air projection from .raw file.
H = 1024
W = 1536
flatField = imreadRaw('/path/to/flatField.raw', H, W)

sourceProjPath = '/path/to/source/projections'
destProjPath = '/path/to/save'

# Iterate projections, correct them and save.
for file in os.listdir(sourceProjPath):
    proj = imreadTiff(os.path.join(sourceProjPath, file))
    airCorrected = flatDarkFieldCorrection(proj, flatField)
    imwriteTiff(airCorrected, os.path.join(destProjPath, file), dtype=np.float32)
