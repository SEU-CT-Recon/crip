'''
    I/O module of crip.

    by z0gSh1u @ https://github.com/z0gSh1u/crip
'''

import os
import numpy as np
import tifffile
import pydicom
import natsort

from .utils import *

def listDirectory(folder, sort='nat', joinFolder=False, reverse=False):
    """
        List files under `folder` and `sort` using `"nat"` (natural) or \\
        `"dict"` (dictionary) order. Set `joinFolder` to True to get paths, \\
        otherwise filenames only.
    """
    cripAssert(sort == 'nat' or sort == 'dict', 'Invalid `sort` method.')

    files = os.listdir(folder)
    files = sorted(files, reverse=reverse) if sort == 'dict' else natsort.natsorted(files, reverse=reverse)
    if joinFolder:
        files = [os.path.join(folder, file) for file in files]

    return files


def imreadDicom(path):
    """
        Read DICOM file. Return numpy array.
    """
    dcm = pydicom.read_file(path)
    return np.array(dcm.pixel_array)


def imreadRaw(path, h, w, dtype=np.float32, nSlice=1, offset=0):
    """
        Read binary raw file. Return numpy array. `offset` in bytes.
    """
    with open(path, 'rb') as fp:
        fp.seek(offset)
        arr = np.frombuffer(fp.read(), dtype=dtype).reshape((nSlice, h, w)).squeeze()
    return arr


def imwriteRaw(img, path, dtype='keep'):
    """
        Write raw file.
    """
    if dtype != 'keep':
        img = img.astype(dtype)
    with open(path, 'wb') as fp:
        fp.write(img.flatten().tobytes())


def imreadTiff(path: str):
    """
        Read TIFF file. Return numpy array.
    """
    return np.array(tifffile.imread(path))


def imwriteTiff(img: np.ndarray, path: str, dtype=None):
    """
        Write TIFF file.
    """
    # TODO float* -> float32

    if dtype is not None:
        img = img.astype(dtype)
    tifffile.imwrite(path, img)


def readFileText():
    pass
