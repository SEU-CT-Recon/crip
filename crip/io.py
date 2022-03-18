'''
    I/O module of crip.

    https://github.com/z0gSh1u/crip
'''

__all__ = ['listDirectory', 'imreadDicom', 'imreadRaw', 'imwriteRaw', 'imreadTiff', 'imwriteTiff']

import os
import numpy as np
import tifffile
import pydicom
import natsort

from .utils import *
from ._typing import *


def listDirectory(folder: str, sort='nat', style='filename', natAlg='default', reverse=False):
    '''
        List files under `folder` and sort using `"nat"` (natural) or \\
        `"dict"` (dictionary) order. The return `style` can be `filename`, `fullpath` or `both`.
    '''
    cripAssert(sort in ['nat', 'dict'], 'Invalid sort.')
    cripAssert(style in ['filename', 'fullpath', 'both'], 'Invalid style.')
    cripAssert(natAlg in ['default', 'locale'], 'Invalid natAlg.')

    files = os.listdir(folder)
    files = sorted(files, reverse=reverse) if sort == 'dict' else natsort.natsorted(
        files, reverse=reverse, alg={
            'default': natsort.ns.DEFAULT,
            'locale': natsort.ns.LOCALE
        }[natAlg])

    if style == 'filename':
        return files
    elif style == 'fullpath':
        return [os.path.join(folder, file) for file in files]
    elif style == 'both':
        return zip([os.path.join(folder, file) for file in files], files)


def imreadDicom(path: str, dtype=None) -> np.ndarray:
    '''
        Read DICOM file. Return numpy array.

        Convert dtype with `dtype != None`.
    '''
    dcm = pydicom.read_file(path)

    if dtype is not None:
        return np.array(dcm.pixel_array).astype(dtype)
    else:
        return np.array(dcm.pixel_array)


def imreadRaw(path: str, h: int, w: int, dtype=DefaultFloatDType, nSlice: int = 1, offset: int = 0) -> np.ndarray:
    '''
        Read binary raw file. Return numpy array with shape `(nSlice, h, w)`. `offset` in bytes.
    '''
    with open(path, 'rb') as fp:
        fp.seek(offset)
        arr = np.frombuffer(fp.read(), dtype=dtype).reshape((nSlice, h, w)).squeeze()

    return arr


@ConvertListNDArray
def imwriteRaw(img: TwoOrThreeD, path: str, dtype=None):
    '''
        Write raw file. Convert dtype with `dtype != None`.
    '''
    if dtype is not None:
        img = img.astype(dtype)

    with open(path, 'wb') as fp:
        fp.write(img.flatten().tobytes())


def imreadTiff(path: str, dtype=None):
    '''
        Read TIFF file. Return numpy array. Convert dtype with `dtype != None`.
    '''
    if dtype is not None:
        return np.array(tifffile.imread(path)).astype(dtype)
    else:
        return np.array(tifffile.imread(path))


@ConvertListNDArray
def imwriteTiff(img: TwoOrThreeD, path: str, dtype=None):
    '''
        Write TIFF file. Convert dtype with `dtype != None`.
        
        Note that any floating dtype will be converted to float32.
    '''
    if dtype is not None:
        img = img.astype(dtype)
    if isFloatDtype(img.dtype):
        img = img.astype(np.float32)  # always use float32 for floating image.

    tifffile.imwrite(path, img)
