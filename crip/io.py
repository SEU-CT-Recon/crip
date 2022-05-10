'''
    I/O module of crip.

    https://github.com/z0gSh1u/crip
'''

__all__ = [
    'listDirectory', 'imreadDicom', 'readDicom', 'imreadRaw', 'imwriteRaw', 'imreadTiff', 'imwriteTiff', 'CTPARAMS',
    'fetchCTParam'
]

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


def readDicom(path: str) -> pydicom.Dataset:
    '''
        Read DICOM file as pydicom object.
    '''
    return pydicom.read_file(path)


def imreadRaw(path: str, h: int, w: int, dtype=DefaultFloatDType, nSlice: int = 1, offset: int = 0, order='CHW') -> np.ndarray:
    '''
        Read binary raw file. Return numpy array with shape `(nSlice, h, w)`. `offset` in bytes.
    '''
    cripAssert(order in ['CHW', 'HWC'], 'Invalid order.')

    with open(path, 'rb') as fp:
        fp.seek(offset)
        arr = np.frombuffer(fp.read(), dtype=dtype).reshape((nSlice, h, w)).squeeze()
        if order == 'HWC':
            arr = np.transpose(arr, (2, 0, 1))

    return arr


@ConvertListNDArray
def imwriteRaw(img: TwoOrThreeD, path: str, dtype=None, order='CHW'):
    '''
        Write raw file. Convert dtype with `dtype != None`.
    '''
    cripAssert(order in ['CHW', 'HWC'], 'Invalid order.')

    if dtype is not None:
        img = img.astype(dtype)

    with open(path, 'wb') as fp:
        if order == 'HWC':
            img = np.transpose(img, (1, 2, 0))
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


def _CTPARAM(loc, type_):
    return {'loc': loc, 'type': type_}


# These CT parameters will be retrieved.
CTPARAMS = {
    # For WC sometimes two values will be fetched. Usually they are the same.
    # If not, it means that the view is recommended to be displayed using to windows.
    # So as WD.
    'Window Center': _CTPARAM([0x0028, 0x1050], str),
    'Window Width': _CTPARAM([0x0028, 0x1051], str),

    # Manufacturer
    'Manufacturer': _CTPARAM([0x0008, 0x0070], str),
    'Manufacturer Model Name': _CTPARAM([0x0008, 0x1090], str),

    # Patient status
    'Body Part Examined': _CTPARAM([0x0018, 0x0015], str),
    'Patient Position': _CTPARAM([0x0018, 0x5100], str),  # (H/F)F(S/P)

    # X-Ray exposure
    'KVP': _CTPARAM([0x0018, 0x0060], float),  # kVpeak
    'X Ray Tube Current': _CTPARAM([0x0018, 0x1151], float),  # mA
    'Exposure Time': _CTPARAM([0x0018, 0x1150], float),
    'Exposure': _CTPARAM([0x0018, 0x1152], float),

    # CT Reconstruction
    'Slice Thickness': _CTPARAM([0x0018, 0x0050], float),
    'Data Collection Diameter': _CTPARAM([0x0018, 0x0090], float),
    'Reconstruction Diameter': _CTPARAM([0x0018, 0x1100], float),
    'Rows': _CTPARAM([0x0028, 0x0010], int),
    'Columns': _CTPARAM([0x0028, 0x0011], int),
    'Pixel Spacing': _CTPARAM([0x0028, 0x0030], str),  # u/v, mm
    'Distance Source To Detector': _CTPARAM([0x0018, 0x1110], float),  # SDD (S-Image-D), mm
    'Distance Source To Patient': _CTPARAM([0x0018, 0x1111], float),  # SOD (S-Isocenter-D), mm
    'Rotation Direction': _CTPARAM([0x0018, 0x1140], str),  # CW/CCW
    'Bits Allocated': _CTPARAM([0x0028, 0x0100], int),

    # Table
    'Table Height': _CTPARAM([0x0018, 0x1130], float),
    'Table Speed': _CTPARAM([0x0018, 0x9309], float),
    'Table Feed Per Rotation': _CTPARAM([0x0018, 0x9310], float),

    # CT Value rescaling
    # e.g.  HU = 1X-1024
    'Rescale Intercept': _CTPARAM([0x0028, 0x1052], float),  # b
    'Rescale Slope': _CTPARAM([0x0028, 0x1053], float),  # k
    'Rescale Type': _CTPARAM([0x0028, 0x1054], str),

    # For helical CT
    'Spiral Pitch Factor': _CTPARAM([0x0018, 0x9311], float)
}


def fetchCTParam(dicom: pydicom.Dataset, key: str):
    '''
        Fetch CT related parameter from DICOM file.

        @See CTPARAMS in the source code for available keys.
    '''
    metaParam = CTPARAMS[key]
    if metaParam is None:
        return None
    try:
        value = metaParam['type'](dicom[metaParam['loc'][0], metaParam['loc'][1]].value)
    except:
        return None
    return value
