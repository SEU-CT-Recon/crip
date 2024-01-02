'''
    I/O module of crip.

    https://github.com/SEU-CT-Recon/crip
'''

__all__ = [
    'listDirectory', 'imreadDicom', 'readDicom', 'imreadRaw', 'imwriteRaw', 'imreadTiff', 'imwriteTiff', 'CTPARAMS',
    'fetchCTParam', 'readEVI', 'imreadEVI'
]

import os
import re
import numpy as np
import tifffile
import pydicom
import natsort

from .utils import *
from ._typing import *


def listDirectory(folder: str,
                  style='filename',
                  match: Or[re.Pattern, str, None] = None,
                  sort='nat',
                  reverse=False) -> Or[List[str], Iterable[Tuple[str, str]]]:
    '''
        List files under `folder` and sort in `nat`(ural) or `dict`(ionary) order. 
        Return `style` can be `filename`, `fullpath` or `both` (path, name) tuple.
    '''
    cripAssert(sort in ['nat', 'dict'], 'Invalid sort.')
    cripAssert(style in ['filename', 'fullpath', 'both'], 'Invalid style.')

    files = os.listdir(folder)

    if match is not None:
        if isinstance(match, str):
            match = re.compile(match)
        files = list(filter(lambda x: match.search(x), files))

    files = sorted(files, reverse=reverse) if sort == 'dict' else natsort.natsorted(
        files, reverse=reverse, alg=natsort.ns.DEFAULT)

    if style == 'filename':
        return files
    elif style == 'fullpath':
        return [os.path.join(folder, file) for file in files]
    elif style == 'both':
        return zip([os.path.join(folder, file) for file in files], files)


def imreadDicom(path: str, dtype=None, attrs: Or[None, Dict[str, Any]] = None) -> np.ndarray:
    '''
        Read DICOM file. Return numpy array. Use `attrs` to supplement DICOM tags for non-standard images.
        You should be very careful about the whether Rescale Slope is cancelled for CT images.
    
        Convert dtype with `dtype != None`.
    '''
    dcm = pydicom.dcmread(path)

    if attrs is not None:
        for key in attrs:
            dcm.__setattr__(key, attrs[key])

    if dtype is not None:
        return np.array(dcm.pixel_array).astype(dtype)
    else:
        return np.array(dcm.pixel_array)


def imreadDicoms(dir_: str, dtype=None, attrs: Or[None, Dict[str, Any]] = None) -> np.ndarray:
    '''
        Read series of DICOM files in directory.
    '''
    imgs = [imreadDicom(x, dtype, attrs) for x in listDirectory(dir_, style='fullpath')]

    return np.array(imgs)


def readDicom(path: str) -> pydicom.Dataset:
    '''
        Read DICOM file as pydicom object.
    '''
    return pydicom.read_file(path)


def imreadRaw(path: str,
              h: int,
              w: int,
              dtype=DefaultFloatDType,
              nSlice: int = 1,
              offset: int = 0,
              gap: int = 0,
              order='CHW') -> np.ndarray:
    '''
        Read binary raw file. Return numpy array with shape `(nSlice, h, w)`. `offset` from head in bytes.
        `gap` between images in bytes.
    '''
    cripAssert(order in ['CHW', 'HWC'], 'Invalid order.')

    with open(path, 'rb') as fp:
        fp.seek(offset)
        if gap == 0:
            arr = np.frombuffer(fp.read(), dtype=dtype).reshape((nSlice, h, w)).squeeze()
        else:
            cripAssert(order == 'CHW', 'gap is only availble in CHW order.')
            imageBytes = h * w * np.dtype(dtype).itemsize
            arr = np.zeros((nSlice, h, w), dtype=dtype)
            for i in range(nSlice):
                arr[i, ...] = np.frombuffer(fp.read(imageBytes), dtype=dtype).reshape((h, w)).squeeze()
                fp.seek(gap, os.SEEK_CUR)

    if order == 'HWC':
        arr = np.transpose(arr, (2, 0, 1))

    return arr


def imreadRaws(dir_: str,
               h: int,
               w: int,
               dtype=DefaultFloatDType,
               nSlice: int = 1,
               offset: int = 0,
               gap: int = 0,
               order='CHW'):
    '''
        Read series of raw images in directory.
    '''
    imgs = [imreadRaw(x, h, w, dtype, nSlice, offset, gap, order) for x in listDirectory(dir_, style='fullpath')]

    return np.array(imgs)


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


def imreadTiffs(dir_: str, dtype=None):
    '''
        Read series of tiff images in directory. Will add one dim.
    '''
    imgs = [imreadTiff(x, dtype) for x in listDirectory(dir_, style='fullpath')]

    return np.array(imgs)


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


def readEVI(path: str):
    '''
        Read EVI file saved by XCounter Hydra PCD detector. Return the images and metadata.
    '''
    metadata = {
        'ImageType': None,
        'Width': None,
        'Height': None,
        'NumberOfImages': None,
        'OffsetToFirstImage': None,
        'GapBetweenImages': None,
        'FrameRate': None,
        'LowThreshold': None,
        'HighThreshold': None,
    }

    mapping = {
        'Image_Type': 'ImageType',
        'Width': 'Width',
        'Height': 'Height',
        'Nr_of_images': 'NumberOfImages',
        'Offset_To_First_Image': 'OffsetToFirstImage',
        # interesting
        'Gap_between_iamges_in_bytes': 'GapBetweenImages',
        'Gap_between_images_in_bytes': 'GapBetweenImages',
        'Frame_Rate_HZ': 'FrameRate',
        'Low_Thresholds_KV': 'LowThreshold',
        'High_Thresholds_KV': 'HighThreshold',
    }

    take = lambda str, idx: str.split(' ')[idx]
    with open(path, 'r', encoding='utf-8', errors='ignore') as fp:
        line = fp.readline().strip()
        while len(line):
            if line == 'COMMENT':
                break

            key = take(line, 0)
            if key in mapping:
                val = take(line, 1)
                metadata[mapping[key]] = int(val) if str.isdigit(val) else val

            line = fp.readline().strip()

    nones = list(filter(lambda x: x[1] is None, metadata.items()))
    cripAssert(len(nones) == 0, f'Failed to find metadata for {list(map(lambda x: x[0], nones))}')
    cripAssert(metadata['ImageType'] in ['Single', 'Double'], f'Unsupported ImageType: {metadata["ImageType"]}')
    dtype = {'Single': np.float32, 'Double': np.float64}

    img = imreadRaw(path, metadata['Height'], metadata['Width'], dtype[metadata['ImageType']],
                    metadata['NumberOfImages'], metadata['OffsetToFirstImage'], metadata['GapBetweenImages'])

    return img, metadata


def imreadEVI(path: str):
    '''
        Read EVI file saved by XCounter Hydra PCD detector. Return the images only.
    '''
    return readEVI(path)[0]


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

    # Study
    'Series Instance UID': _CTPARAM([0x0020, 0x000E], str),

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
        Fetch CT related parameter from DICOM file. Use `readDicom` to get DICOM Dataset.

        @see CTPARAMS in the source code for available keys.
    '''
    metaParam = CTPARAMS[key]
    if metaParam is None:
        return None
    try:
        value = metaParam['type'](dicom[metaParam['loc'][0], metaParam['loc'][1]].value)
    except:
        return None
    return value
