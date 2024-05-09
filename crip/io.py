'''
    I/O module of crip.

    https://github.com/SEU-CT-Recon/crip
'''

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
    ''' List files under `folder` and `sort` in `nat`(ural) or `dict`(ionary) order.
        Return `style` can be `filename`, `fullpath` or `both` (path, name) tuple.
        Results can be filtered by `match` and reversed by `reverse`.
    '''
    cripAssert(sort in ['nat', 'dict'], f'Invalid `sort`: {sort}.')
    cripAssert(style in ['filename', 'fullpath', 'both'], f'Invalid `style`: {style}.')

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


def imreadDicom(path: str, dtype=None, attrs: Or[None, Dict[str, Any]] = None) -> NDArray:
    ''' Read DICOM file, return numpy array. Use `attrs` to supplement DICOM tags for non-standard images.
        You should be very careful whether the Rescale is cancelled for CT images.
        Convert dtype when `dtype` is not `None`.
    '''
    dcm = pydicom.dcmread(path)

    if attrs is not None:
        for key in attrs:
            dcm.__setattr__(key, attrs[key])

    if dtype is not None:
        return np.array(dcm.pixel_array).astype(dtype)
    else:
        return np.array(dcm.pixel_array)


def imreadDicoms(dir_: str, **kwargs) -> NDArray:
    ''' Read series of DICOM files in a directory. `kwargs` is forwarded to `imreadDicom`.
    '''
    imgs = [imreadDicom(x, **kwargs) for x in listDirectory(dir_, style='fullpath')]

    return np.array(imgs)


def readDicom(path: str) -> pydicom.Dataset:
    ''' Read DICOM file as pydicom Dataset object.
    '''
    return pydicom.read_file(path)


def imreadRaw(path: str,
              h: int,
              w: int,
              dtype=DefaultFloatDType,
              nSlice: int = 1,
              offset: int = 0,
              gap: int = 0,
              swapEndian=False) -> NDArray:
    ''' Read binary raw file stored in `dtype`.
        Return numpy array with shape `(min(nSlice, actualNSlice), h, w)`.
        Input `nSlice` can be unequal to the actual number of slices.
        Allow setting `offset` from head and `gap` between images in bytes.
        Allow change the endian-ness to the contrary of your machine by `swapEndian`.
        This function acts like ImageJ's `Import Raw`.
    '''
    simpleValidate([
        offset >= 0,
        h > 0 and w > 0 and nSlice > 0,
    ])

    slices = []
    sliceBytes = h * w * np.dtype(dtype).itemsize  # bytes per slice
    fileBytes = os.path.getsize(path)  # actual bytes of the file

    fp = open(path, 'rb')
    fp.seek(offset)
    for _ in range(nSlice):
        slices.append(np.frombuffer(fp.read(sliceBytes), dtype=dtype).reshape((h, w)).squeeze())
        fp.seek(gap, os.SEEK_CUR)
        if fp.tell() >= fileBytes:
            break
    fp.close()

    slices = np.array(slices).astype(dtype)

    return slices if not swapEndian else slices.byteswap(inplace=True)


def imreadRaws(dir_: str, **kwargs: Any) -> NDArray:
    ''' Read series of raw images in directory. `kwargs` will be forwarded to `imreadRaw`.
    '''
    imgs = [imreadRaw(x, **kwargs) for x in listDirectory(dir_, style='fullpath')]

    return np.array(imgs)


@ConvertListNDArray
def imwriteRaw(img: TwoOrThreeD, path: str, dtype=None, order='CHW', swapEndian=False, fortranOrder=False):
    ''' Write `img` to raw file `path`. Convert dtype when `dtype` is not `None`.
        `order` can be [CHW or HWC].
        Allow change the endian-ness to the contrary of your machine by `swapEndian`.
        Allow using Fortran order (Column-major) by setting `fortranOrder`.
    '''
    cripAssert(order in ['CHW', 'HWC'], f'Invalid order: {order}.')

    if dtype is not None:
        img = img.astype(dtype)
    if order == 'HWC':
        img = chw2hwc(img)
    if swapEndian:
        img = img.byteswap(inplace=False)

    with open(path, 'wb') as fp:
        fp.write(img.tobytes('F' if fortranOrder else 'C'))


def imreadTiff(path: str, dtype=None) -> NDArray:
    ''' Read TIFF file, return numpy array. Convert dtype when `dtype` is not `None`.
    '''
    if dtype is not None:
        return np.array(tifffile.imread(path)).astype(dtype)
    else:
        return np.array(tifffile.imread(path))


def imreadTiffs(dir_: str, **kwargs) -> NDArray:
    ''' Read series of tiff images in directory. `kwargs` will be forwarded to `imreadTiff`.
    '''
    imgs = [imreadTiff(x, **kwargs) for x in listDirectory(dir_, style='fullpath')]

    return np.array(imgs)


@ConvertListNDArray
def imwriteTiff(img: TwoOrThreeD, path: str, dtype=None):
    ''' Write TIFF file. Convert dtype if `dtype` is not `None`.
        All floating dtype will be converted to float32.
    '''
    if dtype is not None:
        img = img.astype(dtype)
    if isFloatDtype(img.dtype):
        img = img.astype(np.float32)

    tifffile.imwrite(path, img)


def readEVI(path: str) -> Tuple[NDArray, Dict[str, Any]]:
    ''' Read EVI file saved by XCounter Hydra PCD detector. Return the images and metadata.
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

    # mapping between EVI file prelude and metadata to fill in
    mapping = {
        'Image_Type': 'ImageType',
        'Width': 'Width',
        'Height': 'Height',
        'Nr_of_images': 'NumberOfImages',
        'Offset_To_First_Image': 'OffsetToFirstImage',
        'Gap_between_iamges_in_bytes': 'GapBetweenImages',  # for compatibility
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


def imreadEVI(path: str) -> NDArray:
    ''' Read EVI file saved by XCounter Hydra PCD detector. Return the images only.
    '''
    return readEVI(path)[0]


# These CT parameters can be retrieved.
CTPARAMS = {
    # Manufacturer
    'Manufacturer': ([0x0008, 0x0070], str),
    'Manufacturer Model Name': ([0x0008, 0x1090], str),

    # Study
    'Series Instance UID': ([0x0020, 0x000E], str),

    # Patient status
    'Body Part Examined': ([0x0018, 0x0015], str),
    'Patient Position': ([0x0018, 0x5100], str),  # (H/F)F(S/P)

    # X-Ray exposure
    'KVP': ([0x0018, 0x0060], float),
    'X Ray Tube Current': ([0x0018, 0x1151], float),  # mA
    'Exposure Time': ([0x0018, 0x1150], float),
    'Exposure': ([0x0018, 0x1152], float),

    # CT Reconstruction
    'Slice Thickness': ([0x0018, 0x0050], float),
    'Data Collection Diameter': ([0x0018, 0x0090], float),
    'Reconstruction Diameter': ([0x0018, 0x1100], float),
    'Rows': ([0x0028, 0x0010], int),
    'Columns': ([0x0028, 0x0011], int),
    'Pixel Spacing': ([0x0028, 0x0030], str),  # u/v, mm
    'Distance Source To Detector': ([0x0018, 0x1110], float),  # mm
    'Distance Source To Patient': ([0x0018, 0x1111], float),  # mm
    'Rotation Direction': ([0x0018, 0x1140], str),  # CW/CCW
    'Bits Allocated': ([0x0028, 0x0100], int),

    # Table
    'Table Height': ([0x0018, 0x1130], float),
    'Table Speed': ([0x0018, 0x9309], float),
    'Table Feed Per Rotation': ([0x0018, 0x9310], float),

    # CT value rescaling
    'Rescale Intercept': ([0x0028, 0x1052], float),
    'Rescale Slope': ([0x0028, 0x1053], float),
    'Rescale Type': ([0x0028, 0x1054], str),

    # Helical CT
    'Spiral Pitch Factor': ([0x0018, 0x9311], float)
}


def fetchCTParam(dicom: pydicom.Dataset, key: str) -> Any:
    ''' Fetch CT metadata from DICOM Dataset read by `readDicom`
        Refer to `CTPARAMS` for available keys.
    '''
    cripAssert(key in CTPARAMS.keys(), f'Invalid key: {key}.')

    addr, cast = CTPARAMS[key]

    return cast(dicom[addr[0], addr[1]].value)
