'''
    I/O module of crip.

    by z0gSh1u @ https://github.com/z0gSh1u/crip
'''

import os
import numpy as np
import tifffile
import pydicom
import natsort


def listDirectory(folder, sort='nat', joinFolder=False):
    """
        List files under `folder` and `sort` using `"nat"` (natural) or \\
        `"dict"` (dictionary) order. Set `joinFolder` to True to get the paths, \\
        otherwise filename only.
    """
    assert sort == 'nat' or sort == 'dict', 'Invalid `sort` method.'
    files = os.listdir(folder)
    files = sorted(files) if sort == 'dict' else natsort.natsorted(files)
    if joinFolder:
        files = [os.path.join(folder, file) for file in files]
    return files


def combineRawImageUnderDirectory(folder, h, w, dtype=np.float32, nSlice=1, offset=0, sort='nat', reverse=False):
    """
        Combine Image under 'folder' and 'sort' using `"nat"` (natural) or `"dict"` (dictionary) order.
        Do: views * (h, w) -> (views, h, w)
    """
    list_dir = natsort.natsorted(listDirectory(folder, sort, joinFolder=True), reverse=reverse)
    views = len(list_dir)
    combine_rawImage = np.zeros((views, h, w), dtype=np.float32)  # dtype frozen

    for i, file in enumerate(list_dir):
        combine_rawImage[i, :, :] = imreadRaw(file, h, w, dtype, nSlice, offset)
    return combine_rawImage


def combineTiffImageUnderDirectory(folder, sort='nat', reverse=False):
    """
        Combine Image under 'folder' and 'sort' using `"nat"` (natural) or `"dict"` (dictionary) order.
        Do: views * (h, w) -> (views, h, w)
    """
    list_dir = natsort.natsorted(listDirectory(folder, sort, joinFolder=True), reverse=reverse)
    views = len(list_dir)
    h, w = imreadTiff(list_dir[0]).shape
    combine_rawImage = np.zeros((views, h, w), dtype=np.float32)  # dtype frozen

    for i, file in enumerate(list_dir):
        combine_rawImage[i, :, :] = imreadTiff(file)
    return combine_rawImage


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


def imreadTiff(path):
    """
        Read TIFF file. Return numpy array.
    """
    return np.array(tifffile.imread(path))


def imwriteRaw(img, path, dtype='keep'):
    """
        Write raw file.
    """
    if dtype != 'keep':
        img = img.astype(dtype)
    with open(path, 'wb') as fp:
        fp.write(img.flatten().tobytes())


def imwriteTiff(img, path, dtype='keep'):
    """
        Write TIFF file.
    """
    if dtype != 'keep':
        img = img.astype(dtype)
    tifffile.imwrite(path, img)
