import numpy as np
import tifffile
import pydicom

def imreadDicom(path):
    dcm = pydicom.read_file(path)
    return dcm.pixel_array

def imreadRaw(path, h, w, nSlice=1, dtype=np.float32):
    with open(path, 'rb') as fp:
        arr = np.frombuffer(fp.read(), dtype=dtype).reshape((nSlice, h, w)).squeeze()
    return arr


def imreadTiff(path):
    return tifffile.imread(path)


def imwriteRaw(img, path, dtype=np.float32):
    with open(path, 'wb') as fp:
        fp.write(img.astype(dtype).tobytes())


def imwriteTiff(img, path):
    tifffile.imwrite(img, path)