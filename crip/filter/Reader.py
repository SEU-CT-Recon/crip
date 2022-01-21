import glob
import numpy as np
from tqdm import tqdm

def RawSingleReader(glob_:str, h: int, w: int, dtype, offset = 0):
    files = glob.glob(glob)
    images = np.zeros((len(files), h, w), dtype=dtype)
    for i, file in tqdm(enumerate(files)):
        with open(file, 'rb') as fp:
            fp.seek(offset)
            arr = np.frombuffer(fp.read(), dtype=dtype).reshape((h, w))
            images[i, :, :] = arr
    return images

def RawStackReader(path: str, h :int ,w:int, nSlice:int, dtype, offset=0):
    with open(path, 'rb') as fp:
        fp.seek(offset)
        arr = np.frombuffer(fp.read(), dtype=dtype).reshape((nSlice,h,w))
    return arr