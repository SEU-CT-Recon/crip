import numpy as np
from typing import *

Or = Union

DefaultFloatDType = np.float32

UintLike = Union[np.uint8, np.uint16, np.uint32, np.uint64]
SignedIntLike = Union[int, np.int8, np.int16, np.int32, np.int64]
IntLike = Union[UintLike, SignedIntLike]
try:
    FloatLike = Union[np.float16, np.float32, np.float64, np.float128, float]
except:
    FloatLike = Union[np.float16, np.float32, np.float64, float]
Proj = np.ndarray
ProjList = List[np.ndarray]
ProjStack = np.ndarray
