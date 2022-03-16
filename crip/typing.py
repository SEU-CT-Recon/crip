from typing import Literal
import numpy as np
from typing import *

Or = Union

DefaultFloatDType = np.float32

UintLike = Union[np.uint8, np.uint16, np.uint32, np.uint64]
SignedIntLike = Union[int, np.int8, np.int16, np.int32, np.int64]
IntLike = Union[UintLike, SignedIntLike]

# @see https://stackoverflow.com/questions/58686018
try:
    FloatLike = Union[np.float16, np.float32, np.float64, np.float128, float]
except:
    FloatLike = Union[np.float16, np.float32, np.float64, float]
Proj = np.ndarray
ProjList = List[np.ndarray]
ProjStack = np.ndarray

VoltageUnit = str
# VoltageUnit = Literal['keV', 'eV'] # Future: python 3.8+
def checkVoltageUnit(unit):
    return unit in ['keV', 'eV']
