'''
    Internal typing module of crip.

    https://github.com/SEU-CT-Recon/crip
'''

import numpy as np
from typing import List, Union, Any, Dict, Callable, Iterable, Tuple

Or = Union

DefaultFloatDType = np.float32
DefaultEnergyUnit = 'keV'
BuiltInAttenEnergyUnit = 'MeV'
DefaultMuUnit = 'mm-1'

UintLike = Or[np.uint8, np.uint16, np.uint32, np.uint64]
SignedIntLike = Or[int, np.int8, np.int16, np.int32, np.int64]
IntLike = Or[UintLike, SignedIntLike]

try:
    FloatLike = Or[np.float16, np.float32, np.float64, np.float128, float]
except:
    FloatLike = Or[np.float16, np.float32, np.float64, float]

NDArray = np.ndarray
ListNDArray = List[NDArray]

TwoD = NDArray
ThreeD = Or[NDArray, ListNDArray]
TwoOrThreeD = Or[TwoD, ThreeD]
