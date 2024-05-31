'''
    Internal typing module of crip.

    https://github.com/SEU-CT-Recon/crip
'''

import numpy as np
from typing import List, Union, Any, Dict, Callable, Iterable, Tuple, Sequence

DefaultFloatDType = np.float32  # Default data type for decimal numbers
DefaultEnergyUnit = 'keV'  # Default energy unit
DefaultMuUnit = 'mm-1'  # Default attenuation coefficient unit
BuiltInAttenEnergyUnit = 'MeV'  # Built-in attenuation files' energy unit

EnergyUnits = ['MeV', 'keV', 'eV']  # Supported energy units
ConcentrationUnits = ['g/mL', 'mg/mL']  # Supported concentration units

Or = Union
UintLike = Or[np.uint8, np.uint16, np.uint32, np.uint64]
SignedIntLike = Or[int, np.int8, np.int16, np.int32, np.int64]
IntLike = Or[UintLike, SignedIntLike]
try:
    FloatLike = Or[np.float16, np.float32, np.float64, np.float128, float]
except:
    FloatLike = Or[np.float16, np.float32, np.float64, float]

NDArray = np.ndarray
ListNDArray = List[NDArray]  # List of NDArray
TwoD = NDArray  # Two-dimensional array (not strictly)
ThreeD = Or[NDArray, ListNDArray]  # Three-dimensional array (not strictly)
TwoOrThreeD = Or[TwoD, ThreeD]  # Two or three-dimensional array (not strictly)
