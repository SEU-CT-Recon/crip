'''
    Utilities of crip.

    https://github.com/SEU-CT-Recon/crip
'''

__all__ = [
    'readFileText', 'CripException', 'cripAssert', 'cripWarning', 'ConvertListNDArray', 'asFloat', 'is2D', 'is3D',
    'is2or3D', 'isInt', 'isIntDtype', 'isFloatDtype', 'isIntType', 'isFloatType', 'isType', 'isNumber', 'isList',
    'isListNDArray', 'isOfSameShape', 'inRange', 'inArray', 'getAsset', 'cvtEnergyUnit', 'cvtLengthUnit', 'cvtMuUnit',
    'radToDeg', 'degToRad', 'sysPlatform', 'getHW', 'is1D', 'as3D', 'nextPow2'
]

import os
import logging
import math
import numpy as np
import sys
import functools
from ._typing import *
from ._rc import *


def readFileText(path_):
    with open(path_, 'r') as fp:
        content = fp.read()
    return content


### Expection ###


class CripException(BaseException):
    '''
        The universal expection class for crip.
    '''

    def __init__(self, *args: object) -> None:
        super().__init__(*args)


def cripAssert(cond, hint):
    '''
        The only assert method for crip.
    '''
    if not cond:
        raise CripException(hint)


def cripWarning(ensure, hint, dumpStack=False):
    '''
        The only warning method for crip.
    '''
    if not SUPPRESS_WARNING and not ensure:
        logging.warning(hint, stack_info=dumpStack)


### Type check ###


def ConvertListNDArray(f):
    '''
        Decorator to convert List[ndarray] to ndarray.
    '''

    @functools.wraps(f)
    def fn(*args, **kwargs):
        # args and kwargs are immutable
        argsn = []
        for a in args:
            if isListNDArray(a):
                a = np.array(a)
            argsn.append(a)
        kwargsn = {}
        for k in kwargs:
            if isListNDArray(kwargs[k]):
                kwargs[k] = np.array(kwargs[k])
            kwargsn[k] = kwargs[k]
        return f(*argsn, **kwargsn)

    return fn


def asFloat(arr):
    '''
        Make sure `arr` has floating type.
    '''
    if isType(arr, np.ndarray) and isIntType(arr):
        arr = arr.astype(DefaultFloatDType)
    return arr


def as3D(x: np.ndarray):
    cripAssert(is2or3D(x))
    return x if len(x.shape) == 3 else x[np.newaxis, ...]


def is1D(x: np.ndarray):
    return isType(x, NDArray) and len(x.shape) == 1


def is2D(x: np.ndarray):
    return isType(x, NDArray) and len(x.shape) == 2


def is3D(x: np.ndarray):
    return isType(x, NDArray) and len(x.shape) == 3


def is2or3D(x: np.ndarray):
    return is2D(x) or is3D(x)


def isInt(n):
    return math.floor(n) == n


def isIntDtype(dtype):
    return np.issubdtype(dtype, np.integer)


def isFloatDtype(dtype):
    return np.issubdtype(dtype, np.floating)


def isIntType(arr: np.ndarray):
    return isIntDtype(arr.dtype)


def isFloatType(arr: np.ndarray):
    return isFloatDtype(arr.dtype)


def isType(x, t):
    '''
        Check if `x` has type `t` or isinstance of `t`.
    '''
    if t is Callable:
        return callable(x)
    return type(x) == t or isinstance(x, t)


def isNumber(a):
    return isType(a, int) or isType(a, float)


def isList(x):
    return isType(x, list)


def isListNDArray(arr):
    return isType(arr, list) and isType(arr[0], np.ndarray)


def isOfSameShape(a: np.ndarray, b: np.ndarray):
    return np.array_equal(a.shape, b.shape)


def inRange(a, range_=None, low=None, high=None):
    if range_:
        if isType(range_, range):
            range_ = (range_[0], range_[-1])
        low, high = range_

    return low <= a and a < high


def inArray(a, arr):
    return a in arr


def getAsset(folder, prefix='_asset'):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), f'{prefix}/{folder}')


def cvtEnergyUnit(arr, from_, to):
    '''
        Convert between energy units. (ev, keV, MeV)
    '''
    units = ['eV', 'keV', 'MeV']
    from_ = units.index(from_)
    to = units.index(to)

    a = 1000
    b = 1 / a
    mat = np.array([
        [1, b, b**2],
        [a, 1, b],
        [a**2, a, 1],
    ])  # mat[from, to]

    return arr * mat[from_, to]


def cvtLengthUnit(arr, from_, to):
    '''
        Convert between length units. (um, mm, cm, m)
    '''
    units = ['um', 'mm', 'cm', 'm']
    from_ = units.index(from_)
    to = units.index(to)

    mat = np.array([
        [1, 1e-3, 1e-4, 1e-6],
        [1e3, 1, 1e-1, 1e-3],
        [1e4, 1e1, 1, 1e-2],
        [1e6, 1e3, 1e2, 1],
    ])  # mat[from, to]

    return arr * mat[from_, to]


def cvtMuUnit(arr, from_, to):
    '''
        Convert between mu value units. (um-1, mm-1, cm-1, m-1)
    '''
    from_ = from_.replace('-1', '')
    to = to.replace('-1', '')

    return cvtLengthUnit(arr, to, from_)


def cvtConcentrationUnit(arr, from_, to):
    '''
        Convert between concentration units. (g/mL, mg/mL)
    '''
    units = ['g/mL', 'mg/mL']
    from_ = units.index(from_)
    to = units.index(to)

    # g/mL, mg/mL
    mat = np.array([
        [1, 1000],  # from g/mL
        [1 / 1000, 1]  # from mg/mL
    ])

    return arr * mat[from_, to]


def radToDeg(x):
    return x / np.pi * 180


def degToRad(x):
    return x / 180 * np.pi


def sysPlatform():
    platform = sys.platform

    if platform.find('win32') != -1:
        return 'Windows'
    elif platform.find('linux') != -1:
        return 'Linux'

    cripAssert(False, f'Unsupported platform: {platform}.')


def getHW(img: np.ndarray):
    '''
        Get height and width of `img`.
    '''
    if is3D(img):
        _, h, w = img.shape
    elif is2D(img):
        h, w = img.shape
    else:
        cripAssert(False, 'img should be 2D or 3D.')

    return h, w


def nextPow2(x):
    '''
        Get the next power of 2 of `x`.
    '''
    return 1 if x == 0 else 2**math.ceil(math.log2(x))
