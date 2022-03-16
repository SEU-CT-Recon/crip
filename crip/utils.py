'''
    Utilities of crip.

    by z0gSh1u @ https://github.com/z0gSh1u/crip
'''

import logging
import math
import numpy as np
from .typing import *

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
    if not ensure:
        logging.warning(hint, stack_info=dumpStack)


### Type check ###


def ConvertListNDArray(f):
    '''
        Decorator to convert List[np.ndarray] to np.ndarray.
    '''
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


def ensureFloatArray(arr):
    '''
        Make sure `arr` has floating type.
    '''
    if isType(arr, np.ndarray) and isIntType(arr):
        arr = arr.astype(DefaultFloatDType)
    return arr


def is2D(x: np.ndarray):
    return len(x.shape) == 2


def is3D(x: np.ndarray):
    return len(x.shape) == 3


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


def isList(x):
    return isType(x, list)


def isListNDArray(arr):
    return isType(arr, list) and isType(arr[0], np.ndarray)


def haveSameShape(a: np.ndarray, b: np.ndarray):
    return np.array_equal(a.shape, b.shape)


def inRange(a, range_=None, low=None, high=None):
    if range_:
        if isType(range_, range):
            range_ = (range_[0], range_[-1])
        low, high = range_

    return low <= a and a < high


import os


def getChildFolder(folder):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), f'./{folder}')


def inArray(a, arr):
    return a in arr


def cvtEnergyUnit(arr, from_, to):
    units = ['eV', 'keV', 'MeV']
    from_ = units.index(from_)
    to = units.index(to)

    a = 1000
    b = 1 / a
    mat = np.array([[1, b, b**2], [a, 1, b], [a**2, a, 1]])  # mat[from, to]

    return arr * mat[from_, to]


def cvtLengthUnit(arr, from_, to):
    units = ['um', 'mm', 'cm', 'm']
    from_ = units.index(from_)
    to = units.index(to)

    mat = np.array([[1, 1e-3, 1e-4, 1e-6], [1e3, 1, 1e-1, 1e-3], [1e4, 1e1, 1, 1e-2], [1e6, 1e3, 1e2,
                                                                                       1]])  # mat[from, to]

    return arr * mat[from_, to]


def cvtMuUnit(arr, from_, to):
    from_ = from_.replace('-1', '')
    to = to.replace('-1', '')

    return cvtLengthUnit(arr, to, from_)


def isNumber(a):
    return isType(a, int) or isType(a, float)


def readFileText(path_):
    with open(path_, 'r') as fp:
        content = fp.read()
    return content
