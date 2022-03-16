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


def ConvertProjList(f):
    '''
        Decorator to convert ProjList to ProjStack.
    '''
    def fn(*args, **kwargs):
        # args and kwargs are immutable
        argsn = []
        for a in args:
            if isProjList(a):
                a = np.array(a)
            argsn.append(a)
        kwargsn = {}
        for k in kwargs:
            if isProjList(kwargs[k]):
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
    return type(x) == t or isinstance(x, t)


def isList(x):
    return isType(x, list)


def isProjList(arr):
    return isType(arr, list) and isType(arr[0], Proj)


def haveSameShape(a: np.ndarray, b: np.ndarray):
    return np.array_equal(a.shape, b.shape)


def inRange(a, range_=None, low=None, high=None):
    if range_:
        low, high = range_

    return low <= a and a < high


import os

def getChildFolder(folder):
    return os.path.join(os.path.abspath(''), f'./{folder}')