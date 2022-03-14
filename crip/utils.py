'''
    Utilities of crip.

    by z0gSh1u @ https://github.com/z0gSh1u/crip
'''

import logging
import numpy as np
from .typing import *


class CripException(BaseException):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


def cripAssert(cond, hint):
    if not cond:
        raise CripException(hint)


def cripWarning(ensure, hint, dumpStack=False):
    if not ensure:
        logging.warning(hint, stack_info=dumpStack)


def is2D(x):
    return len(x.shape) == 2


def is3D(x):
    return len(x.shape) == 3


def is2or3D(x):
    return is2D(x) or is3D(x)


def ConvertProjList(f):
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
    if isType(arr, np.ndarray) and hasIntType(arr):
        arr = arr.astype(DefaultFloatDType)
    return arr


def isFloatDtype(dtype):
    return np.issubdtype(dtype, np.floating)


def hasIntType(arr: np.ndarray):
    return np.issubdtype(arr.dtype, np.integer)


def hasFloatType(arr: np.ndarray):
    return np.issubdtype(arr.dtype, np.floating)


def isType(x, t):
    return type(x) == t or isinstance(x, t)


def isProjList(arr):
    return isType(arr, list) and isType(arr[0], Proj)


def haveSameShape(a: np.ndarray, b: np.ndarray):
    return np.array_equal(a.shape, b.shape)
