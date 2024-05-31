'''
    Utilities of crip.

    https://github.com/SEU-CT-Recon/crip
'''

import os
import logging
import math
import numpy as np
import functools

from ._typing import *


def readFileText(path_: str, encoding=None) -> str:
    ''' Read text file.
    '''
    with open(path_, 'r', encoding=encoding) as fp:
        content = fp.read()

    return content


### Expection ###


class CripException(BaseException):
    ''' The universal expection class for crip.
    '''

    def __init__(self, *args) -> None:
        super().__init__(*args)


def cripAssert(cond: Any, hint=''):
    ''' The only assert method for crip.
    '''
    if not cond:
        raise CripException(hint)


def cripWarning(ensure: Any, hint='', stackTrace=False):
    ''' The only warning method for crip.
    '''
    if not ensure:
        logging.warning(hint, stack_info=stackTrace)


### Type check ###


def ConvertListNDArray(f):
    ''' Function decorator to convert List[ndarray] to ndarray.
    '''

    @functools.wraps(f)
    def fn(*args, **kwargs):
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


def asFloat(arr: NDArray) -> NDArray:
    ''' Ensure `arr` has `DefaultFloatDType` dtype.
    '''
    cripAssert(isType(arr, NDArray), '`arr` should be NDArray.')

    return arr.astype(DefaultFloatDType)


def is1D(x: NDArray) -> bool:
    ''' Check if `x` is 1D ndarray.
    '''
    return isType(x, NDArray) and len(x.shape) == 1


def is2D(x: NDArray) -> bool:
    ''' Check if `x` is 2D ndarray.
    '''
    return isType(x, NDArray) and len(x.shape) == 2


def is3D(x: NDArray) -> bool:
    ''' Check if `x` is 3D ndarray.
    '''
    return isType(x, NDArray) and len(x.shape) == 3


def is2or3D(x: NDArray) -> bool:
    ''' Check if `x` is 2D or 3D ndarray.
    '''
    return is2D(x) or is3D(x)


def as3D(x: NDArray) -> NDArray:
    ''' Ensure `x` to be 3D ndarray.
    '''
    cripAssert(is2or3D(x))

    return x if is3D(x) else x[np.newaxis, ...]


def isInt(n) -> bool:
    ''' Check if `n` is int.
    '''
    return math.floor(n) == n


def isIntDtype(dtype) -> bool:
    ''' Check if `dtype` is integer type.
    '''
    return np.issubdtype(dtype, np.integer)


def isFloatDtype(dtype) -> bool:
    ''' Check if `dtype` is float type.
    '''
    return np.issubdtype(dtype, np.floating)


def hasIntDtype(arr: NDArray) -> bool:
    ''' Check if `arr` has integer dtype.
    '''
    return isIntDtype(arr.dtype)


def isType(x, t) -> bool:
    ''' Check if `x` has type `t` or isinstance of `t`.
    '''
    if t is Callable:
        return callable(x)
    return type(x) == t or isinstance(x, t)


def isListNDArray(x) -> bool:
    ''' Check if `x` is List[NDArray].
    '''
    return isType(x, list) and len(x) > 0 and isType(x[0], NDArray)


def isOfSameShape(a: NDArray, b: NDArray) -> bool:
    ''' Check if two NDArray have the same shape.
    '''
    return np.array_equal(a.shape, b.shape)


def getAsset(folder: str, prefix='_asset') -> str:
    ''' Get asset path under `crip/<prefix>/<folder>`.
    '''
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), f'{prefix}/{folder}')


def convertEnergyUnit(arr: Or[NDArray, float], from_: str, to: str) -> Or[NDArray, float]:
    ''' Convert between energy units. [ev, keV, MeV]
    '''
    units = ['eV', 'keV', 'MeV']
    cripAssert(from_ in units and to in units, f'Invalid unit: from_={from_}, to={to}.')

    from_ = units.index(from_)
    to = units.index(to)
    a = 1000
    b = 1 / a
    M = np.array([
        [1, b, b**2],
        [a, 1, b],
        [a**2, a, 1],
    ])  # M[from_, to]

    return arr * M[from_, to]


def convertLengthUnit(arr: Or[NDArray, float], from_: str, to: str) -> Or[NDArray, float]:
    ''' Convert between length units. [um, mm, cm, m]
    '''
    units = ['um', 'mm', 'cm', 'm']
    cripAssert(from_ in units and to in units, f'Invalid unit: from_={from_}, to={to}.')

    from_ = units.index(from_)
    to = units.index(to)
    M = np.array([
        [1, 1e-3, 1e-4, 1e-6],
        [1e3, 1, 1e-1, 1e-3],
        [1e4, 1e1, 1, 1e-2],
        [1e6, 1e3, 1e2, 1],
    ])  # M[from_, to]

    return arr * M[from_, to]


def convertMuUnit(arr: Or[NDArray, float], from_: str, to: str) -> Or[NDArray, float]:
    ''' Convert between mu value units. [um-1, mm-1, cm-1, m-1]
    '''
    units = ['um-1', 'mm-1', 'cm-1', 'm-1']
    cripAssert(from_ in units and to in units, f'Invalid unit: from_={from_}, to={to}.')

    return convertLengthUnit(arr, to.replace('-1', ''), from_.replace('-1', ''))


def convertConcentrationUnit(arr: Or[NDArray, float], from_: str, to: str) -> Or[NDArray, float]:
    ''' Convert between concentration units. [g/mL, mg/mL]
    '''
    units = ['g/mL', 'mg/mL']
    cripAssert(from_ in units and to in units, f'Invalid unit: from_={from_}, to={to}.')

    from_ = units.index(from_)
    to = units.index(to)
    # to g/mL, mg/mL
    M = np.array([
        [1, 1000],  # from g/mL
        [1 / 1000, 1]  # from mg/mL
    ])

    return arr * M[from_, to]


def getHnW(img: NDArray) -> Tuple[int, int]:
    ''' Get height and width of `img` with shape [CHW] or [HW].
    '''
    cripAssert(is2or3D(img), 'img should be 2D or 3D.')

    return img.shape[-2], img.shape[-1]


def nextPow2(x: int) -> int:
    ''' Get the next power of 2 of integer `x`.
    '''
    return 1 if x == 0 else 2**math.ceil(math.log2(x))


def getAttrKeysOfObject(obj: object) -> List[str]:
    ''' Get all attribute keys of `obj` excluding methods, private and default attributes.
    '''
    keys = [
        a for a in (set(dir(obj)) - set(dir(object)))
        if not a.startswith('__') and not callable(getattr(obj, a)) and getattr(obj, a) is not None
    ]

    return keys


def chw2hwc(img: ThreeD) -> ThreeD:
    ''' Convert CHW to HWC.
    '''
    cripAssert(is3D(img), 'img should be 3D.')

    return np.moveaxis(img, 0, -1)


def hwc2chw(img: ThreeD) -> ThreeD:
    ''' Convert HWC to CHW.
    '''
    cripAssert(is3D(img), 'img should be 3D.')

    return np.moveaxis(img, -1, 0)


def simpleValidate(conds: List[bool]):
    ''' Validate conditions.
    '''
    for i in range(len(conds)):
        cripAssert(conds[i], f'Condition {i} validation failed.')


def identity(x: Any) -> Any:
    ''' Identity function.
    '''
    return x
