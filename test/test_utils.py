import pytest
import numpy as np
from crip.utils import *
import os


def test_readFileText():
    dirname = os.path.dirname(__file__)
    path = os.path.join(dirname, '_asset/spectrum.txt')
    content = readFileText(path)
    assert content.strip() == '''
keV omega
30 10000
31 15000
32 20000
33 -1'''.strip()


def test_CripException():
    assert issubclass(CripException, BaseException)


def test_cripAssert():
    with pytest.raises(CripException):
        cripAssert(False)


def test_cripWarning():
    cripWarning(False)


def test_ConvertListNDArray():

    @ConvertListNDArray
    def fn(x):
        return x

    fanIn = [np.array([1]), np.array([2])]
    fanOut = fn(fanIn)
    assert isinstance(fanOut, np.ndarray)
    assert fanOut.ndim == 2
    assert np.array_equal(fanOut, fanIn)


def test_asFloat():
    x = np.array([1], dtype=np.int32)
    assert asFloat(x).dtype == np.float32


def test_is1D():
    oneD = np.array([1, 2])
    twoD = np.array([[1, 2], [3, 4]])
    assert is1D(oneD) and not is1D(twoD)



def test_chw2hwc():
    pass


def test_nextPow2():
    assert list(map(nextPow2, [1, 2, 3, 4, 5, 6, 7, 8])) == [1, 2, 4, 4, 8, 8, 8, 8]


def test_identity():
    assert identity(1) == 1
