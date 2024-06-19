import pytest
import numpy as np
from crip.plot import *


def test_smooth1D():
    data = np.array([1, 2, 3, 4, 5])
    res = smooth1D(data, 3)
    assert res.ndim == 1
    assert len(res) == 5


def test_smoothZ():
    s1 = np.ones((2, 2)) * 1.5
    s2 = np.ones((2, 2)) * 2
    s3 = np.ones((2, 2)) * 3.5
    img = np.array([s1, s2, s3])
    res = smoothZ(img, 3)
    assert res.shape == (3, 2, 2)
    assert res[1, 0, 0] == (1.5 + 2 + 3.5) / 3

def test_window():
    img = np.array([[0, 1], [2, 3]])
    res = window(img, (1, 2), 'wwwl', '01')
    assert res.ndim == 2
    assert res.max() == 1
    assert res.min() == 0