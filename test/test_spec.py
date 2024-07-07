import pytest
import numpy as np

from crip.spec import *


def test_compose2():
    v1 = np.array([1, 0])
    v2 = np.array([0, 1])
    assert np.allclose(compose2(1, 1, v1, v2), np.array([1, 1]))


def test_compose3():
    v1 = np.array([1, 0, 0])
    v2 = np.array([0, 1, 0])
    v3 = np.array([0, 0, 1])
    assert np.allclose(compose3(1, 1, 1, v1, v2, v3), np.array([1, 1, 1]))

