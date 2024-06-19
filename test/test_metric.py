import pytest
import numpy as np
from crip.metric import *


class Test_computeMAPE:

    def test_same_shape(self):
        x = np.array([[1, 2], [3, 4]])
        y = np.array([[1, 2], [3, 4]])

        assert abs(computeMAPE(x, y)) < 1e-6

    def test_diff_shape(self):
        x = np.array([[1], [3]])
        y = np.array([[1, 2], [3, 4]])

        with pytest.raises(CripException):
            computeMAPE(x, y)


class Test_computePSNR:

    def test_same_shape(self):
        x = np.array([[1, 2], [3, 4]])
        y = x + 1

        assert computePSNR(x / 255, y / 255, range_=1) > 35

    def test_diff_shape(self):
        x = np.array([[1], [3]])
        y = np.array([[1, 2], [3, 4]])

        with pytest.raises(CripException):
            computePSNR(x, y)


class Test_computeSSIM:

    def test_same_shape(self):
        x = np.random.randint(0, 200, size=(64, 64))
        y = x + 1

        assert computeSSIM(x / 255, y / 255, range_=1) > 0.95

    def test_diff_shape(self):
        x = np.array([[1], [3]])
        y = np.array([[1, 2], [3, 4]])

        with pytest.raises(CripException):
            computeSSIM(x, y)


def test_computeRMSE():
    x = np.array([[1, 2], [3, 4]])
    y = x + 1

    assert computeRMSE(x, y) == 1.0
    assert np.allclose(computeRMSE(x, y, pixelwise=True), np.ones((2, 2)))


def test_computeMAE():
    x = np.array([[1, 2], [3, 4]])
    y1 = x + 1
    y2 = x - 1

    assert computeMAE(x, y1) == 1.0
    assert computeMAE(x, y2) == 1.0
    assert np.allclose(computeMAE(x, y2, pixelwise=True) == 1.0, np.ones((2, 2)))


def test_pvalueInd():
    samples = np.random.randn(100)
    s1 = samples[:50]
    s2 = samples[50:]

    assert pvalueInd(s1, s2, True) > 0.05


def test_pvalueRel():
    samples = np.random.randn(100)
    s1 = samples[:50]
    s2 = samples[50:]

    assert pvalueInd(s1, s2, True) > 0.05


class Test_AverageMeter:
    meter = AverageMeter()

    def test_init(self):
        assert (self.meter.count, self.meter.sum, self.meter.avg, self.meter.val) == (0, 0, 0, 0)

    def test_update(self):
        self.meter.update(1)
        assert (self.meter.count, self.meter.sum, self.meter.avg, self.meter.val) == (1, 1, 1, 1)

        self.meter.update(2)
        assert (self.meter.count, self.meter.sum, self.meter.avg, self.meter.val) == (2, 3, 1.5, 2)
