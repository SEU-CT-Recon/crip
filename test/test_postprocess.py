import pytest
import numpy as np
from crip.postprocess import *


def test_fovCropRadius():
    assert fovCropRadius(1000, 1000, 1000, 1) == 447


class test_fovCrop():
    twoD = np.ones((16, 16))
    threeD = np.ones((2, 16, 16))

    def test_twoD(self):
        cropped = fovCrop(self.twoD, radius=8)
        assert cropped[0, 0] == 0
        assert cropped[8, 8] == 1

    def test_threeD(self):
        cropped = fovCrop(self.threeD, radius=8, fill=10)
        assert np.all(cropped[:, 0, 0] == 10)
        assert np.all(cropped[:, 8, 8] == 1)


def test_muToHU():
    image = np.array([[1, 2], [3, 4]])
    assert np.allclose(muToHU(image, 1), [[0, 1000], [2000, 3000]])


def test_huToMu():
    image = np.array([[0, 1000], [2000, 3000]])
    assert np.allclose(huToMu(image, 1), [[1, 2], [3, 4]])


def test_huNoRescale():
    image = np.array([[-1000, 0], [1000, 2000]])
    assert np.allclose(huNoRescale(image), [[0, 1000], [2000, 3000]])


def test_postlogsToRaws():
    postlogs = np.array([[0, 1]])
    flat = 1000
    assert np.allclose(postlogsToRaws(postlogs, flat), [[flat, flat * np.exp(-1)]])
