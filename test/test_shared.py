import pytest
import numpy as np
from crip.shared import *


class Test_rotate():
    twoD = np.array([[1, 2], [3, 4]])
    threeD = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

    def test_twoD(self):
        img = self.twoD
        assert np.array_equal(rotate(img, 90), np.array([[3, 1], [4, 2]]))
        assert np.array_equal(rotate(img, 180), np.array([[4, 3], [2, 1]]))
        assert np.array_equal(rotate(img, 270), np.array([[2, 4], [1, 3]]))
        assert np.array_equal(rotate(img, 360), img)
        assert np.array_equal(rotate(img, 0), img)

        with pytest.raises(CripException):
            rotate(img, 45)

    def test_threeD(self):
        img = self.threeD
        assert np.array_equal(rotate(img, 90), np.array([[[3, 1], [4, 2]], [[7, 5], [8, 6]]]))
        assert np.array_equal(rotate(img, 180), np.array([[[4, 3], [2, 1]], [[8, 7], [6, 5]]]))
        assert np.array_equal(rotate(img, 270), np.array([[[2, 4], [1, 3]], [[6, 8], [5, 7]]]))
        assert np.array_equal(rotate(img, 360), img)
        assert np.array_equal(rotate(img, 0), img)

        with pytest.raises(CripException):
            rotate(img, 45)


class Test_verticalFlip():
    twoD = np.array([[1, 2], [3, 4]])
    threeD = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

    def test_twoD(self):
        img = self.twoD
        assert np.array_equal(verticalFlip(img), np.array([[3, 4], [1, 2]]))

    def test_threeD(self):
        img = self.threeD
        assert np.array_equal(verticalFlip(img), np.array([[[3, 4], [1, 2]], [[7, 8], [5, 6]]]))


class Test_horizontalFlip():
    twoD = np.array([[1, 2], [3, 4]])
    threeD = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

    def test_twoD(self):
        img = self.twoD
        assert np.array_equal(horizontalFlip(img), np.array([[2, 1], [4, 3]]))

    def test_threeD(self):
        img = self.threeD
        assert np.array_equal(horizontalFlip(img), np.array([[[2, 1], [4, 3]], [[6, 5], [8, 7]]]))


class Test_stackFlip():
    twoD = np.array([[1, 2], [3, 4]])
    threeD = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

    def test_twoD(self):
        img = self.twoD
        with pytest.raises(CripException):
            stackFlip(img)

    def test_threeD(self):
        img = self.threeD
        assert np.array_equal(stackFlip(img), np.array([[[5, 6], [7, 8]], [[1, 2], [3, 4]]]))


class Test_resizeTo():
    pass


class Test_resizeBy():
    pass


class Test_resize3D():
    pass


def test_permute():
    top = np.array([[1, 2], [3, 4]])
    bottom = np.array([[5, 6], [7, 8]])
    threeD = np.stack([top, bottom])


def test_fitPolyV2D2():
    pass


def test_applyPolyV2D2():
    pass
