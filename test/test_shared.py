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


def test_resizeTo():
    img = np.random.randint(0, 255, (100, 100)).astype(np.float32)
    assert resizeTo(img, (50, 50)).shape == (50, 50)

    with pytest.raises(CripException):
        resizeTo(img, (50, 50, 50))


def test_resizeBy():
    img = np.random.randint(0, 255, (100, 100)).astype(np.float32)
    assert resizeBy(img, 2).shape == (200, 200)
    assert resizeBy(img, (0.5, 0.5)).shape == (50, 50)


def test_resize3D():
    img = np.random.randint(0, 255, (10, 100, 100)).astype(np.float32)
    assert resize3D(img, (0.5, 0.5, 0.5)).shape == (5, 50, 50)


def test_gaussianSmooth():
    img = np.random.randint(0, 255, (100, 100)).astype(np.float32)
    assert gaussianSmooth(img, 1).shape == (100, 100)
    assert gaussianSmooth(img, (1, 1)).shape == (100, 100)
    assert gaussianSmooth(img, (1, 1), 5).shape == (100, 100)
    assert gaussianSmooth(img, (1, 1), (5, 5)).shape == (100, 100)


def test_stackImages():
    img1 = np.ones((100, 100))
    img2 = np.zeros((100, 100))
    assert stackImages([img1, img2]).shape == (2, 100, 100)


def test_splitImages():
    img = np.random.randint(0, 255, (2, 100, 100))
    list_ = splitImages(img)
    assert isinstance(list_, list)
    assert len(splitImages(img)) == 2


def test_transpose():
    img = np.random.randint(0, 255, (10, 20, 30))
    assert transpose(img, (1, 0, 2)).shape == (20, 10, 30)
    assert transpose(img, (2, 1, 0)).shape == (30, 20, 10)


def test_permute():
    pass


def test_shepplogan():
    for size in [256, 512, 1024]:
        s = shepplogan(size)
        assert s.shape == (size, size)

    with pytest.raises(CripException):
        shepplogan(128)


def test_fitPolyV2D2():
    x1 = np.random.randint(-50, 50, 100).astype(np.float32)
    x2 = np.random.randint(-50, 50, 100).astype(np.float32)
    y = 1 * x1**2 + 1.5 * x2**2 + 1.8 * x1 * x2 + 1 * x1 + 0.5 * x2 + 0.2
    coeff = fitPolyV2D2(x1, x2, y, bias=True)
    assert np.allclose(coeff, np.array([1, 1.5, 1.8, 1, 0.5, 0.2]), atol=1e-3)


def test_applyPolyV2D2():
    x1 = np.random.randint(-50, 50, 100).astype(np.float32)
    x2 = np.random.randint(-50, 50, 100).astype(np.float32)
    y = 1 * x1**2 + 1.5 * x2**2 + 1.8 * x1 * x2 + 1 * x1 + 0.5 * x2 + 0.2
    coeffs = np.array([1, 1.5, 1.8, 1, 0.5])
    assert np.allclose(applyPolyV2D2(coeffs, x1, x2), y - 0.2, atol=1e-3)


def test_fitPolyV1D2():
    x1 = np.random.randint(-50, 50, 100).astype(np.float32)
    y = 1 * x1**2 + 1 * x1 + 0.5
    coeff = fitPolyV1D2(x1, y, bias=True)
    assert np.allclose(coeff, np.array([1, 1, 0.5]), atol=1e-3)


def test_applyPolyV1D2():
    x1 = np.random.randint(-50, 50, 100).astype(np.float32)
    y = 1 * x1**2 + 1 * x1 + 0.5
    coeffs = np.array([1, 1])
    assert np.allclose(applyPolyV1D2(coeffs, x1), y - 0.5, atol=1e-3)
