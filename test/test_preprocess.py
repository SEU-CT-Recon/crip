import numpy as np
import pytest
from crip.preprocess import *
from crip.utils import CripException


class Test_averageProjections:
    twoD = np.array([
        [1, 2],
        [3, 4],
    ])
    threeD = np.array([twoD, twoD])

    def test_twoD(self):
        # a 2d array should raise an error
        with pytest.raises(CripException):
            averageProjections(self.twoD)

    def test_threeD(self):
        # a 3d array should return the average of the first axis
        res = averageProjections(self.threeD)
        assert res.ndim == 2
        assert np.allclose(res, self.twoD)


def test_correctFlatDarkField():
    projections = np.array([np.ones((3, 3)), np.ones((3, 3)) * 2])

    # provide flat
    flat = np.ones((3, 3)) * 3
    res1 = correctFlatDarkField(projections, flat)
    assert np.allclose(res1, -np.log(projections / flat))

    # no flat
    projections[0, 0, 0] = 10
    res2 = correctFlatDarkField(projections)
    assert res2[0, 0, 0] == pytest.approx(0)
    assert res2[0, 0, 1] == pytest.approx(-np.log(1 / 10))
    assert res2[1, 0, 0] == pytest.approx(0)

    # nan
    projections[0, 0, 0] = np.nan
    res3 = correctFlatDarkField(projections, fillNaN=-1)
    assert res3[0, 0, 0] == -1


def test_projectionsToSinograms():
    projections = np.ones((1, 2, 3))
    res = projectionsToSinograms(projections)
    assert res.shape == (2, 1, 3)


def test_sinogramsToProjections():
    sinograms = np.ones((2, 1, 3))
    res = sinogramsToProjections(sinograms)
    assert res.shape == (1, 2, 3)


def test_padImage():
    image = np.ones((3, 3))
    res1 = padImage(image, (2, 2, 2, 2), mode='constant', cval=10)
    assert res1.shape == (7, 7)
    assert res1[0, 0] == 10

    res2 = padImage(image, (2, 2, 2, 2), mode='reflect')
    assert res2[0, 0] == 1

    image3D = np.ones((2, 3, 3))
    res3 = padImage(image3D, (1, 1, 1, 1))
    assert res3.shape == (2, 5, 5)


def test_correctRingArtifactProjLi():
    pass


def test_fanToPara():
    pass


import numpy as np
import pytest
from crip.preprocess import *
from crip.utils import CripException


class Test_averageProjections:
    twoD = np.array([
        [1, 2],
        [3, 4],
    ])
    threeD = np.array([twoD, twoD])

    def test_twoD(self):
        # a 2d array should raise an error
        with pytest.raises(CripException):
            averageProjections(self.twoD)

    def test_threeD(self):
        # a 3d array should return the average of the first axis
        res = averageProjections(self.threeD)
        assert res.ndim == 2
        assert np.allclose(res, self.twoD)


def test_correctFlatDarkField():
    projections = np.array([np.ones((3, 3)), np.ones((3, 3)) * 2])

    # provide flat
    flat = np.ones((3, 3)) * 3
    res1 = correctFlatDarkField(projections, flat)
    assert np.allclose(res1, -np.log(projections / flat))

    # no flat
    projections[0, 0, 0] = 10
    res2 = correctFlatDarkField(projections)
    assert res2[0, 0, 0] == pytest.approx(0)
    assert res2[0, 0, 1] == pytest.approx(-np.log(1 / 10))
    assert res2[1, 0, 0] == pytest.approx(0)

    # nan
    projections[0, 0, 0] = np.nan
    res3 = correctFlatDarkField(projections, fillNaN=-1)
    assert res3[0, 0, 0] == -1


def test_projectionsToSinograms():
    projections = np.ones((1, 2, 3))
    res = projectionsToSinograms(projections)
    assert res.shape == (2, 1, 3)


def test_sinogramsToProjections():
    sinograms = np.ones((2, 1, 3))
    res = sinogramsToProjections(sinograms)
    assert res.shape == (1, 2, 3)


def test_padImage():
    image = np.ones((3, 3))
    res1 = padImage(image, (2, 2, 2, 2), mode='constant', cval=10)
    assert res1.shape == (7, 7)
    assert res1[0, 0] == 10

    res2 = padImage(image, (2, 2, 2, 2), mode='reflect')
    assert res2[0, 0] == 1

    image3D = np.ones((2, 3, 3))
    res3 = padImage(image3D, (1, 1, 1, 1))
    assert res3.shape == (2, 5, 5)


def test_correctRingArtifactProjLi():
    pass


def test_fanToPara():
    sgm = np.ones((4, 3))
    gammas = np.array([0.1, 0.2, 0.3])
    betas = np.array([0.4, 0.5, 0.6, 0.7])
    sid = 100.0
    oThetas = (0.0, 0.1, 0.2)
    oLines = (-50.0, 10.0, 50.0)

    res = fanToPara(sgm, gammas, betas, sid, oThetas, oLines)
    assert res.shape == (2, 10)
    assert set(list(res.flatten())) == set([0, 1])
