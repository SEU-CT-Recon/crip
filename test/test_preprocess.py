import unittest
from unittest import TestCase

from crip.preprocess import *
from crip.utils import isOfSameShape


class test_averageProjections(TestCase):
    def test(self):
        testProjs = [np.array([[1, 2, 3]]), np.array([[4, 5, 6]])]
        ans = np.array([[2.5, 3.5, 4.5]])

        # Should support List[Proj] and ProjStack.
        avg1 = averageProjections(testProjs)
        avg2 = averageProjections(np.array(testProjs))
        self.assertTrue(np.array_equal(ans, avg1))
        self.assertTrue(np.array_equal(ans, avg2))

        # The output should be 2D.
        self.assertTrue(len(avg1.shape) == 2)


class test_flatDarkFieldCorrection(TestCase):
    def test(self):
        proj = np.array([[10, 20, 30]])
        flat = np.array([[10, 10, 10]])

        postlogGT = np.array([[0, 0.693147, 1.098612]])
        postlog = flatDarkFieldCorrection(proj, flat)

        self.assertTrue(np.allclose(postlogGT, postlog))


class test_injectGaussianNoise(TestCase):
    def test(self):
        testProjsList = [np.array([[1, 2, 3]]), np.array([[4, 5, 6]])]
        testProjsStack = np.array(testProjsList)

        noisy1 = injectGaussianNoise(testProjsList)
        noisy2 = injectGaussianNoise(testProjsStack)

        self.assertTrue(isOfSameShape(noisy1, testProjsStack))
        self.assertTrue(isOfSameShape(noisy2, testProjsStack))


class test_injectPoissonNoise(TestCase):
    def test(self):
        testProjsList = [np.array([[1, 2, 3]]), np.array([[4, 5, 6]])]
        testProjsStack = np.array(testProjsList)

        noisy1 = injectPoissonNoise(testProjsList)
        noisy2 = injectPoissonNoise(testProjsStack)

        self.assertTrue(isOfSameShape(noisy1, testProjsStack))
        self.assertTrue(isOfSameShape(noisy2, testProjsStack))


class test_projectionsToSinograms(TestCase):
    def test(self):
        projs = np.zeros((10, 20, 30))  # (views, h, w)
        sinos = projectionsToSinograms(projs)
        expectedShape = np.array([20, 10, 30])

        self.assertTrue(np.array_equal(sinos.shape, expectedShape))


class test_sinogramsToProjections(TestCase):
    def test(self):
        projs = np.zeros((20, 10, 30))  # (h, views, w)
        sinos = sinogramsToProjections(projs)
        expectedShape = np.array([10, 20, 30])

        self.assertTrue(np.array_equal(sinos.shape, expectedShape))

