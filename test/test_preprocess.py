import unittest
import unittest.mock as mock

from crip.preprocess import *


class test_averageProjections(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()

    def test(self):
        testProjs = [np.array([[1, 2, 3]]), np.array([[4, 5, 6]])]
        ans = np.array([[2.5, 3.5, 4.5]])

        avg1 = averageProjections(testProjs)
        avg2 = averageProjections(np.array(testProjs))
        self.assertTrue(np.array_equal(ans, avg1))
        self.assertTrue(np.array_equal(ans, avg2))

        # The output should be 2D.
        self.assertTrue(len(avg1.shape) == 2)


class test_flatDarkFieldCorrection(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()

    def test(self):
        proj = np.array([[10, 20, 30]])
        flat = np.array([[10, 10, 10]])
        postlog = np.array([[0, 0.693147, 1.098612]])
