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
        assert np.allclose(res, self.twoD)

