import pytest
import numpy as np
from scipy import stats
from crip.lowdose import *
from crip.utils import isOfSameShape


class Test_injectGaussianNoise:
    clean = np.random.randint(16, 196, (16, 16))
    sigma = 10
    mu = 0

    def test_1(self):
        noisy = injectGaussianNoise(self.clean, self.sigma, self.mu)
        assert isOfSameShape(noisy, self.clean)

        noise = noisy - self.clean
        assert np.mean(noise) == pytest.approx(self.mu, abs=1)
        assert np.std(noise) == pytest.approx(self.sigma, abs=1)
