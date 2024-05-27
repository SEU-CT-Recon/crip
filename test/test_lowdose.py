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
        another = np.random.randn(*self.clean.shape) * self.sigma + self.mu

        assert stats.ttest_ind(noise.flatten(), another.flatten(), equal_var=True).pvalue > 0.1
