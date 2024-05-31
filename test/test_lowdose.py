import pytest
import numpy as np
from scipy import stats
from crip.lowdose import *
from crip.utils import isOfSameShape
import matplotlib.pyplot as plt
import skimage


def test_injectGaussianNoise():
    clean = np.random.randint(16, 196, (16, 16))
    sigma = 10
    mu = 0

    noisy = injectGaussianNoise(clean, sigma, mu)
    assert isOfSameShape(noisy, clean)

    noise = noisy - clean
    assert np.mean(noise) == pytest.approx(mu, abs=2)
    assert np.std(noise) == pytest.approx(sigma, abs=2)


def test_injectPoissonNoise():
    clean = np.ones((16, 16)) * 0.2
    noisy = injectPoissonNoise(clean, type_='postlog', nPhoton=1e5)
    assert isOfSameShape(noisy, clean)
    assert noisy.std() > clean.std()


def test_totalVariation():
    # 2D image
    tv = totalVariation(np.ones((16, 16)))
    assert tv == 0
    # 3D image
    tv = totalVariation(np.ones((2, 16, 16)))
    assert tv[0] == 0 and tv[1] == 0


class Test_nps2D:
    img = skimage.data.brain()[1]  # 256x256
    noisy = np.zeros((8, *img.shape))
    for i in range(8):
        noisy[i] = img + np.random.randn(*img.shape) * 100
    roi = noisy[:, 128:128 + 48, 70:70 + 48]

    def test_1(self):
        nps = nps2D(np.ones((2, 16, 16)), 1, detrend='individual')
        assert np.all(nps == 0)

    def test_2(self):
        nps1 = nps2D(self.roi, 1, detrend='individual', n=512, normalize='max')
        nps2 = nps2D(self.roi, 1, detrend='mutual', n=256, normalize='sum')
        nps3 = nps2D(self.roi, 1, detrend=None)

        assert nps1.ndim == 2 and nps2.ndim == 2 and nps3.ndim == 2
        assert nps1[256, 256] == pytest.approx(0, abs=0.01)


def test_nps2DRadAvg():
    img = skimage.data.brain()[1]  # 256x256
    noisy = np.zeros((8, *img.shape))
    for i in range(8):
        noisy[i] = img + np.random.randn(*img.shape) * 100
    roi = noisy[:, 128:128 + 48, 70:70 + 48]

    nps = nps2D(roi, 1, detrend='individual', n=512, normalize=None, fftshift=True)
    nps1d1 = nps2DRadAvg(nps, normalize='sum')
    nps1d2 = nps2DRadAvg(nps)
    assert nps1d1.ndim == 1 and nps1d2.ndim == 1
    assert nps1d1[0] == pytest.approx(0, abs=0.01)
