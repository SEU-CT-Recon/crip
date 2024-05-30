import os
import pytest
import numpy as np
from crip.physics import *
from crip.utils import CripException


def test_getCommonDensity():
    assert getCommonDensity('H2O') == 1.0
    assert getCommonDensity('Water') == 1.0

    with pytest.raises(CripException):
        getCommonDensity('h2o')


class Test_Spectrum:

    def test_init(self):
        s = Spectrum(omega=[0, 10, 20, 0, *[0] * (151 - 4)])
        assert s.omega[1] == 10
        assert s.omega[2] == 20
        assert s.omega[3] == 0
        assert s.omega[4:].sum() == 0
        assert s.sumOmega == 30

    def test_isMonochromatic(self):
        omega = [0.0] * 151
        omega[50] = 1.0
        is_mono, energy = Spectrum(omega).isMonochromatic()
        assert is_mono == True
        assert energy == 50

    def test_fromText(self):
        spec = Spectrum.fromText('50 1\n51 2\n52 3\n53 4\n54 5\n55 -1\n')
        assert list(spec.omega[50:56]) == [1, 2, 3, 4, 5, 0]

    def test_fromFile(self):
        spec = Spectrum.fromFile(os.path.join(os.path.dirname(__file__), '_asset/spectrum.txt'))
        assert list(spec.omega[30:34]) == [10000, 15000, 20000, 0]

    def test_monochromatic(self):
        spec = Spectrum.monochromatic(100)
        assert [spec.omega[100], spec.omega[101]] == [10**5, 0]
