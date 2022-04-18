import unittest
import unittest.mock as mock

from crip.physics import *


class test_getClassicDensity(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()

    def test(self):
        self.assertAlmostEqual(getClassicDensity('Water'), 1.0)
