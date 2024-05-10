import unittest
import numpy as np


class test_permute(unittest.TestCase):

    def setUp(self):
        super().setUp()

        self.cube = np.array()

    def test(self):
        import crip.shared

        crip.shared.permute()

        # self.assertTrue(np.array_equal())
