import unittest
import numpy as np


class test_permute(unittest.TestCase):

    def setUp(self):
        super().setUp()

        self.cube = np.array([
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ])
        self.transverse = np.array([
            [[1, 5], [3, 7]],
            [[2, 6], [4, 8]],
        ])
        self.sagittal = np.array([
            [[1, 3], [2, 4]],
            [[5, 7], [6, 8]],
        ])
        self.coronal = np.array([
            [[1, 2], [5, 6]],
            [[3, 4], [7, 8]],
        ])

    def test(self):
        import crip.shared

        # crip.shared.permute()

        # self.assertTrue(np.array_equal())
