import unittest


class test_itJustImports(unittest.TestCase):

    def setUp(self):
        return super().setUp()

    def test(self):
        self.assertEqual(0, 0)

        import crip
        import crip._rc
        import crip.de
        import crip.io
        import crip.lowdose
        import crip.physics
        import crip.postprocess
        import crip.preprocess
        import crip.shared
        import crip.utils
        import crip.mangoct
        import crip.plot
        import crip.metric

        self.assertEqual(1, 1)
