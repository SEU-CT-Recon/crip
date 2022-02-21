# Test if crip can be imported without error.

import unittest


class canImport(unittest.TestCase):
    def test(self):
        import sys
        sys.path.append('../')
        somethingRaised = False
        try:
            import crip
        except:
            somethingRaised = True
        self.assertFalse(somethingRaised)


if __name__ == '__main__':
    unittest.main()
