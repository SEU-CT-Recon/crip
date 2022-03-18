import unittest
import unittest.mock as mock

from crip.io import *


class test_listDirectory(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()

    def test_sortParamVailidation(self):
        with mock.patch('os.listdir') as mockOsListdir:
            mockOsListdir.return_value = []
            hasRaised = False

            try:
                listDirectory('somewhere', sort='nat')
                listDirectory('somewhere', sort='dict')
            except:
                hasRaised = True
            self.assertFalse(hasRaised)

            try:
                listDirectory('somewhere', sort='blah')
            except:
                hasRaised = True
            self.assertTrue(hasRaised)

    def test_sortFilesCorrectly(self):
        with mock.patch('os.listdir') as mockOsListdir:
            mockOsListdir.return_value = ['10.png', '1.png', '2.png']

            list1 = listDirectory('somewhere', sort='nat', style='filename')
            self.assertTupleEqual(tuple(list1), ('1.png', '2.png', '10.png'))

            list2 = listDirectory('somewhere', sort='dict', style='filename')
            self.assertTupleEqual(tuple(list2), ('1.png', '10.png', '2.png'))
