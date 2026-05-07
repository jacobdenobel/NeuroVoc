import unittest

import numpy as np

from neurovoc.generate import process_neurogram

class TestZero(unittest.TestCase):
    def test_zero_columns(self):
        ng_data = np.array([[1, 0, 0, 1], [0, 1, 1, 0]])

        neurogram = process_neurogram(
            ng_data, [100, 100], [50, 150], None, 1e-1, False, 'test', 4e-1
        )
        self.assertTrue(neurogram.data.shape == ng_data.shape)
        