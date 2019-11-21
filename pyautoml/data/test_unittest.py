import unittest

import numpy as np
import pandas as pd

from pyautoml.data.data import Data
from pyautoml.util import *


class TestData(unittest.TestCase):
    
    def testutil_checkmissingdata(self):

        data = np.array(
            [
                (1, 1, 0, "hi my name is pyautoml", "green", 532.1),
                (2, 0, None, "this is my story", "yellow", 213.5),
                (3, None, None, "its me", "yellow", 154.2),
            ]
        )
        columns = ["pid", "col1", "col2", "col3", "col4", "col5"]

        dataset = pd.DataFrame(data, columns=columns)
        has_null = check_missing_data(dataset)

        self.assertTrue(has_null)

    def testutil_dropandreplacecolumns(self):
        data_zeros = np.zeros((2, 2))
        columns_zeros = ["col1", "col2"]
        data_ones = np.ones((2, 1))
        columns_ones = ["col3"]

        dataset_zeros = pd.DataFrame(data_zeros, columns=columns_zeros)
        dataset_ones = pd.DataFrame(data_ones, columns=columns_ones)
        df_new = drop_replace_columns(dataset_zeros, "col2", dataset_ones)

        self.assertListEqual(df_new.columns.tolist(), ["col1", "col3"])


if __name__ == "__main__":
    unittest.main()
