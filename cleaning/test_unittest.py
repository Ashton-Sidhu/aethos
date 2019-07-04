import unittest

import numpy as np
import pandas as pd

from clean import Clean
from numeric import *


class TestCleaning(unittest.TestCase):    
    
    def test_clean_removecolumns(self):

        int_missing_data = np.array([(1, 0, 0),
                                 (0, None, None),
                                 (None, None, None)])
        columns = ["col1", "col2", "col3"]        
        data = pd.DataFrame(int_missing_data, columns=columns)

        clean = Clean(data)
        clean.RemoveColumns(0.5)

        self.assertListEqual(clean.df.columns.tolist(), ["col1"])

    def test_clean_removerows(self):

        int_missing_data = np.array([(1, 0, 0),
                                 (0, None, None),
                                 (None, None, None)])
        columns = ["col1", "col2", "col3"]        
        data = pd.DataFrame(int_missing_data, columns=columns)

        clean = Clean(data)
        clean.RemoveRows(0.5)

        self.assertListEqual(clean.df.values.tolist(), np.array([(1, 0, 0)]).tolist())

    def test_cleannumeric_mean(self):
        int_missing_data = np.array([(1, 0, 2),
                                 (0, None, 1),
                                 (None, None, None)])

        columns = ["col1", "col2", "col3"]        
        data = pd.DataFrame(int_missing_data, columns=columns)

        clean_data, _ = ReplaceMissingMMM("mean", data=data)

        self.assertListEqual(clean_data.values.tolist(), np.array([(1, 0, 2),
                                                                (0, 0, 1),
                                                                (0.5, 0, 1.5)]).tolist())

    def test_cleannumeric_median(self):
        int_missing_data = np.array([(1, 0, 2),
                                 (0, None, 1),
                                 (None, None, None)])

        columns = ["col1", "col2", "col3"]        
        data = pd.DataFrame(int_missing_data, columns=columns)

        clean_data, _ = ReplaceMissingMMM("median", data=data)

        self.assertListEqual(clean_data.values.tolist(), np.array([(1, 0, 2),
                                                                (0, 0, 1),
                                                                (0.5, 0, 1.5)]).tolist())

    def test_cleannumeric_mostfrequent(self):
        int_missing_data = np.array([(1, 0, 2),
                                 (1, np.nan, 1),
                                 (np.nan, np.nan, 1)])

        columns = ["col1", "col2", "col3"]        
        data = pd.DataFrame(int_missing_data, columns=columns)

        clean_data, _ = ReplaceMissingMMM("most_frequent", data=data)

        self.assertListEqual(clean_data.values.tolist(), np.array([(1., 0., 2.),
                                                                (1., 0., 1.),
                                                                (1., 0., 1.)]).tolist())

    def test_cleannumeric_constant(self):
        int_missing_data = np.array([(1, 0, 2),
                                 (1, None, 1),
                                 (None, None, None)])

        columns = ["col1", "col2", "col3"]        
        data = pd.DataFrame(int_missing_data, columns=columns)

        clean_data, _ = ReplaceMissingConstant(10.5, ["col1", "col3"], data=data)

        self.assertListEqual(clean_data.values.tolist(), np.array([(0, 1, 2),
                                                                (None, 1, 1),
                                                                (None, 10.5, 10.5)]).tolist())

if __name__ == "__main__":
    unittest.main()
