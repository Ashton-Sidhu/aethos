import unittest

import numpy as np
import pandas as pd

from pyautoml.preprocessing.preprocess import Preprocess


class TestPreprocessing(unittest.TestCase):

    def test_preprocessnumeric_normalize(self):

        unnormal_data = [[5.0, 3, 1],
                        [2.0, 2, 1],
                        [10.0, 1, 1]]

        columns = ["col1", "col2", "col3"]        
        data = pd.DataFrame(unnormal_data, columns=columns)

        preprocess = Preprocess(data, test_split_percentage=0.5, use_full_data=True)
        normal_data = preprocess.NormalizeNumeric()
        validate = normal_data.values.tolist()

        self.assertListEqual(validate, [[.375, 1.0, 0.0],
                                        [0, 0.5, 0.0],
                                        [1.0, 0, 0.0]])

if __name__ == "__main__":
    unittest.main()
