import unittest

import numpy as np
import pandas as pd
from pyautoml import Preprocess


class TestPreprocessing(unittest.TestCase):

    def test_preprocessnumeric_normalize(self):

        unnormal_data = [[5.0, 3, 1],
                        [2.0, 2, 1],
                        [10.0, 1, 1]]

        columns = ["col1", "col2", "col3"]        
        data = pd.DataFrame(unnormal_data, columns=columns)

        preprocess = Preprocess(data=data, test_split_percentage=0.5, use_full_data=True)
        normal_data = preprocess.normalize_numeric()
        validate = normal_data.values.tolist()

        self.assertListEqual(validate, [[.375, 1.0, 0.0],
                                        [0, 0.5, 0.0],
                                        [1.0, 0, 0.0]])

    def test_preprocess_traindata(self):

        unnormal_train_data = [[5.0, 3, 1],
                            [2.0, 2, 1],
                            [10.0, 1, 1]]

        unnormal_test_data = [[5.0, 3, 1],
                            [2.0, 2, 1],
                            [10.0, 1, 1]]

        columns = ["col1", "col2", "col3"]        
        train_data = pd.DataFrame(unnormal_train_data, columns=columns)
        test_data = pd.DataFrame(unnormal_test_data, columns=columns)

        preprocess = Preprocess(train_data=train_data, test_data=test_data, test_split_percentage=0.5, use_full_data=False)
        train_normal_data, test_normal_data = preprocess.normalize_numeric()
        validate_train = train_normal_data.values.tolist()
        validate_test = test_normal_data.values.tolist()

        self.assertListEqual(validate_train, validate_test)

if __name__ == "__main__":
    unittest.main()
