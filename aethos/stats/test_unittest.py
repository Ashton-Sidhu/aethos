import unittest

import numpy as np
import pandas as pd
import seaborn as sns

from aethos.core import Data


class Test_TestBase(unittest.TestCase):

    def test_compare_dist_predict(self):
    
        data = np.random.randint(0, 2, size=(1000, 3))
        data = pd.DataFrame(data=data, columns=["col1", "col2", "col3"])

        df = Data(data, target_field='col3')
        df.predict_data()

        self.assertTrue(True)

    def test_compare_dist_ks(self):
    
        data = np.random.randint(0, 2, size=(1000, 3))
        data = pd.DataFrame(data=data, columns=["col1", "col2", "col3"])

        df = Data(data, target_field='col3')
        df['col4'] = np.random.normal(1, 2, size=(1, 800))[0]
        df['col4'] = np.random.normal(10, 20, size=(1, 200))[0]
        df.ks_feature_distribution()

        self.assertTrue(True)
