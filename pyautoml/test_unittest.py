import unittest

import pandas as pd
from pyautoml.base import MethodBase


class Test_TestBase(unittest.TestCase):


    def test_setitem_constant(self):

        int_missing_data = [[1, 0, 0],
                            [0, 2, 3],
                            [0, 3, 4],
                            [1, 2, 3]]
        columns = ["col1", "col2", "col3"]        
        data = pd.DataFrame(int_missing_data, columns=columns)

        base = MethodBase(data=data, train_data=None, test_data=None, use_full_data=False, target_field='', report_name=None, test_split_percentange=0.25)

        base['col4'] = 5

        validate = any(base.train_data['col4'].isnull()) and any(base.test_data['col4'].isnull())

        self.assertFalse(validate)


    def test_setitem_equalsize_list(self):

        int_missing_data = [[1, 0, 0],
                            [0, 2, 3],
                            [0, 3, 4],
                            [1, 2, 3]]
        columns = ["col1", "col2", "col3"]        
        data = pd.DataFrame(int_missing_data, columns=columns)

        base = MethodBase(data=data, train_data=None, test_data=None, use_full_data=False, target_field='', report_name=None, test_split_percentange=0.5)

        base['col4'] = [5, 5]

        validate = any(base.train_data['col4'].isnull()) and any(base.test_data['col4'].isnull())

        self.assertFalse(validate)

    def test_setitem_traindata(self):

        int_missing_data = [[1, 0, 0],
                            [0, 2, 3],
                            [0, 3, 4],
                            [1, 2, 3]]
        columns = ["col1", "col2", "col3"]        
        data = pd.DataFrame(int_missing_data, columns=columns)

        base = MethodBase(data=data, train_data=None, test_data=None, use_full_data=False, target_field='', report_name=None, test_split_percentange=0.25)

        base['col4'] = [5, 5, 5]

        validate = any(base.train_data['col4'].isnull())

        self.assertFalse(validate)

    def test_setitem_testdata(self):

        int_missing_data = [[1, 0, 0],
                            [0, 2, 3],
                            [0, 3, 4],
                            [1, 2, 3]]
        columns = ["col1", "col2", "col3"]        
        data = pd.DataFrame(int_missing_data, columns=columns)

        base = MethodBase(data=data, train_data=None, test_data=None, use_full_data=False, target_field='', report_name=None, test_split_percentange=0.75)

        base['col4'] = [5, 5, 5]

        validate = any(base.test_data['col4'].isnull())

        self.assertFalse(validate)

    def test_dropcolumns(self):

        int_missing_data = [[1, 0, 0],
                            [0, 2, 3],
                            [0, 3, 4],
                            [1, 2, 3]]
        columns = ["col1", "col2", "col3"]        
        data = pd.DataFrame(int_missing_data, columns=columns)

        base = MethodBase(data=data, train_data=None, test_data=None, use_full_data=False, target_field='', report_name="test", test_split_percentange=0.5)
        base.drop("col1", "col3", reason="Columns were unimportant.")

        validate = (base.train_data.columns == ['col2'] and base.test_data.columns == ['col2'])

        self.assertTrue(validate)

if __name__ == "__main__":
    unittest.main()
