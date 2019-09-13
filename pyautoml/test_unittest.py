import unittest

import numpy as np
import pandas as pd
from pyautoml import Clean
from pyautoml.base import MethodBase


class Test_TestBase(unittest.TestCase):


    def test_setitem_constant(self):

        int_missing_data = [[1, 0, 0],
                            [0, 2, 3],
                            [0, 3, 4],
                            [1, 2, 3]]
        columns = ["col1", "col2", "col3"]        
        data = pd.DataFrame(int_missing_data, columns=columns)

        base = MethodBase(data=data, train_data=None, test_data=None, split=True, target_field='', report_name=None, test_split_percentage=0.25)

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

        base = MethodBase(data=data, train_data=None, test_data=None, split=True, target_field='', report_name=None, test_split_percentage=0.5)

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

        base = MethodBase(data=data, train_data=None, test_data=None, split=True, target_field='', report_name=None, test_split_percentage=0.25)

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

        base = MethodBase(data=data, train_data=None, test_data=None, split=True, target_field='', report_name=None, test_split_percentage=0.75)

        base['col4'] = [5, 5, 5]

        validate = any(base.test_data['col4'].isnull())

        self.assertFalse(validate)

    def test_setitem_tupleeven(self):
        int_missing_data = [[1, 0, 0],
                            [0, 2, 3],
                            [0, 3, 4],
                            [1, 2, 3]]
        columns = ["col1", "col2", "col3"]        
        data = pd.DataFrame(int_missing_data, columns=columns)

        base = MethodBase(data=data, train_data=None, test_data=None, split=True, target_field='', report_name=None, test_split_percentage=0.5)
        base['col4'] = ([5, 5], [2,2])

        validate = any(base.train_data['col4'].isnull()) and any(base.test_data['col4'].isnull())

        self.assertFalse(validate)

    def test_setitem_tupleuneven(self):
        
        int_missing_data = [[1, 0, 0],
                            [0, 2, 3],
                            [0, 3, 4],
                            [1, 2, 3]]
        columns = ["col1", "col2", "col3"]        
        data = pd.DataFrame(int_missing_data, columns=columns)

        base = MethodBase(data=data, train_data=None, test_data=None, split=True, target_field='', report_name=None, test_split_percentage=0.25)
        base['col4'] = ([5, 5, 5], [2])

        validate = any(base.train_data['col4'].isnull()) and any(base.test_data['col4'].isnull())

        self.assertFalse(validate)

    def test_dropcolumns(self):

        int_missing_data = [[1, 0, 0],
                            [0, 2, 3],
                            [0, 3, 4],
                            [1, 2, 3]]
        columns = ["col1", "col2", "col3"]        
        data = pd.DataFrame(int_missing_data, columns=columns)

        clean = Clean(data=data, report_name="test", test_split_percentage=0.5)
        clean_inst = clean.drop("col1", "col3", reason="Columns were unimportant.")

        validate = (clean_inst.train_data.columns == ['col2'] and clean_inst.test_data.columns == ['col2'] and isinstance(clean_inst, Clean))

        self.assertTrue(validate)

    def test_dropcolumns_keep(self):

        int_missing_data = [[1, 0, 0],
                            [0, 2, 3],
                            [0, 3, 4],
                            [1, 2, 3]]
        columns = ["col1", "col2", "col3"]        
        data = pd.DataFrame(int_missing_data, columns=columns)

        clean = Clean(data=data, report_name="test", test_split_percentage=0.5)
        clean_inst = clean.drop(keep=['col2'], reason="Columns were unimportant.")

        validate = (clean_inst.train_data.columns == ['col2'] and clean_inst.test_data.columns == ['col2'] and isinstance(clean_inst, Clean))

        self.assertTrue(validate)

    def test_dropcolumns_complex(self):

        int_missing_data = [[1, 0, 0, 3],
                            [0, 2, 3, 4],
                            [0, 3, 4, 4],
                            [1, 2, 3, 6]]
        columns = ["col1", "col2", "col3", "py"]        
        data = pd.DataFrame(int_missing_data, columns=columns)

        clean = Clean(data=data, train_data=None, test_data=None, split=True, target_field='', report_name="test", test_split_percentage=0.5)
        clean.drop("col1", keep=['col2'], regexp=r'col*', reason="Columns were unimportant.")

        validate = (list(clean.train_data.columns) == ['col2', 'py'] and list(clean.test_data.columns) ==  ['col2', 'py'])

        self.assertTrue(validate)    

    def test_getattr(self):
        
        int_missing_data = [[1, 0, 0],
                            [0, 2, 3],
                            [0, 3, 4],
                            [1, 2, 3]]
        columns = ["col1", "col2", "col3"]        
        data = pd.DataFrame(int_missing_data, columns=columns)

        base = MethodBase(data=data, train_data=None, test_data=None, split=True, target_field='', report_name="test", test_split_percentage=0.5)

        self.assertIsNotNone(base.col1)

    
    def test_setattr_new(self):
        
        int_missing_data = [[1, 0, 0],
                            [0, 2, 3],
                            [0, 3, 4],
                            [1, 2, 3]]
        columns = ["col1", "col2", "col3"]        
        data = pd.DataFrame(int_missing_data, columns=columns)

        base = MethodBase(data=data, train_data=None, test_data=None, split=True, target_field='', report_name="test", test_split_percentage=0.5)
        base.col4 = 4

        self.assertIsNotNone(base.col4)

    def test_setattr_old(self):
        
        int_missing_data = [[1, 0, 0],
                            [0, 2, 3],
                            [0, 3, 4],
                            [1, 2, 3]]
        columns = ["col1", "col2", "col3"]        
        data = pd.DataFrame(int_missing_data, columns=columns)

        base = MethodBase(data=data, train_data=None, test_data=None, split=True, target_field='', report_name="test", test_split_percentage=0.5)
        base._data_properties.target_field = "col3"

        self.assertEquals('col3', base.target_field)

    def test_setter(self):

        int_missing_data = [[1, 0, 0],
                            [0, 2, 3],
                            [0, 3, 4],
                            [1, 2, 3]]

        int_missing_data_rep = [[1, 0, 0],
                            [0, 2, 3],
                            [0, 3, 4],
                            [1, 2, 3]]

        base = MethodBase(data=int_missing_data, train_data=None, test_data=None, split=False, target_field='', report_name="test", test_split_percentage=0.5)

        base.data = int_missing_data_rep

        self.assertEquals(base.data, int_missing_data_rep)

    def test_where(self):

        int_missing_data = [[1, 0, 0],
                            [0, 2, 3],
                            [0, 3, 4],
                            [1, 2, 3]]

        columns = ["col1", "col2", "col3"]        
        data = pd.DataFrame(int_missing_data, columns=columns)

        base = MethodBase(data=data, train_data=None, test_data=None, split=False, target_field='', report_name="test", test_split_percentage=0.5)

        subset = base.where(col1=0, col2=2, col3=[3,4])
        validate = subset.values.tolist()

        self.assertListEqual(validate, [[0, 2, 3]])

    def test_groupbyanalysis(self):

        data = pd.DataFrame({'A': [1, 1, 2, 2], 
                           'B': [1, 2, 3, 4],
                           'C': np.random.randn(4),
                           'D': ['A', 'A', 'B', 'B']})

        base = MethodBase(data=data, train_data=None, test_data=None, split=False, target_field='', report_name="test", test_split_percentage=0.5)

        base.groupby_analysis(['A'])

        self.assertTrue(True)

    def test_groupby(self):

        data = pd.DataFrame({'A': [1, 1, 2, 2], 
                           'B': [1, 2, 3, 4],
                           'C': np.random.randn(4),
                           'D': ['A', 'A', 'B', 'B']})

        clean = Clean(data=data, train_data=None, test_data=None, split=False, target_field='', report_name="test", test_split_percentage=0.5)

        clean.groupby('A', replace=True)

        self.assertTrue(True)

if __name__ == "__main__":
    unittest.main()
