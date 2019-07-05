import unittest

import numpy as np
import pandas as pd

from categorical import *
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
        validate = clean.df.columns.tolist()

        self.assertListEqual(validate, ["col1"])

    def test_clean_removerows(self):

        int_missing_data = np.array([(1, 0, 0),
                                 (0, None, None),
                                 (None, None, None)])
        columns = ["col1", "col2", "col3"]        
        data = pd.DataFrame(int_missing_data, columns=columns)

        clean = Clean(data)
        clean.RemoveRows(0.5)
        validate = clean.df.values.tolist()

        self.assertListEqual(validate, np.array([(1, 0, 0)]).tolist())

    def test_cleannumeric_mean(self):
        int_missing_data = np.array([(1, 0, 2),
                                 (0, None, 1),
                                 (None, None, None)])

        columns = ["col1", "col2", "col3"]        
        data = pd.DataFrame(int_missing_data, columns=columns)

        clean_data = ReplaceMissingMeanMedianMode("mean", data=data)
        validate = clean_data.values.tolist()

        self.assertListEqual(validate, np.array([(1, 0, 2),
                                                (0, 0, 1),
                                                (0.5, 0, 1.5)]).tolist())

    def test_cleannumeric_median(self):
        int_missing_data = np.array([(1, 0, 2),
                                 (0, None, 1),
                                 (None, None, None)])

        columns = ["col1", "col2", "col3"]        
        data = pd.DataFrame(int_missing_data, columns=columns)

        clean_data = ReplaceMissingMeanMedianMode("median", data=data)
        validate = clean_data.values.tolist()

        self.assertListEqual(validate, np.array([(1, 0, 2),
                                                (0, 0, 1),
                                                (0.5, 0, 1.5)]).tolist())

    def test_cleannumeric_mostfrequent(self):
        int_missing_data = np.array([(1, 0, 2),
                                 (1, np.nan, 1),
                                 (np.nan, np.nan, 1)])

        columns = ["col1", "col2", "col3"]        
        data = pd.DataFrame(int_missing_data, columns=columns)

        clean_data= ReplaceMissingMeanMedianMode("most_frequent", data=data)
        validate = clean_data.values.tolist()

        self.assertListEqual(validate, np.array([(1., 0., 2.),
                                                (1., 0., 1.),
                                                (1., 0., 1.)]).tolist())

    def test_cleannumeric_constant(self):
        int_missing_data = np.array([(1, 0, 2),
                                 (1, None, 1),
                                 (None, None, None)])

        columns = ["col1", "col2", "col3"]        
        data = pd.DataFrame(int_missing_data, columns=columns)

        clean_data = ReplaceMissingConstant(10.5, ["col1", "col3"], data=data)
        validate = clean_data.values.tolist()

        self.assertListEqual(validate, np.array([(1, 0, 2),
                                                (1, None, 1),
                                                (10.5, None, 10.5)]).tolist())

    def test_cleancategorical_removerow(self):

        int_missing_data = np.array([(1, 0, 2),
                                 (1, None, 1),
                                 (None, None, None)])

        columns = ["col1", "col2", "col3"]        
        data = pd.DataFrame(int_missing_data, columns=columns)

        clean_data = ReplaceMissingRemoveRow(["col1", "col2"], data=data)
        validate = clean_data.values.tolist()

        self.assertListEqual(validate, np.array([(1, 0, 2)]).tolist())

    def test_cleancategorical_replacemissingnewcategory_dict(self):

        missing_data = np.array([(1, "Green", 2),
                                 (1, None, 1),
                                 (None, None, None)])

        columns = ["col1", "col2", "col3"]        
        data = pd.DataFrame(missing_data, columns=columns)
        category_dict_mapping = {"col1": 2, "col2": "Blue"}

        clean_data = ReplaceMissingNewCategory(col_to_category=category_dict_mapping, data=data)
        validate = clean_data.values.tolist()

        self.assertListEqual(validate, np.array([(1, "Green", 2),
                                                (1, "Blue", 1),
                                                (2, "Blue", None)]).tolist())

    def test_cleancategorical_replacemissingnewcategory_list_constantnotnone(self):

        missing_data = np.array([(1, "Green", 2),
                                 (1, "Other", 1),
                                 (None, None, None)])

        columns = ["col1", "col2", "col3"]        
        data = pd.DataFrame(missing_data, columns=columns)
        list_col = ["col1", "col3"]

        clean_data = ReplaceMissingNewCategory(col_to_category=list_col, constant=0, data=data)
        validate = clean_data.values.tolist()

        self.assertListEqual(validate, np.array([(1, "Green", 2),
                                                (1, "Other", 1),
                                                (0, None, 0)]).tolist())

    def test_cleancategorical_replacemissingnewcategory_list_constantisnone(self):

        missing_data = [[1.0, "Green", 2],
                        [1.0, "Other", 1],
                        [np.nan, None, np.nan]]

        columns = ["col1", "col2", "col3"]        
        data = pd.DataFrame(missing_data, columns=columns)
        list_col = ["col1", "col2"]

        clean_data = ReplaceMissingNewCategory(col_to_category=list_col, data=data)
        #Replacing NaNs with strings for validations as regular assert does == and to compare NaNs you need `is`
        clean_data = clean_data.fillna("NaN was here")
        validate = clean_data.values.tolist()

        self.assertListEqual(validate, [[1, "Green", 2.0],
                                        [1, "Other", 1.0],
                                        [-1, "Unknown", "NaN was here"]])

    def test_cleancategorical_replacemissingnewcategory_constantnotnone(self):

        missing_data = np.array([(1.0, "Green", 2),
                                 (1.0, "Other", 1),
                                 (np.nan, None, np.nan)])

        columns = ["col1", "col2", "col3"]        
        data = pd.DataFrame(missing_data, columns=columns)

        clean_data = ReplaceMissingNewCategory(constant=1, data=data)
        validate = clean_data.values.tolist()

        self.assertListEqual(validate, [[1.0, "Green", 2],
                                        [1.0, "Other", 1],
                                        [1.0, 1, 1]])

    def test_cleancategorical_replacemissingnewcategory_noparams(self):

        missing_data = [[1.0, "Green", 2],
                        [1.0, "Other", 1],
                        [np.nan, np.nan, np.nan]]

        columns = ["col1", "col2", "col3"]        
        data = pd.DataFrame(missing_data, columns=columns)

        clean_data = ReplaceMissingNewCategory(data=data)
        validate = clean_data.values.tolist()

        self.assertListEqual(validate, [[1, "Green", 2],
                                        [1, "Other", 1],
                                        [-1, "Unknown", -1]])






if __name__ == "__main__":
    unittest.main()
