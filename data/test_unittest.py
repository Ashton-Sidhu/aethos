import unittest

import numpy as np
import pandas as pd

from data.data import Data
from data.util import *


class TestData(unittest.TestCase):

    def test_data_getfieldtypes(self):

        data = np.array([(1, 1, 0, "hi my name is pyautoml", "green", 532.1),
                        (2, 0, None, "this is my story", "yellow", 213.5),
                        (3, None, None, "its me", "yellow", 154.2)])
        columns = ["pid","col1", "col2", "col3", "col4", "col5"]

        dataset = pd.DataFrame(data, columns=columns)
        data = Data(dataset)
        data.GetInputTypes(dataset)
        self.assertDictEqual(data.field_types, {"col1": "numeric"
                                                ,"col3": "text"
                                                ,"col4": "str_categorical"
                                                ,"col5": "numeric"})

    def test_data_normalizecolumnames_dfcolumnames(self):

        data = np.zeros((4,4))
        columns = ["PID", "CapsLock", "space column name", "Caps Space"]

        dataset = pd.DataFrame(data, columns=columns)
        data = Data(dataset)
        new_df = data.NormalizeColNames(dataset)

        self.assertListEqual(new_df.columns.tolist(), ["pid", "capslock", "space_column_name", "caps_space"])
        self.assertDictEqual(data.colMapping, {"PID": "pid"
                                                ,"CapsLock": "capslock"
                                                ,"space column name": "space_column_name"
                                                ,"Caps Space": "caps_space"})

    def test_data_normalizecolumnames_colmapping(self):

        data = np.zeros((4,4))
        columns = ["PID", "CapsLock", "space column name", "Caps Space"]

        dataset = pd.DataFrame(data, columns=columns)
        data = Data(dataset)
        new_df = data.NormalizeColNames(dataset)

        self.assertDictEqual(data.colMapping, {"PID": "pid"
                                                ,"CapsLock": "capslock"
                                                ,"space column name": "space_column_name"
                                                ,"Caps Space": "caps_space"})     

    def test_data_reducedata(self):

        data = np.zeros((4,4))
        columns = ["col1", "col2", "col3", "col4"]

        dataset = pd.DataFrame(data, columns=columns)
        data = Data(dataset)
        data.field_types = {"col1": 0, "col3": 0}
        new_df = data.ReduceData(dataset)

        self.assertListEqual(new_df.columns.tolist(), ["col1", "col3"])

    def test_data_standardizedata(self):

        data = np.array([(1, 1, 0, "hi my name is pyautoml", "green", 532.1),
                        (2, 0, None, "this is my story", "yellow", 213.5),
                        (3, None, None, "its me", "yellow", 154.2)])
        columns = ["pid","col1", "col2", "col3", "col4", "col5"]

        dataset = pd.DataFrame(data, columns=columns)
        data = Data(dataset)
        new_df = data.StandardizeData(dataset)

        self.assertIsNotNone(new_df)

    def test_datautil_checkmissingdata(self):
        data = np.array([(1, 1, 0, "hi my name is pyautoml", "green", 532.1),
                        (2, 0, None, "this is my story", "yellow", 213.5),
                        (3, None, None, "its me", "yellow", 154.2)])
        columns = ["pid","col1", "col2", "col3", "col4", "col5"]

        dataset = pd.DataFrame(data, columns=columns)
        has_null = CheckMissingData(dataset)

        self.assertTrue(has_null)

    def test_datautil_getkeysbyvalue(self):
        data = {"eagle": "bird",
                "sparrow": "bird",
                "mosquito": "insect"}
        
        list_of_keys = GetKeysByValues(data, "bird")

        self.assertListEqual(list_of_keys, ["eagle", "sparrow"])

    def test_datautil_getlistofcols_duplicatecustomcols_overridefalse(self):
        data = {"eagle": "bird",
                "sparrow": "bird",
                "mosquito": "insect"}

        list_of_cols = GetListOfCols("bird", data, False, custom_cols=["eagle"])

        self.assertListEqual(list_of_cols, ["eagle", "sparrow"])

    def test_datautil_getlistofcols_uniquecustomcols_overridefalse(self):
        data = {"eagle": "bird",
                "sparrow": "bird",
                "mosquito": "insect"}

        list_of_cols = GetListOfCols("bird", data, False, custom_cols=["frog"])

        self.assertListEqual(list_of_cols, ["eagle", "sparrow", "frog"])

    def test_datautil_getlistofcols_overridetrue(self):
        data = {"eagle": "bird",
                "sparrow": "bird",
                "mosquito": "insect"}

        list_of_cols = GetListOfCols("bird", data, True, custom_cols=["frog"])

        self.assertListEqual(list_of_cols, ["frog"])

    def test_datautil_dropandreplacecolumns(self):
        data_zeros = np.zeros((2,2))
        columns_zeros = ["col1", "col2"]
        data_ones = np.ones((2,1))
        columns_ones = ["col3"]

        dataset_zeros = pd.DataFrame(data_zeros, columns=columns_zeros)
        dataset_ones = pd.DataFrame(data_ones, columns=columns_ones)
        df_new = DropAndReplaceColumns(dataset_zeros, "col2", dataset_ones)

        self.assertListEqual(df_new.columns.tolist(), ["col1", "col3"])

if __name__ == "__main__":
    unittest.main()
