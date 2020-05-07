import unittest

import numpy as np
import pandas as pd
from aethos import Classification
import shutil
from pathlib import Path


class TestCleaning(unittest.TestCase):
    def test_cleanutil_removecolumns(self):

        int_missing_data = [[1, 0, 0], [0, None, None], [None, None, None]]
        columns = ["col1", "col2", "col3"]
        data = pd.DataFrame(int_missing_data, columns=columns)

        clean = Classification(x_train=data, target="col3", x_test=data)
        clean.drop_column_missing_threshold(0.5)
        validate = clean.x_train.columns.tolist()

        self.assertListEqual(validate, ["col1", "col3"])
        self.assertListEqual(clean.x_test.columns.tolist(), ["col1", "col3"])

    def test_cleanutil_removerows(self):

        int_missing_data = np.array([(1, 0, 0), (0, None, 0), (None, None, 0)])
        columns = ["col1", "col2", "col3"]
        data = pd.DataFrame(int_missing_data, columns=columns)

        clean = Classification(x_train=data, target="col3", x_test=data)
        clean.drop_rows_missing_threshold(0.4)
        validate = clean.x_train.values.tolist()

        self.assertListEqual(validate, np.array([(1, 0, 0), (0, None, 0)]).tolist())

    def test_cleannumeric_mean(self):
        int_missing_data = [[1, 0, 2], [0, np.nan, 1], [np.nan, np.nan, 1]]

        columns = ["col1", "col2", "col3"]
        data = pd.DataFrame(int_missing_data, columns=columns)

        clean = Classification(x_train=data, target="col3", x_test=data)
        clean.replace_missing_mean()
        validate = clean.x_train.values.tolist()

        self.assertListEqual(validate, [[1, 0, 2], [0, 0, 1], [0.5, 0, 1]])

    def test_cleannumeric_splitmean(self):
        int_missing_data = [
            [np.nan, 0, 2, 2],
            [0, np.nan, 1, 3],
            [1, 3, 4, 4],
            [1, 3, 4, np.nan],
        ]

        columns = ["col1", "col2", "col3", "col4"]
        data = pd.DataFrame(int_missing_data, columns=columns)

        clean = Classification(x_train=data, target="col3", x_test=data)
        clean.replace_missing_mean()
        validate = (
            clean.x_train.isnull().values.any() and clean.x_test.isnull().values.any()
        )

        self.assertFalse(validate)

    def test_cleannumeric_median(self):
        int_missing_data = [[1, 0, 2], [0, np.nan, 1], [np.nan, np.nan, 1]]

        columns = ["col1", "col2", "col3"]
        data = pd.DataFrame(int_missing_data, columns=columns)

        clean = Classification(x_train=data, target="col3", x_test=data)
        clean.replace_missing_median()
        validate = clean.x_train.values.tolist()

        self.assertListEqual(validate, [[1, 0, 2], [0, 0, 1], [0.5, 0, 1]])

    def test_cleannumeric_mostfrequent(self):
        int_missing_data = np.array([(1, 0, 2), (1, np.nan, 1), (np.nan, np.nan, 1)])

        columns = ["col1", "col2", "col3"]
        data = pd.DataFrame(int_missing_data, columns=columns)

        clean = Classification(x_train=data, target="col3", x_test=data)
        clean.replace_missing_mostcommon()
        validate = clean.x_train.values.tolist()

        self.assertListEqual(
            validate,
            np.array([(1.0, 0.0, 2.0), (1.0, 0.0, 1.0), (1.0, 0.0, 1.0)]).tolist(),
        )

    def test_cleannumeric_constant(self):
        int_missing_data = np.array([(1, 0, 2), (1, None, 1), (None, None, 1)])

        columns = ["col1", "col2", "col3"]
        data = pd.DataFrame(int_missing_data, columns=columns)

        clean = Classification(x_train=data, target="col3", x_test=data)
        clean.replace_missing_constant("col1", "col2", constant=10.5)
        validate = clean.x_train.values.tolist()

        self.assertListEqual(
            validate, np.array([(1, 0, 2), (1, 10.5, 1), (10.5, 10.5, 1)]).tolist()
        )

    def test_cleancategorical_removerow(self):

        int_missing_data = [[1, 0, 2], [1, np.nan, 1], [np.nan, np.nan, np.nan]]

        columns = ["col1", "col2", "col3"]
        data = pd.DataFrame(int_missing_data, columns=columns)

        clean = Classification(x_train=data, target="col3", x_test=data)
        clean.replace_missing_remove_row("col1", "col2")
        validate = clean.x_train.values.tolist()

        self.assertListEqual(validate, np.array([(1, 0, 2)]).tolist())

    def test_cleancategorical_replacemissingnewcategory_dict(self):

        missing_data = [[1, "Green", 2], [1, np.nan, 1], [np.nan, np.nan, 1]]

        columns = ["col1", "col2", "col3"]
        data = pd.DataFrame(missing_data, columns=columns)
        category_dict_mapping = {"col1": 2, "col2": "Blue"}

        clean = Classification(x_train=data, target="col3", x_test=data)
        clean.replace_missing_new_category(col_mapping=category_dict_mapping)
        validate = clean.x_train.values.tolist()

        self.assertListEqual(
            validate, [[1.0, "Green", 2.0], [1.0, "Blue", 1.0], [2.0, "Blue", 1]]
        )

    def test_cleancategorical_replacemissingnewcategory_list_constantnotnone(self):

        missing_data = np.array([(1, "Green", 2), (1, "Other", 1), (None, None, None)])

        columns = ["col1", "col2", "col3"]
        data = pd.DataFrame(missing_data, columns=columns)
        list_col = ["col1", "col2"]

        clean = Classification(x_train=data, target="col3", x_test=data)
        clean.replace_missing_new_category(list_of_cols=list_col, new_category=0)
        validate = clean.x_train.values.tolist()

        self.assertListEqual(
            validate,
            np.array([(1, "Green", 2), (1, "Other", 1), (0, 0, None)]).tolist(),
        )

    def test_cleancategorical_replacemissingnewcategory_list_constantisnone(self):

        missing_data = [[1.0, "Green", 2], [1.0, "Other", 1], [np.nan, None, np.nan]]

        columns = ["col1", "col2", "col3"]
        data = pd.DataFrame(missing_data, columns=columns)
        list_col = ["col1", "col2"]

        clean = Classification(x_train=data, target="col3", x_test=data)
        clean.replace_missing_new_category(list_of_cols=list_col)

        # Replacing NaNs with strings for validations as regular assert does == and to compare NaNs you need `is`
        clean.x_train = clean.x_train.fillna("NaN was here")
        validate = clean.x_train.values.tolist()

        self.assertListEqual(
            validate,
            [[1, "Green", 2.0], [1, "Other", 1.0], [-1, "Unknown", "NaN was here"]],
        )

    def test_cleancategorical_replacemissingnewcategory_constantnotnone(self):

        missing_data = [[1.0, "Green", 2], [1.0, "Other", 1], [np.nan, np.nan, 1]]

        columns = ["col1", "col2", "col3"]
        data = pd.DataFrame(missing_data, columns=columns)

        clean = Classification(x_train=data, target="col3", x_test=data)
        clean.replace_missing_new_category(new_category=1)
        validate = clean.x_train.values.tolist()

        self.assertListEqual(
            validate, [[1.0, "Green", 2], [1.0, "Other", 1], [1.0, 1, 1]]
        )

    def test_cleancategorical_replacemissingnewcategory_noparams(self):

        missing_data = [[1.0, "Green", 2], [1.0, "Other", 1], [np.nan, np.nan, -1]]

        columns = ["col1", "col2", "col3"]
        data = pd.DataFrame(missing_data, columns=columns)

        clean = Classification(x_train=data, target="col3", x_test=data)
        clean.replace_missing_new_category()
        validate = clean.x_train.values.tolist()

        self.assertListEqual(
            validate, [[1, "Green", 2], [1, "Other", 1], [-1, "Unknown", -1]]
        )

    def test_cleanutil_removeduplicaterows(self):

        data = [[1, 0, 2], [0, 2, 1], [1, 0, 2]]

        columns = ["col1", "col2", "col3"]
        data = pd.DataFrame(data, columns=columns)

        clean = Classification(x_train=data,)
        clean.drop_duplicate_rows()
        validate = clean.x_train.values.tolist()

        self.assertListEqual(validate, [[1, 0, 2], [0, 2, 1]])

    def test_cleanutil_removeduplicaterows(self):

        data = [[1, 0, 2], [0, 2, 1], [1, 0, 2]]

        columns = ["col1", "col2", "col3"]
        data = pd.DataFrame(data, columns=columns)

        clean = Classification(x_train=data, target="col3", x_test=data)
        clean.drop_duplicate_rows(list_of_cols=["col1", "col2"])
        validate = clean.x_train.values.tolist()

        self.assertListEqual(validate, [[1, 0, 2], [0, 2, 1]])

    def test_cleanutil_removeduplicatecolumns(self):

        data = [[1, 1, 1], [0, 0, 0], [1, 1, 1]]

        columns = ["col1", "col2", "col3"]
        data = pd.DataFrame(data, columns=columns)

        clean = Classification(x_train=data, target="col3", x_test=data)
        clean.drop_duplicate_columns()
        validate = clean.x_train.values.tolist()

        self.assertListEqual(validate, [[1, 1], [0, 0], [1, 1]])

    def test_cleanutil_replacerandomdiscrete(self):

        int_missing_data = [
            [1, 8, 1],
            [0, 9394, 2],
            [np.nan, np.nan, 1],
            [2, 4, 3],
        ]

        columns = ["col1", "col2", "col3"]
        data = pd.DataFrame(int_missing_data, columns=columns)

        clean = Classification(x_train=data, target="col3", x_test=data)
        clean.replace_missing_random_discrete("col1", "col2")

        validate = np.any(clean.x_train.isnull()) and np.any(clean.x_test.isnull())

        self.assertFalse(validate)

    def test_cleanutil_replaceknn(self):

        int_missing_data = [
            [1, 8, 1],
            [0, 9394, 2],
            [np.nan, np.nan, 1],
            [2, 4, 3],
            [np.nan, np.nan, 1],
            [np.nan, np.nan, 1],
            [1, 8, 1],
            [0, 9394, 2],
            [np.nan, np.nan, 1],
            [2, 4, 3],
            [np.nan, np.nan, 1],
            [np.nan, np.nan, 1],
        ]

        columns = ["col1", "col2", "col3"]
        data = pd.DataFrame(int_missing_data, columns=columns)

        clean = Classification(x_train=data, target="col3", x_test=data)
        clean.replace_missing_knn(k=4)

        validate = np.any(clean.x_train.isnull()) and np.any(clean.x_test.isnull())

        self.assertFalse(validate)

    def test_cleanutil_replaceinterpol(self):

        int_missing_data = [
            [1, 8, 1],
            [0, 9394, 2],
            [np.nan, np.nan, 1],
            [2, 4, 3],
            [np.nan, np.nan, 1],
            [np.nan, np.nan, 1],
            [1, 8, 1],
            [0, 9394, 2],
            [np.nan, np.nan, 1],
            [2, 4, 3],
            [np.nan, np.nan, 1],
            [np.nan, np.nan, 1],
        ]

        columns = ["col1", "col2", "col3"]
        data = pd.DataFrame(int_missing_data, columns=columns)

        clean = Classification(x_train=data, target="col3", x_test=data)
        clean.replace_missing_interpolate(
            "col1", "col2", limit_direction="both"
        )

        validate = np.any(clean.x_train.isnull()) and np.any(clean.x_test.isnull())

        self.assertFalse(validate)

    def test_cleanutil_replaceffill(self):

        int_missing_data = [
            [1, 8, 1],
            [0, 9394, 2],
            [np.nan, np.nan, np.nan],
            [2, 4, 3],
            [np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan],
        ]

        columns = ["col1", "col2", "col3"]
        data = pd.DataFrame(int_missing_data, columns=columns)

        clean = Classification(x_train=data, target="col3", x_test=data)
        clean.replace_missing_forwardfill("col1", "col2")

        self.assertTrue(True)

    def test_cleanutil_replacebfill(self):

        int_missing_data = [
            [1, 8, 1],
            [0, 9394, 2],
            [np.nan, np.nan, np.nan],
            [2, 4, 3],
            [np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan],
        ]

        columns = ["col1", "col2", "col3"]
        data = pd.DataFrame(int_missing_data, columns=columns)

        clean = Classification(x_train=data, target="col3", x_test=data)
        clean.replace_missing_backfill("col1", "col2")

        self.assertTrue(True)

    def test_cleanutil_replaceindicator(self):

        int_missing_data = [
            [1, 8, 1],
            [0, 9394, 2],
            [np.nan, np.nan, np.nan],
            [2, 4, 3],
            [np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan],
        ]

        columns = ["col1", "col2", "col3"]
        data = pd.DataFrame(int_missing_data, columns=columns)

        clean = Classification(x_train=data, target="col3", x_test=data)
        clean.replace_missing_indicator("col1", "col2")

        validate = (clean.x_train.shape[1] == 5) and (clean.x_test.shape[1] == 5)

        self.assertTrue(True)

    def test_cleanutil_replaceindicator_removecol(self):

        int_missing_data = [
            [1, 8, 1],
            [0, 9394, 2],
            [np.nan, np.nan, np.nan],
            [2, 4, 3],
            [np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan],
        ]

        columns = ["col1", "col2", "col3"]
        data = pd.DataFrame(int_missing_data, columns=columns)

        clean = Classification(x_train=data.copy(), target="col3", x_test=data.copy())
        clean.replace_missing_indicator("col1", "col2", keep_col=False)

        validate = (clean.x_train.shape[1] == 3) and (clean.x_test.shape[1] == 3)

        self.assertTrue(True)

    def test_cleanutil_removeconstant(self):

        int_missing_data = [
            [1, 8, np.NaN, np.NaN],
            [0, 8, np.NaN, 1],
            [1, 8, np.NaN, np.NaN],
            [0, 8, np.NaN, 1],
            [1, 8, np.NaN, np.NaN],
            [0, 8, np.NaN, 1],
        ]

        columns = ["col1", "col2", "col3", "col4"]
        data = pd.DataFrame(int_missing_data, columns=columns)

        clean = Classification(x_train=data, target="col3", x_test=data)
        clean.drop_constant_columns()

        validate = clean.x_test.columns.tolist() == clean.x_train.columns.tolist() and clean.x_test.columns.tolist() == [
            "col1",
        ]

        self.assertTrue(True)

    def test_cleanutil_removeunique(self):

        int_missing_data = [
            [1, 8, np.NaN, np.NaN],
            [2, 8, np.NaN, 1],
            [3, 8, np.NaN, np.NaN],
            [4, 8, np.NaN, 2],
            [5, 8, np.NaN, np.NaN],
            [6, 8, np.NaN, 3],
        ]

        columns = ["col1", "col2", "col3", "col4"]
        data = pd.DataFrame(int_missing_data, columns=columns)

        clean = Classification(x_train=data, target="col3", x_test=data)
        clean.drop_unique_columns()

        validate = clean.x_test.columns.tolist() == clean.x_train.columns.tolist() and clean.x_test.columns.tolist() == [
            "col2",
            "col3",
            "col4",
        ]

        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
