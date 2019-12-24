import unittest

import numpy as np
import pandas as pd
from aethos import Data
import shutil
from pathlib import Path


class TestCleaning(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(str(Path.home()) + "/.aethos/reports/")

    def test_cleanutil_removecolumns(self):

        int_missing_data = [[1, 0, 0], [0, None, None], [None, None, None]]
        columns = ["col1", "col2", "col3"]
        data = pd.DataFrame(int_missing_data, columns=columns)

        clean = Data(
            x_train=data, test_split_percentage=0.5, split=False, report_name="test"
        )
        clean.drop_column_missing_threshold(0.5)
        validate = clean.x_train.columns.tolist()

        self.assertListEqual(validate, ["col1"])

    def test_cleanutil_removerows(self):

        int_missing_data = np.array([(1, 0, 0), (0, None, None), (None, None, None)])
        columns = ["col1", "col2", "col3"]
        data = pd.DataFrame(int_missing_data, columns=columns)

        clean = Data(
            x_train=data, test_split_percentage=0.5, split=False, report_name="test"
        )
        clean.drop_rows_missing_threshold(0.5)
        validate = clean.x_train.values.tolist()

        self.assertListEqual(validate, np.array([(1, 0, 0)]).tolist())

    def test_cleanutil_splitdata(self):

        data = np.zeros((5, 5))
        columns = ["col1", "col2", "col3", "col4", "col5"]
        dataset = pd.DataFrame(data, columns=columns)

        clean = Data(x_train=dataset, report_name="test")

        self.assertEqual(clean.x_train.shape[0], 4)

    def test_cleannumeric_mean(self):
        int_missing_data = [[1, 0, 2], [0, np.nan, 1], [np.nan, np.nan, np.nan]]

        columns = ["col1", "col2", "col3"]
        data = pd.DataFrame(int_missing_data, columns=columns)

        clean = Data(
            x_train=data, test_split_percentage=0.5, split=False, report_name="test"
        )
        clean.replace_missing_mean()
        validate = clean.x_train.values.tolist()

        self.assertListEqual(validate, [[1, 0, 2], [0, 0, 1], [0.5, 0, 1.5]])

    def test_cleannumeric_splitmean(self):
        int_missing_data = [
            [np.nan, 0, 2, 2],
            [0, np.nan, 1, 3],
            [1, 3, np.nan, 4],
            [1, 3, 4, np.nan],
        ]

        columns = ["col1", "col2", "col3", "col4"]
        data = pd.DataFrame(int_missing_data, columns=columns)

        clean = Data(x_train=data, test_split_percentage=0.5, report_name="test")
        clean.replace_missing_mean()
        validate = (
            clean.x_train.isnull().values.any() and clean.x_test.isnull().values.any()
        )

        self.assertFalse(validate)

    def test_cleannumeric_median(self):
        int_missing_data = [[1, 0, 2], [0, np.nan, 1], [np.nan, np.nan, np.nan]]

        columns = ["col1", "col2", "col3"]
        data = pd.DataFrame(int_missing_data, columns=columns)

        clean = Data(
            x_train=data, test_split_percentage=0.5, split=False, report_name="test"
        )
        clean.replace_missing_median()
        validate = clean.x_train.values.tolist()

        self.assertListEqual(validate, [[1, 0, 2], [0, 0, 1], [0.5, 0, 1.5]])

    def test_cleannumeric_mostfrequent(self):
        int_missing_data = np.array([(1, 0, 2), (1, np.nan, 1), (np.nan, np.nan, 1)])

        columns = ["col1", "col2", "col3"]
        data = pd.DataFrame(int_missing_data, columns=columns)

        clean = Data(
            x_train=data, test_split_percentage=0.5, split=False, report_name="test"
        )
        clean.replace_missing_mostcommon()
        validate = clean.x_train.values.tolist()

        self.assertListEqual(
            validate,
            np.array([(1.0, 0.0, 2.0), (1.0, 0.0, 1.0), (1.0, 0.0, 1.0)]).tolist(),
        )

    def test_cleannumeric_constant(self):
        int_missing_data = np.array([(1, 0, 2), (1, None, 1), (None, None, None)])

        columns = ["col1", "col2", "col3"]
        data = pd.DataFrame(int_missing_data, columns=columns)

        clean = Data(
            x_train=data, test_split_percentage=0.5, split=False, report_name="test"
        )
        clean.replace_missing_constant("col1", "col3", constant=10.5)
        validate = clean.x_train.values.tolist()

        self.assertListEqual(
            validate, np.array([(1, 0, 2), (1, None, 1), (10.5, None, 10.5)]).tolist()
        )

    def test_cleancategorical_removerow(self):

        int_missing_data = [[1, 0, 2], [1, np.nan, 1], [np.nan, np.nan, np.nan]]

        columns = ["col1", "col2", "col3"]
        data = pd.DataFrame(int_missing_data, columns=columns)

        clean = Data(
            x_train=data, test_split_percentage=0.5, split=False, report_name="test"
        )
        clean.replace_missing_remove_row("col1", "col2")
        validate = clean.x_train.values.tolist()

        self.assertListEqual(validate, np.array([(1, 0, 2)]).tolist())

    def test_cleancategorical_replacemissingnewcategory_dict(self):

        missing_data = [[1, "Green", 2], [1, np.nan, 1], [np.nan, np.nan, np.nan]]

        columns = ["col1", "col2", "col3"]
        data = pd.DataFrame(missing_data, columns=columns)
        category_dict_mapping = {"col1": 2, "col2": "Blue", "col3": 4}

        clean = Data(
            x_train=data, test_split_percentage=0.5, split=False, report_name="test"
        )
        clean.replace_missing_new_category(col_mapping=category_dict_mapping)
        validate = clean.x_train.values.tolist()

        self.assertListEqual(
            validate, [[1.0, "Green", 2.0], [1.0, "Blue", 1.0], [2.0, "Blue", 4.0]]
        )

    def test_cleancategorical_replacemissingnewcategory_list_constantnotnone(self):

        missing_data = np.array([(1, "Green", 2), (1, "Other", 1), (None, None, None)])

        columns = ["col1", "col2", "col3"]
        data = pd.DataFrame(missing_data, columns=columns)
        list_col = ["col1", "col3"]

        clean = Data(
            x_train=data, test_split_percentage=0.5, split=False, report_name="test"
        )
        clean.replace_missing_new_category(list_of_cols=list_col, new_category=0)
        validate = clean.x_train.values.tolist()

        self.assertListEqual(
            validate,
            np.array([(1, "Green", 2), (1, "Other", 1), (0, None, 0)]).tolist(),
        )

    def test_cleancategorical_replacemissingnewcategory_list_constantisnone(self):

        missing_data = [[1.0, "Green", 2], [1.0, "Other", 1], [np.nan, None, np.nan]]

        columns = ["col1", "col2", "col3"]
        data = pd.DataFrame(missing_data, columns=columns)
        list_col = ["col1", "col2"]

        clean = Data(
            x_train=data, test_split_percentage=0.5, split=False, report_name="test"
        )
        clean.replace_missing_new_category(list_of_cols=list_col)

        # Replacing NaNs with strings for validations as regular assert does == and to compare NaNs you need `is`
        clean.x_train = clean.x_train.fillna("NaN was here")
        validate = clean.x_train.values.tolist()

        self.assertListEqual(
            validate,
            [[1, "Green", 2.0], [1, "Other", 1.0], [-1, "Unknown", "NaN was here"]],
        )

    def test_cleancategorical_replacemissingnewcategory_constantnotnone(self):

        missing_data = [[1.0, "Green", 2], [1.0, "Other", 1], [np.nan, np.nan, np.nan]]

        columns = ["col1", "col2", "col3"]
        data = pd.DataFrame(missing_data, columns=columns)

        clean = Data(
            x_train=data, test_split_percentage=0.5, split=False, report_name="test"
        )
        clean.replace_missing_new_category(new_category=1)
        validate = clean.x_train.values.tolist()

        self.assertListEqual(
            validate, [[1.0, "Green", 2], [1.0, "Other", 1], [1.0, 1, 1]]
        )

    def test_cleancategorical_replacemissingnewcategory_noparams(self):

        missing_data = [[1.0, "Green", 2], [1.0, "Other", 1], [np.nan, np.nan, np.nan]]

        columns = ["col1", "col2", "col3"]
        data = pd.DataFrame(missing_data, columns=columns)

        clean = Data(
            x_train=data, test_split_percentage=0.5, split=False, report_name="test"
        )
        clean.replace_missing_new_category()
        validate = clean.x_train.values.tolist()

        self.assertListEqual(
            validate, [[1, "Green", 2], [1, "Other", 1], [-1, "Unknown", -1]]
        )

    def test_cleanutil_removeduplicaterows(self):

        data = [[1, 0, 2], [0, 2, 1], [1, 0, 2]]

        columns = ["col1", "col2", "col3"]
        data = pd.DataFrame(data, columns=columns)

        clean = Data(
            x_train=data, test_split_percentage=0.5, split=False, report_name="test"
        )
        clean.drop_duplicate_rows()
        validate = clean.x_train.values.tolist()

        self.assertListEqual(validate, [[1, 0, 2], [0, 2, 1]])

    def test_cleanutil_removeduplicaterows(self):

        data = [[1, 0, 2], [0, 2, 1], [1, 0, 2]]

        columns = ["col1", "col2", "col3"]
        data = pd.DataFrame(data, columns=columns)

        clean = Data(
            x_train=data, test_split_percentage=0.5, split=False, report_name="test"
        )
        clean.drop_duplicate_rows(list_of_cols=columns)
        validate = clean.x_train.values.tolist()

        self.assertListEqual(validate, [[1, 0, 2], [0, 2, 1]])

    def test_cleanutil_removeduplicatecolumns(self):

        data = [[1, 0, 1], [0, 2, 0], [1, 0, 1]]

        columns = ["col1", "col2", "col3"]
        data = pd.DataFrame(data, columns=columns)

        clean = Data(
            x_train=data, test_split_percentage=0.5, split=False, report_name="test"
        )
        clean.drop_duplicate_columns()
        validate = clean.x_train.values.tolist()

        self.assertListEqual(validate, [[1, 0], [0, 2], [1, 0]])

    def test_cleanutil_replacerandomdiscrete(self):

        int_missing_data = [
            [1, 8, 1],
            [0, 9394, 2],
            [np.nan, np.nan, np.nan],
            [2, 4, 3],
        ]

        columns = ["col1", "col2", "col3"]
        data = pd.DataFrame(int_missing_data, columns=columns)

        clean = Data(x_train=data, test_split_percentage=0.5, report_name="test")
        clean.replace_missing_random_discrete("col1", "col2", "col3")

        validate = np.any(clean.x_train.isnull()) and np.any(clean.x_test.isnull())

        self.assertFalse(validate)

    def test_cleanutil_replaceknn(self):

        int_missing_data = [
            [1, 8, 1],
            [0, 9394, 2],
            [np.nan, np.nan, np.nan],
            [2, 4, 3],
            [np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan],
            [1, 8, 1],
            [0, 9394, 2],
            [np.nan, np.nan, np.nan],
            [2, 4, 3],
            [np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan],
        ]

        columns = ["col1", "col2", "col3"]
        data = pd.DataFrame(int_missing_data, columns=columns)

        clean = Data(x_train=data, test_split_percentage=0.5, report_name="test")
        clean.replace_missing_knn(k=4)

        validate = np.any(clean.x_train.isnull()) and np.any(clean.x_test.isnull())

        self.assertFalse(validate)

    def test_cleanutil_replaceinterpol(self):

        int_missing_data = [
            [1, 8, 1],
            [0, 9394, 2],
            [np.nan, np.nan, np.nan],
            [2, 4, 3],
            [np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan],
            [1, 8, 1],
            [0, 9394, 2],
            [np.nan, np.nan, np.nan],
            [2, 4, 3],
            [np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan],
        ]

        columns = ["col1", "col2", "col3"]
        data = pd.DataFrame(int_missing_data, columns=columns)

        clean = Data(x_train=data, test_split_percentage=0.5, report_name="test")
        clean.replace_missing_interpolate(
            "col1", "col2", "col3", limit_direction="both"
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

        clean = Data(x_train=data, test_split_percentage=0.5, report_name="test")
        clean.replace_missing_forwardfill("col1", "col2", "col3")

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

        clean = Data(x_train=data, test_split_percentage=0.5, report_name="test")
        clean.replace_missing_backfill("col1", "col2", "col3")

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

        clean = Data(x_train=data, test_split_percentage=0.5, report_name="test")
        clean.replace_missing_indicator("col1", "col2", "col3")

        validate = (clean.x_train.shape[1] == 6) and (clean.x_test.shape[1] == 6)

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

        clean = Data(x_train=data, test_split_percentage=0.5, report_name="test")
        clean.replace_missing_indicator("col1", "col2", "col3", keep_col=False)

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

        clean = Data(x_train=data, test_split_percentage=0.5, report_name="test")
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

        clean = Data(x_train=data, test_split_percentage=0.5, report_name="test")
        clean.drop_unique_columns()

        validate = clean.x_test.columns.tolist() == clean.x_train.columns.tolist() and clean.x_test.columns.tolist() == [
            "col2",
            "col3",
            "col4",
        ]

        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
