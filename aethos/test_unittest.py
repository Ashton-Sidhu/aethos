import ast
import os
import unittest
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns

from aethos import Analysis


class Test_TestBase(unittest.TestCase):
    # @classmethod
    # def tearDownClass(cls):
    #     shutil.rmtree(str(Path.home()) + "/.aethos/reports/")

    def test_setitem_constant(self):

        int_missing_data = [[1, 0, 0], [0, 2, 3], [0, 3, 4], [1, 2, 3]]
        columns = ["col1", "col2", "col3"]
        data = pd.DataFrame(int_missing_data, columns=columns)

        base = Analysis(x_train=data, x_test=None, target="",)

        base["col4"] = 5

        validate = any(base.x_train["col4"].isnull()) and any(
            base.x_test["col4"].isnull()
        )

        self.assertFalse(validate)

    def test_setitem_equalsize_list(self):

        int_missing_data = [[1, 0, 0], [0, 2, 3], [0, 3, 4], [1, 2, 3]]
        columns = ["col1", "col2", "col3"]
        data_train = pd.DataFrame(int_missing_data, columns=columns)
        data_test = pd.DataFrame(int_missing_data, columns=columns)

        base = Analysis(x_train=data_train, x_test=data_test,)

        base["col4"] = [5, 5, 5, 5]

        validate = any(base.x_train["col4"].isnull()) and any(
            base.x_test["col4"].isnull()
        )

        self.assertFalse(validate)

    def test_setitem_traindata(self):

        int_missing_data = [[1, 0, 0], [0, 2, 3], [0, 3, 4], [1, 2, 3]]
        columns = ["col1", "col2", "col3"]
        data_train = pd.DataFrame(int_missing_data, columns=columns)
        data_test = pd.DataFrame([[1, 0, 0]], columns=columns)

        base = Analysis(x_train=data_train, x_test=data_test,)

        base["col4"] = [5, 5, 5, 5]

        validate = any(base.x_train["col4"].isnull())

        self.assertFalse(validate)

    def test_setitem_testdata(self):

        int_missing_data = [[1, 0, 0], [0, 2, 3], [0, 3, 4], [1, 2, 3]]
        columns = ["col1", "col2", "col3"]
        data_test = pd.DataFrame(int_missing_data, columns=columns)
        data_train = pd.DataFrame([[1, 0, 0]], columns=columns)

        base = Analysis(x_train=data_train, x_test=data_test,)

        base["col4"] = [5, 5, 5, 5]

        validate = any(base.x_test["col4"].isnull())

        self.assertFalse(validate)

    def test_setitem_tupleeven(self):

        int_missing_data = [[1, 0, 0], [0, 2, 3], [0, 3, 4], [1, 2, 3]]
        columns = ["col1", "col2", "col3"]
        data_train = pd.DataFrame(int_missing_data, columns=columns)
        data_test = pd.DataFrame(int_missing_data, columns=columns)

        base = Analysis(x_train=data_train, x_test=data_test)

        base["col4"] = ([5, 5, 5, 5], [2, 2, 2, 2])

        validate = any(base.x_train["col4"].isnull()) and any(
            base.x_test["col4"].isnull()
        )

        self.assertFalse(validate)

    def test_setitem_tupleuneven(self):

        int_missing_data = [[1, 0, 0], [0, 2, 3], [0, 3, 4], [1, 2, 3]]
        columns = ["col1", "col2", "col3"]
        data_train = pd.DataFrame(int_missing_data, columns=columns)
        data_test = pd.DataFrame(int_missing_data[0:2], columns=columns)

        base = Analysis(x_train=data_train, x_test=data_test)

        base["col4"] = ([5, 5, 5, 5], [2, 2])

        validate = any(base.x_train["col4"].isnull()) and any(
            base.x_test["col4"].isnull()
        )

        self.assertFalse(validate)

    def test_dropcolumns(self):

        int_missing_data = [[1, 0, 0], [0, 2, 3], [0, 3, 4], [1, 2, 3]]
        columns = ["col1", "col2", "col3"]
        data = pd.DataFrame(int_missing_data, columns=columns)

        clean = Analysis(x_train=data, x_test=data)
        clean_inst = clean.drop("col1", "col3", reason="Columns were unimportant.")

        validate = (
            clean_inst.x_train.columns == ["col2"]
            and clean_inst.x_test.columns == ["col2"]
            and isinstance(clean_inst, Analysis)
        )

        self.assertTrue(validate)

    def test_dropcolumns_keep(self):

        int_missing_data = [[1, 0, 0], [0, 2, 3], [0, 3, 4], [1, 2, 3]]
        columns = ["col1", "col2", "col3"]
        data = pd.DataFrame(int_missing_data, columns=columns)

        clean = Analysis(x_train=data, x_test=data)
        clean_inst = clean.drop(keep=["col2"], reason="Columns were unimportant.")

        validate = (
            clean_inst.x_train.columns == ["col2"]
            and clean_inst.x_test.columns == ["col2"]
            and isinstance(clean_inst, Analysis)
        )

        self.assertTrue(validate)

    def test_dropcolumns_complex(self):

        int_missing_data = [[1, 0, 0, 3], [0, 2, 3, 4], [0, 3, 4, 4], [1, 2, 3, 6]]
        columns = ["col1", "col2", "col3", "py"]
        data = pd.DataFrame(int_missing_data, columns=columns)

        clean = Analysis(x_train=data, x_test=data)
        clean.drop(
            "col1", keep=["col2"], regexp=r"col*", reason="Columns were unimportant."
        )

        validate = list(clean.x_train.columns) == ["col2", "py"] and list(
            clean.x_test.columns
        ) == ["col2", "py"]

        self.assertTrue(validate)

    def test_dropcolumns_regex(self):

        int_missing_data = [[1, 0, 0, 3], [0, 2, 3, 4], [0, 3, 4, 4], [1, 2, 3, 6]]
        columns = ["agent.hi", "agent.user_name", "agent.hello", "message"]
        data = pd.DataFrame(int_missing_data, columns=columns)

        clean = Analysis(x_train=data, x_test=None,)
        clean.drop(regexp=r"agent*")

        validate = clean.x_train.columns == ["message"]

        self.assertTrue(validate)

    def test_dropcolumns_error(self):

        int_missing_data = [[1, 0, 0, 3], [0, 2, 3, 4], [0, 3, 4, 4], [1, 2, 3, 6]]
        columns = ["col1", "col2", "col3", "py"]
        data = pd.DataFrame(int_missing_data, columns=columns)

        clean = Analysis(x_train=data, x_test=None,)

        self.assertRaises(TypeError, clean.drop, keep="col2")

    def test_getattr(self):

        int_missing_data = [[1, 0, 0], [0, 2, 3], [0, 3, 4], [1, 2, 3]]
        columns = ["col1", "col2", "col3"]
        data = pd.DataFrame(int_missing_data, columns=columns)

        base = Analysis(x_train=data, x_test=None,)

        self.assertIsNotNone(base.col1)

    def test_getattr_pandas(self):

        int_missing_data = [[1, 0, 0], [0, 2, 3], [0, 3, 4], [1, 2, 3]]
        columns = ["col1", "col2", "col3"]
        data = pd.DataFrame(int_missing_data, columns=columns)

        base = Analysis(x_train=data, x_test=None,)

        base.mean()

        self.assertTrue(True)

    def test_setattr_new(self):

        int_missing_data = [[1, 0, 0, 1], [0, 2, 3, 1], [0, 3, 4, 1], [1, 2, 3, 1]]
        columns = ["col1", "col2", "col3", "col4"]
        data = pd.DataFrame(int_missing_data, columns=columns)

        base = Analysis(x_train=data, x_test=None,)
        base["col5"] = 4

        self.assertListEqual(base.col5.tolist(), [4, 4, 4, 4])

    def test_setattr_testset(self):

        int_missing_data = [[1, 0, 0, 1], [0, 2, 3, 1], [0, 3, 4, 1], [1, 2, 3, 1]]
        columns = ["col1", "col2", "col3", "col4"]
        train_data = pd.DataFrame(int_missing_data, columns=columns)
        test_data = pd.DataFrame(int_missing_data[0:1], columns=columns)

        base = Analysis(x_train=train_data, x_test=test_data,)
        base["col5"] = [4]

        self.assertListEqual(base.x_test["col5"].tolist(), [4])

    def test_setattr_trainset(self):

        int_missing_data = [[1, 0, 0, 1], [0, 2, 3, 1], [0, 3, 4, 1], [1, 2, 3, 1]]
        columns = ["col1", "col2", "col3", "col4"]
        train_data = pd.DataFrame(int_missing_data[0:1], columns=columns)
        test_data = pd.DataFrame(int_missing_data, columns=columns)

        base = Analysis(x_train=train_data, x_test=test_data,)
        base["col5"] = [4]

        self.assertListEqual(base["col5"].tolist(), [4])

    def test_setattr_bothset(self):

        int_missing_data = [[1, 0, 0, 1], [0, 2, 3, 1], [0, 3, 4, 1], [1, 2, 3, 1]]
        columns = ["col1", "col2", "col3", "col4"]
        train_data = pd.DataFrame(int_missing_data[0:1], columns=columns)
        test_data = pd.DataFrame(int_missing_data, columns=columns)

        base = Analysis(x_train=train_data, x_test=test_data,)
        base["col5"] = ([4], [4, 4, 4, 4])

        self.assertListEqual(base["col5"].tolist(), [4])
        self.assertListEqual(base.x_test["col5"].tolist(), [4, 4, 4, 4])

    def test_setattr_old(self):

        int_missing_data = [[1, 0, 0], [0, 2, 3], [0, 3, 4], [1, 2, 3]]
        columns = ["col1", "col2", "col3"]
        data = pd.DataFrame(int_missing_data, columns=columns)

        base = Analysis(x_train=data, x_test=None,)
        base.target = "col3"

        self.assertEqual("col3", base.target)

    def test_setter(self):

        data = [[1, 0, 0], [0, 2, 3], [0, 3, 4], [1, 2, 3]]

        int_missing_data_rep = [[1, 0, 0], [0, 2, 3], [0, 3, 4], [1, 2, 3]]

        base = Analysis(x_train=data,)

        base.x_train = int_missing_data_rep

        self.assertEqual(base.x_train, int_missing_data_rep)

    def test_groupbyanalysis(self):

        data = pd.DataFrame(
            {
                "A": [1, 1, 2, 2],
                "B": [1, 2, 3, 4],
                "C": np.random.randn(4),
                "D": ["A", "A", "B", "B"],
            }
        )

        base = Analysis(x_train=data, x_test=None,)

        base.groupby_analysis(["A"])

        self.assertTrue(True)

    def test_groupby(self):

        data = pd.DataFrame(
            {
                "A": [1, 1, 2, 2],
                "B": [1, 2, 3, 4],
                "C": np.random.randn(4),
                "D": ["A", "A", "B", "B"],
            }
        )

        clean = Analysis(x_train=data, x_test=None,)

        clean.groupby("A", replace=True)

        self.assertTrue(True)

    def test_search(self):

        data = pd.DataFrame(
            {
                "A": [1, 1, 2, 2],
                "B": [1, 2, 3, 4],
                "C": np.random.randn(4),
                "D": ["A", "A", "B", "B"],
            }
        )

        clean = Analysis(x_train=data, x_test=None,)
        clean.search("A", replace=True)

        self.assertTrue(True)

    def test_search_notequal(self):

        data = pd.DataFrame(
            {
                "A": [1, 1, 2, 2],
                "B": [1, 2, 3, 4],
                "C": np.random.randn(4),
                "D": ["A", "A", "B", "B"],
            }
        )

        clean = Analysis(x_train=data, x_test=None,)
        clean.search("A", not_equal=True, replace=True)

        self.assertTrue(True)

    def test_gettargetmapping(self):

        data = pd.DataFrame(
            {
                "A": [1, 1, 2, 2],
                "B": [1, 2, 3, 4],
                "C": np.random.randn(4),
                "D": ["A", "A", "B", "B"],
            }
        )

        clean = Analysis(x_train=data, x_test=None,)

        self.assertIsNone(clean.target_mapping)

    def test_settargetmapping(self):

        data = pd.DataFrame(
            {
                "A": [1, 1, 2, 2],
                "B": [1, 2, 3, 4],
                "C": np.random.randn(4),
                "D": ["A", "A", "B", "B"],
            }
        )

        clean = Analysis(x_train=data, x_test=None,)
        clean.target_mapping = "a"

        self.assertEqual(clean.target_mapping, "a")

    def test_encodelabels(self):

        data = pd.DataFrame(
            {
                "A": [1, 1, 2, 2],
                "B": [1, 2, 3, 4],
                "C": np.random.randn(4),
                "D": ["Bca", "Abc", "Bca", "Bca"],
            }
        )

        clean = Analysis(x_train=data, x_test=None, target="D",)
        clean.encode_target()

        self.assertDictEqual(clean.target_mapping, {0: "Abc", 1: "Bca"})

    def test_write_data_tocsv(self):

        np.random.seed(42)
        df = pd.DataFrame(
            {
                "Google": np.random.randn(1000) + 0.2,
                "Apple": np.random.randn(1000) + 0.17,
                "date": pd.date_range("1/1/2000", periods=1000),
            }
        )

        clean = Analysis(x_train=df)
        clean.to_csv("test_write_data")
        os.remove("test_write_data_train.csv")

        self.assertTrue(True)

    def test_write_traindata_tocsv(self):

        int_missing_data = [[1, 0, 0], [0, 2, 3], [0, 3, 4], [1, 2, 3]]
        columns = ["col1", "col2", "col3"]
        train_data = pd.DataFrame(int_missing_data, columns=columns)
        test_data = pd.DataFrame(int_missing_data[0:1], columns=columns)

        base = Analysis(x_train=train_data, x_test=test_data,)

        base.to_csv("titanic123")
        os.remove("titanic123_train.csv")
        os.remove("titanic123_test.csv")

        self.assertTrue(True)

    def test_missing_data(self):

        int_missing_data = [
            [np.NaN, np.NaN, np.NaN],
            [0, 2, 3],
            [np.NaN, np.NaN, np.NaN],
            [np.NaN, np.NaN, np.NaN],
        ]
        columns = ["col1", "col2", "col3"]
        data = pd.DataFrame(int_missing_data, columns=columns)

        base = Analysis(x_train=data, x_test=None,)

        base.missing_values

        self.assertTrue(True)

    def test_describe(self):

        int_missing_data = [[1, 0, 0], [0, 2, 3], [0, 3, 4], [1, 2, 3]]
        columns = ["col1", "col2", "col3"]
        data = pd.DataFrame(int_missing_data, columns=columns)

        base = Analysis(x_train=data, x_test=None,)

        base.describe()

        self.assertTrue(True)

    def test_describe_col(self):

        int_missing_data = [[1, 0, 0], [0, 2, 3], [0, 3, 4], [1, 2, 3]]
        columns = ["col1", "col2", "col3"]
        data = pd.DataFrame(int_missing_data, columns=columns)

        base = Analysis(x_train=data, x_test=None,)

        base.describe_column("col1")

        self.assertTrue(True)

    def test_repr(self):

        int_missing_data = [[1, 0, 0], [0, 2, 3], [0, 3, 4], [1, 2, 3]]
        columns = ["col1", "col2", "col3"]
        data = pd.DataFrame(int_missing_data, columns=columns)

        base = Analysis(x_train=data, x_test=None,)

        print(base)

        self.assertTrue(True)

    def test_checklist(self):

        int_missing_data = [[1, 0, 0], [0, 2, 3], [0, 3, 4], [1, 2, 3]]
        columns = ["col1", "col2", "col3"]
        data = pd.DataFrame(int_missing_data, columns=columns)

        base = Analysis(x_train=data, x_test=None,)

        base.checklist()

        self.assertTrue(True)

    # def test_ytrain_split(self):

    #     data = [[1, 0, 0], [0, 2, 3], [0, 3, 4], [1, 2, 3]]
    #     columns = ["col1", "col2", "col3"]
    #     data = pd.DataFrame(data, columns=columns)

    #     base = Analysis(x_train=data, x_test=None, target="col3",)

    #     validate = len(base.y_train) == 2

    #     self.assertTrue(validate)

    def test_ytrain_nosplit(self):

        data = [[1, 0, 0], [0, 2, 3], [0, 3, 4], [1, 2, 3]]
        columns = ["col1", "col2", "col3"]
        data = pd.DataFrame(data, columns=columns)

        base = Analysis(x_train=data, x_test=None, target="col3",)

        validate = len(base.y_train) == 4

        self.assertTrue(validate)

    def test_ytrain_dne(self):

        data = [[1, 0, 0], [0, 2, 3], [0, 3, 4], [1, 2, 3]]
        columns = ["col1", "col2", "col3"]
        data = pd.DataFrame(data, columns=columns)

        base = Analysis(x_train=data, x_test=None,)

        base.y_train = [1, 1, 1, 1]
        validate = base.x_train["label"].tolist() == [
            1,
            1,
            1,
            1,
        ] and base.y_train.tolist() == [1, 1, 1, 1]

        self.assertTrue(validate)

    def test_ytest_split(self):

        data = [[1, 0, 0], [0, 2, 3], [0, 3, 4], [1, 2, 3]]
        columns = ["col1", "col2", "col3"]
        train_data = pd.DataFrame(data, columns=columns)
        test_data = pd.DataFrame(data, columns=columns)

        base = Analysis(x_train=data, x_test=test_data, target="col3",)

        validate = len(base.y_test) == 4

        self.assertTrue(validate)

    def test_ytest_dne(self):

        data = [[1, 0, 0], [0, 2, 3], [0, 3, 4], [1, 2, 3]]
        columns = ["col1", "col2", "col3"]
        train_data = pd.DataFrame(data, columns=columns)
        test_data = pd.DataFrame(data, columns=columns)

        base = Analysis(x_train=data, x_test=test_data,)

        base.y_test = [1, 1, 1, 1]

        validate = base.y_test.tolist() == [1, 1, 1, 1] and base.x_test[
            "label"
        ].tolist() == [1, 1, 1, 1,]

        self.assertTrue(validate)

    def test_columns_property(self):

        data = pd.DataFrame(np.random.rand(100, 10))

        base = Analysis(x_train=data, x_test=None, target="col3",)

        validate = base.columns

        self.assertTrue(len(validate) == 10)

    def test_todf_1(self):

        data = sns.load_dataset("iris")

        base = Analysis(x_train=data, x_test=None, target="species",)

        x_train = base.to_df()
        validate = x_train.values.tolist() == data.values.tolist()

        self.assertTrue(validate)

    def test_todf_2(self):

        data = sns.load_dataset("iris")

        base = Analysis(x_train=data, x_test=data, target="species",)

        x_train, x_test = base.to_df()

        validate = x_train is not None and x_test is not None

        self.assertTrue(validate)

    def test_normalize_column_names(self):

        data = np.zeros((4, 4))
        columns = ["PID", "CapsLock", "space column name", "Caps Space"]

        data = pd.DataFrame(data, columns=columns)

        base = Analysis(x_train=data, x_test=data,)

        base.standardize_column_names()
        validate = (
            base.columns == ["pid", "capslock", "space_column_name", "caps_space"]
            and base.x_test.columns.tolist() == base.x_train.columns.tolist()
        )

        self.assertTrue(validate)

    def test_json_normalize_split(self):

        data = pd.DataFrame(
            {
                "col1": [1, 2],
                "col2": [
                    ast.literal_eval("{'foo':1, 'bar':2, 'baz':{'foo':2, 'x':1}}"),
                    ast.literal_eval("{'foo':3, 'bar':5, 'baz':{'foo':2, 'x':1}}"),
                ],
            }
        )

        base = Analysis(x_train=data.copy(), x_test=data.copy())

        base.expand_json_column("col2")

        self.assertListEqual(base.columns, ["col1", "foo", "bar", "baz_foo", "baz_x"])
        self.assertListEqual(
            base.x_test.columns.tolist(), ["col1", "foo", "bar", "baz_foo", "baz_x"]
        )


if __name__ == "__main__":
    unittest.main()
