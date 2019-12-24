import unittest

import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import shutil

from aethos.core import Data


class Test_TestBase(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(str(Path.home()) + "/.aethos/reports/")

    def test_histogram_1(self):

        data = sns.load_dataset("iris")

        base = Data(
            x_train=data,
            x_test=None,
            split=True,
            target_field="species",
            report_name="test",
            test_split_percentage=0.5,
        )

        base.histogram("sepal_length")

        self.assertTrue(True)

    def test_histogram_multi(self):

        data = sns.load_dataset("iris")

        base = Data(
            x_train=data,
            x_test=None,
            split=True,
            target_field="species",
            report_name="test",
            test_split_percentage=0.5,
        )

        base.histogram("sepal_length", "sepal_width")

        self.assertTrue(True)

    def test_pairplot(self):

        data = sns.load_dataset("iris")

        base = Data(
            x_train=data,
            x_test=None,
            split=True,
            target_field="species",
            report_name="test",
            test_split_percentage=0.5,
        )

        base.pairplot()

        self.assertTrue(True)

    def test_jointplot(self):

        data = sns.load_dataset("iris")

        base = Data(
            x_train=data,
            x_test=None,
            split=True,
            target_field="species",
            report_name="test",
            test_split_percentage=0.5,
        )

        base.jointplot(x="sepal_width", y="sepal_length")

        self.assertTrue(True)

    def test_rainccloudplot(self):

        data = sns.load_dataset("iris")

        base = Data(
            x_train=data,
            x_test=None,
            split=True,
            target_field="species",
            report_name="test",
            test_split_percentage=0.5,
        )

        base.raincloud(x="sepal_width", y="sepal_length")

        self.assertTrue(True)

    def test_barplot(self):

        data = sns.load_dataset("iris")

        base = Data(
            x_train=data,
            x_test=None,
            split=True,
            target_field="species",
            report_name="test",
            test_split_percentage=0.5,
        )

        base.barplot(
            x="species", y=["sepal_length", "sepal_width"], method="mean", orient="h",
        )
        base.barplot(
            x="species", y=["sepal_length", "sepal_width"], method="mean", orient="v",
        )

        self.assertTrue(True)

    def test_correlation_plot(self):

        data = pd.DataFrame(np.random.rand(100, 10))

        base = Data(
            x_train=data,
            x_test=None,
            split=True,
            target_field="col3",
            report_name="test",
            test_split_percentage=0.5,
        )

        base.correlation_matrix(data_labels=True, hide_mirror=True)

        self.assertTrue(True)

    def test_lineplot(self):

        np.random.seed(42)
        df = pd.DataFrame(
            {
                "Google": np.random.randn(1000) + 0.2,
                "Apple": np.random.randn(1000) + 0.17,
                "date": pd.date_range("1/1/2000", periods=1000),
            }
        )

        clean = Data(x_train=df, split=False)
        clean.lineplot(x="date", y=["Google", "Apple"], show_figure=False)

        self.assertTrue(True)
