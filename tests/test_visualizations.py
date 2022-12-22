import unittest

import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import shutil

from aethos.analysis import Analysis
from sklearn.datasets import make_blobs


class Test_TestBase(unittest.TestCase):
    def test_histogram_1(self):

        data = sns.load_dataset("iris")

        base = Analysis(x_train=data, target="species",)

        base.histogram("sepal_length")

        self.assertTrue(True)

    def test_histogram_multi(self):

        data = sns.load_dataset("iris")

        base = Analysis(x_train=data, target="species",)

        base.histogram()

        self.assertTrue(True)

    def test_pairplot(self):

        data = sns.load_dataset("iris")

        base = Analysis(x_train=data, x_test=None, target="species",)

        base.pairplot()

        self.assertTrue(True)

    def test_pairplot_custom(self):

        data = sns.load_dataset("iris")

        base = Analysis(x_train=data, x_test=None, target="species",)

        base.pairplot(diag_kind="hist", upper_kind="scatter", lower_kind="kde")

        self.assertTrue(True)

    def test_jointplot(self):

        data = sns.load_dataset("iris")

        base = Analysis(x_train=data, target="species",)

        base.jointplot(x="sepal_width", y="sepal_length")

        self.assertTrue(True)

    def test_rainccloudplot(self):

        data = sns.load_dataset("iris")

        base = Analysis(x_train=data, x_test=None, target="species",)

        base.raincloud(x="sepal_width", y="sepal_length")

        self.assertTrue(True)

    def test_barplot(self):

        data = sns.load_dataset("iris")

        base = Analysis(x_train=data, x_test=None, target="species",)

        base.barplot(
            x="species",
            y="sepal_length",
            method="mean",
            orient="h",
            barmode="group",
            asc=True,
        )
        base.barplot(x="species", method="mean", barmode="group", asc=False)

        self.assertTrue(True)

    def test_pieplot(self):

        import plotly.express as px

        data = px.data.tips()

        base = Analysis(x_train=data, x_test=None,)

        base.pieplot("tip", "day")

        self.assertTrue(True)

    def test_boxplot(self):

        data = sns.load_dataset("iris")

        base = Analysis(x_train=data, x_test=None, target="species",)

        base.boxplot(x="species", y="sepal_width", color="species")

        self.assertTrue(True)

    def test_vioplot(self):

        data = sns.load_dataset("iris")

        base = Analysis(x_train=data, x_test=None, target="species",)

        base.violinplot(x="species", y="sepal_width", color="species")

        self.assertTrue(True)

    def test_correlation_plot(self):

        data = pd.DataFrame(np.random.rand(100, 10))

        base = Analysis(x_train=data, x_test=None, target="col3",)

        base.correlation_matrix(data_labels=True, hide_mirror=True)

        self.assertTrue(True)

    def test_lineplot(self):

        np.random.seed(42)
        df = pd.DataFrame(
            {
                "Google": np.random.randn(100) + 0.2,
                "Apple": np.random.randn(100) + 0.17,
                "date": pd.date_range("1/1/2000", periods=100),
            }
        )

        clean = Analysis(x_train=df)
        clean.lineplot(x="date", y="Google", color="Apple")

        self.assertTrue(True)

    def test_plot_clusters_pca(self):

        data, label = make_blobs(100, 4, centers=3)

        df = pd.DataFrame(data)
        df["label"] = label
        df = Analysis(df)
        df.plot_dim_reduction("label", algo="pca", dim=2)

        self.assertTrue(True)

    def test_plot_clusters_lle(self):

        data, label = make_blobs(100, 4, centers=3)

        df = pd.DataFrame(data)
        df["label"] = label

        df = Analysis(df)

        df.plot_dim_reduction("label", algo="lle", dim=2)

        self.assertTrue(True)

    def test_plot_clusters_svd(self):

        data, label = make_blobs(100, 4, centers=3)

        df = pd.DataFrame(data)
        df["label"] = label

        df = Analysis(df)

        df.plot_dim_reduction("label", algo="tsvd", dim=2)

        self.assertTrue(True)

    def test_plot_clusters_3d(self):

        data, label = make_blobs(100, 4, centers=3)
        df = pd.DataFrame(data)
        df["label"] = label

        df = Analysis(df)

        df.plot_dim_reduction("label", algo="tsvd", dim=3)

        self.assertTrue(True)
