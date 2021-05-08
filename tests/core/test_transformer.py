from aethosv2.core.transformer import CoreTransformer
import pytest

import pandas as pd

from sklearn.impute import SimpleImputer


@pytest.fixture(scope="module")
def data():
    return pd.DataFrame(
        data=[[1, 2, 3], [1, 4, 6], [2, 5, 7], [float("nan"), 5, 7]],
        columns=["x1", "x2", "x3"],
    )


@pytest.fixture(scope="module")
def ct(data):
    return CoreTransformer(data)


def test_add_transformer(ct: CoreTransformer):

    ct._add_transformer(["x1", "x2"], SimpleImputer())

    ct._add_transformer(["x2"], SimpleImputer(strategy="most_frequent"))

    assert ct.column_transformer["x1"][0].strategy == "mean"
    assert ct.column_transformer["x2"][0].strategy == "mean"
    assert ct.column_transformer["x2"][1].strategy == "most_frequent"
    assert ct.column_transformer["x3"] == []
