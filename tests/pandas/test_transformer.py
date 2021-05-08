import pytest
import numpy as np
import pandas as pd

from aethosv2.pandas.transformer import PandasTransformer


@pytest.fixture(scope="module")
def data():
    return pd.DataFrame(
        data=[[1, 2, 3], [1, 4, 6], [2, 5, 7], [float("nan"), 5, 7]],
        columns=["x1", "x2", "x3"],
    )


def test_replace_missing_most_common(data):

    pipe = PandasTransformer(data)

    pipe.replace_missing_most_common("x1", "x2")

    assert pipe.pipeline.steps[0][1].transformers[0][0] == "simpleimputer"
    assert pipe.pipeline.steps[0][1].transformers[0][2] == ("x1", "x2")


def test_run(data):

    pipe = PandasTransformer(data)

    pipe.replace_missing_most_common("x1")

    df = pipe.run()

    assert df.tolist() == [[1], [1], [2], [1]]
