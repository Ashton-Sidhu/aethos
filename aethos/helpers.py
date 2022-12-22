import pandas as pd

from .analysis import Analysis
from aethos.util import _get_columns

def groupby_analysis(df: pd.DataFrame, groupby: list, col_filter=[]):
    """
    Groups your data and then provides descriptive statistics for the other columns on the grouped data.

    For numeric data, the descriptive statistics are:

        - count
        - min
        - max
        - mean
        - std (standard deviation)
        - var (variance)
        - median
        - most_common
        - sum
        - mad (Median Absolute Deviation)
        - nunique (number of unique values)

    For other types of data:

        - count
        - most_common
        - nunique (number of unique values)
    
    Parameters
    ----------
    df : pd.DataFrame
        Data to groupby.

    groupby : list
        List of columns to groupby.

    col_filter : list
        Columns to calculated aggregate metrics of, by default all

    Returns
    -------
    Analysis
        Aethos Analysis object to analyze data

    Examples
    --------
    >>> analysis = data.groupby_analysis(df, ["col1", "col2"])
    >>> analysis = data.groupby_analysis(df, ['col1', 'col2'], metrics=["mean", "sum", "var"])
    """

    analysis = {}
    numeric_analysis = [
        "count",
        "min",
        "max",
        "mean",
        "std",
        "var",
        "median",
        ("most_common", lambda x: pd.Series.mode(x)[0]),
        "sum",
        "mad",
        "nunique",
    ]
    other_analysis = [
        "count",
        ("most_common", lambda x: pd.Series.mode(x)[0]),
        "nunique",
    ]

    list_of_cols = _get_columns(col_filter, df)

    for col in list_of_cols:
        if col not in groupby:
            # biufc - bool, int, unsigned, float, complex
            if df[col].dtype.kind in "biufc":
                analysis[col] = numeric_analysis
            else:
                analysis[col] = other_analysis

    analyzed_data = df.groupby(groupby).agg(analysis)

    return Analysis(analyzed_data)
