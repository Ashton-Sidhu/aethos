import pandas as pd
from aethos.util import _get_columns, _numeric_input_conditions, drop_replace_columns
from sklearn.impute import SimpleImputer


def replace_missing_mean_median_mode(
    x_train, x_test=None, list_of_cols=[], strategy=""
):
    """
    Replaces missing values in every numeric column with the mean, median or mode of that column specified by strategy.

    Mean: Average value of the column. Effected by outliers.
    Median: Middle value of a list of numbers. Equal to the mean if x_train follows normal distribution. Not effected much by anomalies.
    Mode: Most common number in a list of numbers.
    
    Parameters
    ----------
    x_train: Dataframe or array like - 2d
        Dataset

    x_test: Dataframe or array like - 2d
        Testing dataset, by default None.

    list_of_cols : list, optional
        A list of specific columns to apply this technique to
        If `list_of_cols` is not provided, the strategy will be
        applied to all numeric columns., by default []

    strategy : str
        Strategy for replacing missing values.
        Can be either "mean", "median" or "most_frequent"
    
    Returns
    -------
    Dataframe, *Dataframe
        Transformed dataframe with rows with a missing values in a specific column are missing

    Returns 2 Dataframes test if x_test is provided.  
    """

    if strategy != "most_frequent":
        list_of_cols = _numeric_input_conditions(list_of_cols, x_train)
    else:
        list_of_cols = _get_columns(list_of_cols, x_train)

    imp = SimpleImputer(strategy=strategy)

    fit_data = imp.fit_transform(x_train[list_of_cols])
    fit_df = pd.DataFrame(fit_data, columns=list_of_cols)
    x_train = drop_replace_columns(x_train, list_of_cols, fit_df)

    if x_test is not None:
        fit_x_test = imp.transform(x_test[list_of_cols])
        fit_test_df = pd.DataFrame(fit_x_test, columns=list_of_cols)
        x_test = drop_replace_columns(x_test, list_of_cols, fit_test_df)

    return x_train, x_test
