"""
This file contains the following methods:

replace_missing_mean_median_mode
replace_missing_constant
"""

import pandas as pd
from aethos.cleaning.categorical import replace_missing_new_category
from aethos.util import _get_columns, _numeric_input_conditions, drop_replace_columns
from sklearn.impute import SimpleImputer
import numpy as np


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


def replace_missing_constant(x_train, x_test=None, col_to_constant=None, constant=0):
    """
    Replaces missing values in every numeric column with a constant. If `col_to_constant` is not provided,
    all the missing values in the x_train will be replaced with `constant`
    
    Parameters
    ----------
    col_to_constant : list, dict, optional
        Either a list of columns to replace missing values or a `column`: `value` dictionary mapping,
        by default None

    constant : int, float, optional
        Value to replace missing values with, by default 0

    x_train: Dataframe or array like - 2d
        Training dataset, by default None.
        
    x_test: Dataframe or array like - 2d
        Testing dataset, by default None.
    
    Returns
    -------
    Dataframe, *Dataframe
        Transformed dataframe with rows with a missing values in a specific column are missing

    Returns 2 Dataframes if x_test is provided.  
    
    Examples
    ------
    >>> replace_missing_constant({'a': 1, 'b': 2, 'c': 3})
    >>> replace_missing_constant(1, ['a', 'b', 'c'])
    """

    if isinstance(col_to_constant, dict):
        x_train, x_test = replace_missing_new_category(
            col_to_cateogory=col_to_constant, x_train=x_train, x_test=x_test
        )
    elif isinstance(col_to_constant, list):
        x_train, x_test = replace_missing_new_category(
            constant=constant,
            col_to_category=col_to_constant,
            x_train=x_train,
            x_test=x_test,
        )
    else:
        x_train, x_test = replace_missing_new_category(
            constant=constant, x_train=x_train, x_test=x_test
        )

    return x_train, x_test
