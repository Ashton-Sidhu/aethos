"""
This function has the following methods:

preprocess_normalize
"""

import numpy as np
import pandas as pd
from aethos.util import _numeric_input_conditions, drop_replace_columns
from sklearn.preprocessing import MinMaxScaler, RobustScaler

SCALER = {"minmax": MinMaxScaler, "robust": RobustScaler}


def scale(
    x_train,
    x_test=None,
    list_of_cols=[],
    method="minmax",
    keep_col=False,
    **algo_kwargs
):
    """
    Scales data according to a specific method.

    Parameters
    ----------
    x_train : DataFrame
        Dataset
        
    x_test : DataFrame
        Testing dataset, by default None

    list_of_cols : list, optional
        A list of specific columns to apply this technique to
        If `list_of_cols` is not provided, the strategy will be
        applied to all numeric columns, by default []

    method : str, optional
        Scaling method, by default 'minmax'

    keep_col : bool, optional
        True to not remove the columns, by default False

    algo_kwargs : optional
        Parmaters to pass into the scaler constructor
        from Scikit-Learn, by default {}
    
    Returns
    -------
    Dataframe, *Dataframe
        Transformed dataframe with rows normalized.

    Returns 2 Dataframes if x_test is provided. 
    """

    list_of_cols = _numeric_input_conditions(list_of_cols, x_train)
    scaler = SCALER[method](**algo_kwargs)

    scaled_data = scaler.fit_transform(x_train[list_of_cols])
    scaled_df = pd.DataFrame(scaled_data, columns=list_of_cols)
    x_train = drop_replace_columns(x_train, list_of_cols, scaled_df, keep_col=keep_col)

    if x_test is not None:
        scaled_x_test = scaler.transform(x_test)
        scaled_test_df = pd.DataFrame(scaled_x_test, columns=list_of_cols)
        x_test = drop_replace_columns(
            x_test, list_of_cols, scaled_test_df, keep_col=keep_col
        )

    return x_train, x_test


def log_scale(x_train, x_test=None, list_of_cols=[], base=None):
    """
    Scales data logarithmically.

    Options are '' for natural log, 2 for base2, 10 for base10.
    
    Parameters
    ----------
    x_train : DataFrame
        Dataset
        
    x_test : DataFrame
        Testing dataset, by default None

    list_of_cols : list, optional
        A list of specific columns to apply this technique to
        If `list_of_cols` is not provided, the strategy will be
        applied to all numeric columns, by default []

    base : str, optional
        Base to logarithmically scale by, by default None
    
    Returns
    -------
    Dataframe, *Dataframe
        Transformed dataframe with rows normalized.

    Returns 2 Dataframes if x_test is provided. 
    """

    list_of_cols = _numeric_input_conditions(list_of_cols, x_train)

    if not base:
        log = np.log
    elif base == 2:
        log = np.log2
    elif base == 10:
        log = np.log10
    else:
        log = np.log

    for col in list_of_cols:
        x_train[col] = log(x_train[col])

        if x_test is not None:
            x_test[col] = log(x_test[col])

    return x_train, x_test
