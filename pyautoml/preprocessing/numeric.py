"""
This function has the following methods:

preprocess_normalize
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from pyautoml.util import (_function_input_validation,
                           _numeric_input_conditions, drop_replace_columns)


def preprocess_normalize(list_of_cols=[], **algo_kwargs):
    """
    Function that normalizes all numeric values between 0 and 1 to bring features into same domain.
    
    Either the full data or training data plus testing data MUST be provided, not both.

    Parameters
    ----------
    list_of_cols : list, optional
        A list of specific columns to apply this technique to
        If `list_of_cols` is not provided, the strategy will be
        applied to all numeric columns, by default []

    algo_kwargs : optional
        Parmaters to pass into MinMaxScaler() constructor
        from Scikit-Learn, by default {}

    data : DataFrame
        Full dataset, by default None

    x_train : DataFrame
        Training dataset, by default None
        
    x_test : DataFrame
        Testing dataset, by default None
    
    Returns
    -------
    Dataframe, *Dataframe
        Transformed dataframe with rows normalized.

    Returns 2 Dataframes if Train and Test data is provided. 
    """

    data = algo_kwargs.pop('data', None)
    x_train = algo_kwargs.pop('x_train', None)
    x_test = algo_kwargs.pop('x_test', None)

    if not _function_input_validation(data, x_train, x_test):
        raise ValueError("Function input is incorrectly provided.")

    list_of_cols = _numeric_input_conditions(list_of_cols, data, x_train)
    scaler = MinMaxScaler(**algo_kwargs)

    if data is not None:
        scaled_data = scaler.fit_transform(data[list_of_cols])
        scaled_df = pd.DataFrame(scaled_data, columns=list_of_cols)
        data = drop_replace_columns(data, list_of_cols, scaled_df)
        
        return data
    
    else:
        scaled_x_train = scaler.fit_transform(x_train)
        scaled_train_df = pd.DataFrame(scaled_x_train, columns=list_of_cols)
        x_train = drop_replace_columns(x_train, list_of_cols, scaled_train_df)

        scaled_x_test = scaler.transform(x_test)
        scaled_test_df = pd.DataFrame(scaled_x_test, columns=list_of_cols)
        x_test = drop_replace_columns(x_test, list_of_cols, scaled_test_df)

        return x_train, x_test
