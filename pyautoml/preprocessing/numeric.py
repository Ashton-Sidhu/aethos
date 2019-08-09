"""
This function has the following methods:

PreprocessNormalize
"""

import pandas as pd
from pyautoml.util import (DropAndReplaceColumns, _FunctionInputValidation,
                           _NumericFunctionInputConditions)
from sklearn.preprocessing import MinMaxScaler


def PreprocessNormalize(list_of_cols=[], params={}, **datasets):
    """
    Function that normalizes all numeric values between 0 and 1 to bring features into same domain.

    Args:
        list_of_cols (list, optional): A list of specific columns to apply this technique to
                                    If `list_of_cols` is not provided, the strategy will be
                                    applied to all numeric columns. Defaults to [].

        params (dict, optional): A dictionary of parmaters to pass into MinMaxScaler() constructor
                                from Scikit-Learn. Defaults to {}

        Either the full data or training data plus testing data MUST be provided, not both.

        data {DataFrame} -- Full dataset. Defaults to None.
        train_data {DataFrame} -- Training dataset. Defaults to None.
        test_data {DataFrame} -- Testing dataset. Defaults to None.

    Returns:
        Dataframe, *Dataframe: Transformed dataframe with rows with a missing values in a specific column are missing

        * Returns 2 Dataframes if Train and Test data is provided.  
    """

    data = datasets.pop('data', None)
    train_data = datasets.pop('train_data', None)
    test_data = datasets.pop('test_data', None)

    if datasets:
        raise TypeError(f"Invalid parameters passed: {str(datasets)}")    

    if not _FunctionInputValidation(data, train_data, test_data):
        raise ValueError("Function input is incorrectly provided.")

    list_of_cols = _NumericFunctionInputConditions(list_of_cols, data, train_data)
    scaler = MinMaxScaler(**params)

    if data is not None:
        scaled_data = scaler.fit_transform(data[list_of_cols])
        scaled_df = pd.DataFrame(scaled_data, columns=list_of_cols)
        data = DropAndReplaceColumns(data, list_of_cols, scaled_df)
        
        return data
    
    else:
        scaled_train_data = scaler.fit_transform(train_data)
        scaled_train_df = pd.DataFrame(scaled_train_data, columns=list_of_cols)
        train_data = DropAndReplaceColumns(train_data, list_of_cols, scaled_train_df)

        scaled_test_data = scaler.transform(test_data)
        scaled_test_df = pd.DataFrame(scaled_test_data, columns=list_of_cols)
        test_data = DropAndReplaceColumns(test_data, list_of_cols, scaled_test_df)

        return train_data, test_data
