"""
This function has the following methods:

PreprocessNormalize
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from data.util import (DropAndReplaceColumns, _FunctionInputValidation,
                       _NumericFunctionInputConditions)


def PreprocessNormalize(list_of_cols=[], data=None, train_data=None, test_data=None):
    """Function that normalizes all numeric values between 0 and 1 to bring features into same domain.

    Either data or train_data or test_data MUST be provided, not both. 
    
    If `list_of_cols` is not provided, the strategy will be applied to all numeric columns.
    
    Keyword Arguments:
        list_of_cols {list} -- A list of specific columns to apply this technique to. (default: []])
        data {DataFrame} -- Full dataset (default: {None})
        train_data {DataFrame} -- Training dataset (default: {None})
        test_data {DataFrame} -- Testing dataset (default: {None})
        
    Returns:
            [DataFrame],  DataFrame] -- Dataframe(s) missing values replaced by the method. If train and test are provided then the cleaned version 
            of both are returned. 
    """

    if not _FunctionInputValidation(data, train_data, test_data):
        return "Function input is incorrectly provided."

    list_of_cols = _NumericFunctionInputConditions(list_of_cols, data, train_data)
    scaler = MinMaxScaler()

    if data is not None:
        scaled_data = scaler.fit_transform(data[list_of_cols])
        scaled_df = pd.DataFrame(scaled_data, columns=list_of_cols)
        data = DropAndReplaceColumns(data, list_of_cols, scaled_df)
        
        return data
    
    else:
        scaled_train_data = scaler.fit_transform(train_data)
        scaled_train_df = pd.DataFrame(scaled_train_data, columns=list_of_cols)
        train_data = DropAndReplaceColumns(train_data, list_of_cols, scaled_train_df)

        scaled_test_data = scaler.transform(self.data_properties.test_data)
        scaled_test_df = pd.DataFrame(scaled_test_data, columns=list_of_cols)
        test_data = DropAndReplaceColumns(test_data, list_of_cols, scaled_test_df)

        return train_data, test_data
