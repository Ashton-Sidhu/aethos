'''
This file contains the following methods:

ReplaceMissingMeanMedianMode
ReplaceMissingConstant
'''

import pandas as pd
from sklearn.impute import SimpleImputer

from pyautoml.cleaning.categorical import ReplaceMissingNewCategory
from pyautoml.util import (DropAndReplaceColumns, _FunctionInputValidation,
                           _NumericFunctionInputConditions)

#TODO: Implement KNN, Interpolation, Extrapolation, Hot-Deck imputation for replacing missing data

def ReplaceMissingMeanMedianMode(strategy, list_of_cols=[], data=None, train_data=None, test_data=None):
    """Replaces missing values in every numeric column with the mean, median or mode of that column specified by strategy.

    Either data or train_data or test_data MUST be provided, not both. 
    
    If `list_of_cols` is not provided, the strategy will be applied to all numeric columns.

    Mean: Average value of the column. Effected by outliers.
    Median: Middle value of a list of numbers. Equal to the mean if data follows normal distribution. Not effected much by anomalies.
    Mode: Most common number in a list of numbers.
    
    Keyword Arguments:
        strategy {boolean} -- Can be either "mean", "median" or "most_frequent"
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
    imp = SimpleImputer(strategy=strategy)
    
    if data is not None:                
        fit_data = imp.fit_transform(data[list_of_cols])
        fit_df = pd.DataFrame(fit_data, columns=list_of_cols)
        data = DropAndReplaceColumns(data, list_of_cols, fit_df)

        return data
    else:
        fit_train_data = imp.fit_transform(train_data[list_of_cols])
        fit_train_df = pd.DataFrame(fit_data, columns=list_of_cols)            
        train_data = DropAndReplaceColumns(train_data, list_of_cols, fit_train_df)
        
        fit_test_data = imp.transform(test_data[list_of_cols])
        fit_test_df = pd.DataFrame(fit_test_data, columns=list_of_cols)      
        test_data = DropAndReplaceColumns(test_data, list_of_cols, fit_test_df)

        return train_data, test_data

def ReplaceMissingConstant(constant=0, col_to_constant=None, data=None, train_data=None, test_data=None):
    """Replaces missing values in every numeric column with a constant. If `col_to_constant` is not provided,
    all the missing values in the data will be replaced with `constant`

    Either data or train_data or test_data MUST be provided, not both. 

    The underlying function resides in `clean/categorical.py`
    
    Keyword Arguments:
        col_to_constant {list} or {dict}  -- A list of specific columns to apply this technique to.        
        constant {int or float} -- Numeric value to replace all missing values with (default: 0)
        data {DataFrame} -- Full dataset (default: {None})
        train_data {DataFrame} -- Training dataset (default: {None})
        test_data {DataFrame} -- Testing dataset (default: {None})

     Returns:
        [DataFrame], DataFrame] -- Dataframe(s) missing values replaced by the method. If train and test are provided then the cleaned version 
        of both are returned.     
    """

    if not _FunctionInputValidation(data, train_data, test_data):
        return "Function input is incorrectly provided."

    if isinstance(col_to_constant, dict):
        if data is not None:

            data = ReplaceMissingNewCategory(col_to_constant, data=data)

            return data
        
        else:

            train_data, test_data = ReplaceMissingNewCategory(col_to_constant, train_data=train_data, test_data=test_data)

            return train_data, test_data

    elif isinstance(col_to_constant, list):

        if data is not None:

            data = ReplaceMissingNewCategory(col_to_constant, constant=constant, data=data)

            return data
        
        else:

            train_data, test_data = ReplaceMissingNewCategory(col_to_constant, constant=constant, train_data=train_data, test_data=test_data)

            return train_data, test_data

    else:

        if data is not None:

            data = ReplaceMissingNewCategory(constant=constant, data=data)

            return data
        
        else:

            train_data, test_data = ReplaceMissingNewCategory(constant=constant, train_data=train_data, test_data=test_data)

            return train_data, test_data
