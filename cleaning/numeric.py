'''
This file contains the following methods:

ReplaceMissingMMM
ReplaceMissingConstant
'''

import pandas as pd
from sklearn.impute import SimpleImputer

from data.util import DropAndReplaceColumns

#TODO: Implement KNN, Interpolation, Extrapolation, Hot-Deck imputation for replacing missing data
#TODO: Add conditional validation based off list_of_cols, data, train_data and test_data input

def ReplaceMissingMMM(strategy, list_of_cols=None, data=None, train_data=None, test_data=None):
    """Replaces missing values in every numeric column with the mean, median or mode of that column specified by strategy.

    Mean: Average value of the column. Effected by outliers.
    Median: Middle value of a list of numbers. Equal to the mean if data follows normal distribution. Not effected much by anomalies.
    Mode: Most common number in a list of numbers.
    
    Keyword Arguments:
        strategy {boolean} -- Can be either "mean", "median" or "most_frequent"
        list_of_cols {list} -- A list of specific columns to apply this technique to. (default: {None})
        data {int or float} -- Numeric value to replace all missing values with (default: {0})
        train_data {int or float} -- Numeric value to replace all missing values with (default: {0})
        test_data {int or float} -- Numeric value to replace all missing values with (default: {0})

    Returns:
        [DataFrame, DataFrame] -- Dataframe(s) missing values replaced by the method. If train and test are provided then the cleaned version 
        of both are returned.              
    """

    if not _FunctionInputValidation(list_of_cols, data, train_data, test_data):
        return "Function input is incorrectly provided."

    list_of_cols = _FunctionInputConditions(list_of_cols, data, train_data, test_data)
    imp = SimpleImputer(strategy=strategy)
    
    if data is not None:                
        fit_data = imp.fit_transform(data[list_of_cols])
        fit_df = pd.DataFrame(fit_data, columns=list_of_cols)
        data = DropAndReplaceColumns(data, list_of_cols, fit_df)

        return data, None
    else:
        fit_train_data = imp.fit_transform(train_data[list_of_cols])
        fit_train_df = pd.DataFrame(fit_data, columns=list_of_cols)            
        train_data = DropAndReplaceColumns(train_data, list_of_cols, fit_train_df)
        
        fit_test_data = imp.transform(test_data[list_of_cols])
        fit_test_df = pd.DataFrame(fit_test_data, columns=list_of_cols)      
        test_data = DropAndReplaceColumns(test_data, list_of_cols, fit_test_df)

        return train_data, test_data

def ReplaceMissingConstant(constant=0, list_of_cols=[], data=None, train_data=None, test_data=None):
    """Replaces missing values in every numeric column with a constant.
    
    Keyword Arguments:
        list_of_cols {list} -- A list of specific columns to apply this technique to.        
        constant {int or float} -- Numeric value to replace all missing values with (default: {0})
        data {int or float} -- Numeric value to replace all missing values with (default: None)
        train_data {int or float} -- Numeric value to replace all missing values with (default: None)
        test_data {int or float} -- Numeric value to replace all missing values with (default: None)

     Returns:
        [DataFrame, DataFrame] -- Dataframe(s) missing values replaced by the method. If train and test are provided then the cleaned version 
        of both are returned.     
    """

    if not _FunctionInputValidation(list_of_cols, data, train_data, test_data):
        return "Function input is incorrectly provided."

    list_of_cols = _FunctionInputConditions(list_of_cols, data, train_data, test_data)

    if data is not None:   
        fit_data = data[list_of_cols].fillna(constant)
        fit_df = pd.DataFrame(fit_data, columns=list_of_cols)             
        data = DropAndReplaceColumns(data, list_of_cols, fit_df)

        return data, None
    else:
        fit_train_data = train_data[list_of_cols].fillna(constant)
        fit_train_df = pd.DataFrame(fit_data, columns=list_of_cols)        
        train_data = DropAndReplaceColumns(train_data, list_of_cols, fit_train_df)
        
        fit_test_data = test_data[list_of_cols].fillna(constant)
        fit_test_df = pd.DataFrame(fit_test_data, columns=list_of_cols)       
        test_data = DropAndReplaceColumns(test_data, list_of_cols, fit_test_df)

        return train_data, test_data

def _FunctionInputValidation(list_of_cols, data, train_data, test_data):
    """
    Helper function to help determine if input is valid.
    """

    if data is not None and (train_data is not None or test_data is not None):
        return False

    if train_data is not None and test_data is None:
        return False

    if test_data is not None and train_data is None:
        return False

    return True


def _FunctionInputConditions(list_of_cols, data, train_data, test_data):
    """
    Helper function to help set variable values of numeric cleaning method functions.
    """

    if list_of_cols:
        list_of_cols = list_of_cols
    else:
        if data is not None:
            list_of_cols = data.columns.tolist()
        else:
            list_of_cols = train_data.columns.tolist()

    return list_of_cols
