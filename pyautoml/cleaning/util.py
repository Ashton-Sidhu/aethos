"""
This file contains the following methods:

RemoveColumns
RemoveRows
SplitData
"""
import pandas as pd
from pyautoml.util import _FunctionInputValidation


def CheckMissingData(df):
    """Utility function that checks if the data has any missing values.

    Arguemnts:
        df {Dataframe} -- Dataframe of the data
            
    Returns:
        [Boolean] -- True if the data is missing values, False o/w.
    """
    
    return df.isnull().values.any()
    
def RemoveColumns(threshold, data=None, train_data=None, test_data=None):
    """Remove columns from the dataframe that have more than the threshold value of missing columns.
    Example: Remove columns where > 50% of the data is missing
        
    Arguments:
        threshold {[float]} -- Value between 0 and 1 that describes what percentage of a column can be missing values.
        data {DataFrame} -- Full dataset (default: {None})
        train_data {DataFrame} -- Training dataset (default: {None})
        test_data {DataFrame} -- Testing dataset (default: {None})

    Returns:
        [DataFrame],  DataFrame] -- Dataframe(s) missing values replaced by the method. If train and test are provided then the cleaned version 
        of both are returned.     
    """

    if not _FunctionInputValidation(data, train_data, test_data):
        raise ValueError("Function input is incorrectly provided.")

    if data is not None:
        criteria_meeting_columns = data.columns[data.isnull().mean() < threshold]

        return data[criteria_meeting_columns]

    else:
        criteria_meeting_columns = train_data.columns(train_data.isnull().mean() < threshold)

        return train_data[criteria_meeting_columns], test_data[criteria_meeting_columns]

def RemoveRows(threshold, data=None, train_data=None, test_data=None):
    """Remove rows from the dataframe that have more than the threshold value of missing rows.
    Example: Remove rows where > 50% of the data is missing
    
    Arguments:
        threshold {[float]} -- Value between 0 and 1 that describes what percentage of a row can be missing values.
        data {DataFrame} -- Full dataset (default: {None})
        train_data {DataFrame} -- Training dataset (default: {None})
        test_data {DataFrame} -- Testing dataset (default: {None})
        
    Returns:
        [DataFrame],  DataFrame] -- Dataframe(s) missing values replaced by the method. If train and test are provided then the cleaned version 
        of both are returned.     
    """
    if not _FunctionInputValidation(data, train_data, test_data):
        raise ValueError("Function input is incorrectly provided.")

    if data is not None:

        return data.dropna(thresh=round(data.shape[1] * threshold), axis=0)

    else:

        train_data = train_data.dropna(thresh=round(train_data.shape[1] * threshold), axis=0)
        test_data = test_data.dropna(thresh=round(test_data.shape[1] * threshold), axis=0)

        return train_data, test_data
