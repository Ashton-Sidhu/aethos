"""
This file contains the following methods:

RemoveColumns
RemoveRows
SplitData
"""
import pandas as pd
from pyautoml.util import _FunctionInputValidation


def MissingData(*dataframes):
    """
    Utility function that shows how many values are missing in each column.

    Arguments:
        *dataframes : Sequence of dataframes
    """

    n_arrays = len(dataframes)
    if n_arrays == 0:
        raise ValueError("At least one dataframe required as input")

    
    for dataframe in dataframes:
        if not dataframe.isnull().values.any():            
            yield
        else:
            total = dataframe.isnull().sum().sort_values(ascending=False)
            percent = (dataframe.isnull().sum()/dataframe.isnull().count()).sort_values(ascending=False)
            missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

            yield missing_data
    
def RemoveColumns(threshold, **datasets):
    """
    Remove columns from the dataframe that have more than the threshold value of missing rows.
    Example: Remove columns where > 50% of the data is missing
    
    Args:
        threshold (int or float, optional): Threshold value between 0 and 1 that if the column
        has more than the specified threshold of missing values, it is removed. 
    
        Either the full data or training data plus testing data MUST be provided, not both.

        data {DataFrame} -- Full dataset (default: {None})
        train_data {DataFrame} -- Training dataset (default: {None})
        test_data {DataFrame} -- Testing dataset (default: {None})
    
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

    if data is not None:
        criteria_meeting_columns = data.columns[data.isnull().mean() < threshold]

        return data[criteria_meeting_columns]

    else:
        criteria_meeting_columns = train_data.columns(train_data.isnull().mean() < threshold)

        return train_data[criteria_meeting_columns], test_data[criteria_meeting_columns]

def RemoveRows(threshold, **datasets):
    """
    Remove rows from the dataframe that have more than the threshold value of missing rows.
    Example: Remove rows where > 50% of the data is missing
    
    Args:
        threshold (int or float, optional): Threshold value between 0 and 1 that if the row
        has more than the specified threshold of missing values, it is removed. 
    
        Either the full data or training data plus testing data MUST be provided, not both.

        data {DataFrame} -- Full dataset (default: {None})
        train_data {DataFrame} -- Training dataset (default: {None})
        test_data {DataFrame} -- Testing dataset (default: {None})
    
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

    if data is not None:

        return data.dropna(thresh=round(data.shape[1] * threshold), axis=0)

    else:

        train_data = train_data.dropna(thresh=round(train_data.shape[1] * threshold), axis=0)
        test_data = test_data.dropna(thresh=round(test_data.shape[1] * threshold), axis=0)

        return train_data, test_data
