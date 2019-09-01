"""
This file contains the following methods:

remove_columns_threshold
remove_rows_threshold
remove_duplicate_rows
remove_duplicate_columns
replace_missing_random_discrete
"""
import numpy as np
import pandas as pd
from pyautoml.util import _function_input_validation


def remove_columns_threshold(threshold: float, **datasets):
    """
    Remove columns from the dataframe that have more than the threshold value of missing rows.
    Example: Remove columns where > 50% of the data is missing

    Parameters
    ----------
    threshold : float
        Threshold value between 0 and 1 that if the column
        has more than the specified threshold of missing values, it is removed. 

    Either the full data or training data plus testing data MUST be provided, not both.

    data: Dataframe or array like - 2d
        Full dataset, by default None.
    train_data: Dataframe or array like - 2d
        Training dataset, by default None.
    test_data: Dataframe or array like - 2d
        Testing dataset, by default None.
    
    Returns
    -------
    Dataframe, *Dataframe
        Transformed dataframe with rows with a missing values in a specific column are missing

    * Returns 2 Dataframes if Train and Test data is provided.
    """

    data = datasets.pop('data', None)
    train_data = datasets.pop('train_data', None)
    test_data = datasets.pop('test_data', None)

    if threshold > 1 or threshold < 0:
        raise ValueError("Threshold cannot be greater than 1 or less than 0.")

    if datasets:
        raise TypeError("Invalid parameters passed: {}".format(str(datasets)))

    if not _function_input_validation(data, train_data, test_data):
        raise ValueError("Function input is incorrectly provided.")

    if data is not None:
        criteria_meeting_columns = data.columns[data.isnull().mean() < threshold]

        return data[criteria_meeting_columns]

    else:
        criteria_meeting_columns = train_data.columns(train_data.isnull().mean() < threshold)

        return train_data[criteria_meeting_columns], test_data[criteria_meeting_columns]

def remove_rows_threshold(threshold: float, **datasets):
    """
    Remove rows from the dataframe that have more than the threshold value of missing rows.
    Example: Remove rows where > 50% of the data is missing
    
    Parameters
    ----------
    threshold : float
        Threshold value between 0 and 1 that if the row
        has more than the specified threshold of missing values, it is removed. 

    Either the full data or training data plus testing data MUST be provided, not both.

    data: Dataframe or array like - 2d
        Full dataset, by default None.
    train_data: Dataframe or array like - 2d
        Training dataset, by default None.
    test_data: Dataframe or array like - 2d
        Testing dataset, by default None.
    
    Returns
    -------
    Dataframe, *Dataframe
        Transformed dataframe with rows with a missing values in a specific row are missing

    * Returns 2 Dataframes if Train and Test data is provided.
    """

    if threshold > 1 or threshold < 0:
        raise ValueError("Threshold cannot be greater than 1 or less than 0.")

    data = datasets.pop('data', None)
    train_data = datasets.pop('train_data', None)
    test_data = datasets.pop('test_data', None)

    if datasets:
        raise TypeError("Invalid parameters passed: {}".format(str(datasets)))

    if not _function_input_validation(data, train_data, test_data):
        raise ValueError("Function input is incorrectly provided.")

    if data is not None:
        return data.dropna(thresh=round(data.shape[1] * threshold), axis=0)

    else:
        train_data = train_data.dropna(thresh=round(train_data.shape[1] * threshold), axis=0)
        test_data = test_data.dropna(thresh=round(test_data.shape[1] * threshold), axis=0)

        return train_data, test_data

def remove_duplicate_rows(list_of_cols=[], **datasets):
    """
    Removes rows that are exact duplicates of each other.
    
    Parameters
    ----------
    list_of_cols: list, optional
        Columns to check if their values are duplicated, by default []
    
    Either the full data or training data plus testing data MUST be provided, not both.

    data: Dataframe or array like - 2d
        Full dataset, by default None.
    train_data: Dataframe or array like - 2d
        Training dataset, by default None.
    test_data: Dataframe or array like - 2d
        Testing dataset, by default None.
    
    Returns
    -------
    Dataframe, *Dataframe
        Transformed dataframe with rows with a missing values in a specific row are missing

    * Returns 2 Dataframes if Train and Test data is provided.
    """

    data = datasets.pop('data', None)
    train_data = datasets.pop('train_data', None)
    test_data = datasets.pop('test_data', None)

    if datasets:
        raise TypeError("Invalid parameters passed: {}".format(str(datasets)))
        
    if not _function_input_validation(data, train_data, test_data):
        raise ValueError('Function input is incorrectly provided.')

    if data is not None:        
        return data.drop_duplicates(list_of_cols)

    else:
        train_data = train_data.drop_duplicates(list_of_cols)
        test_data = test_data.drop_duplicates(list_of_cols)

        return train_data, test_data

def remove_duplicate_columns(**datasets):
    """
    Removes columns whose values are exact duplicates of each other.
    
    Parameters
    ----------
    Either the full data or training data plus testing data MUST be provided, not both.

    data: Dataframe or array like - 2d
        Full dataset, by default None.
    train_data: Dataframe or array like - 2d
        Training dataset, by default None.
    test_data: Dataframe or array like - 2d
        Testing dataset, by default None.
    
    Returns
    -------
    Dataframe, *Dataframe
        Transformed dataframe with rows with a missing values in a specific row are missing

    * Returns 2 Dataframes if Train and Test data is provided.
    """

    data = datasets.pop('data', None)
    train_data = datasets.pop('train_data', None)
    test_data = datasets.pop('test_data', None)

    if datasets:
        raise TypeError("Invalid parameters passed: {}".format(str(datasets)))
    if not _function_input_validation(data, train_data, test_data):
        raise ValueError('Function input is incorrectly provided.')

    if data is not None:
        return data.T.drop_duplicates().T

    else:
        train_data = train_data.T.drop_duplicates().T
        test_data = test_data.T.drop_duplicates().T

        return train_data, test_data

def replace_missing_random_discrete(list_of_cols: list, **datasets):
    """
    Replace missing values in with a random number based off the distribution (number of occurences) 
    of the data.

    Parameters
    ----------
    list_of_cols : list
        A list of specific columns to apply this technique to, by default []

    Either the full data or training data plus testing data MUST be provided, not both.

    data: Dataframe or array like - 2d
        Full dataset, by default None.
    train_data: Dataframe or array like - 2d
        Training dataset, by default None.
    test_data: Dataframe or array like - 2d
        Testing dataset, by default None.
    
    Returns
    -------
    Dataframe, *Dataframe
        Transformed dataframe with rows with a missing values in a specific row are missing

    * Returns 2 Dataframes if Train and Test data is provided.
    """

    data = datasets.pop('data', None)
    train_data = datasets.pop('train_data', None)
    test_data = datasets.pop('test_data', None)

    if datasets:
        raise TypeError("Invalid parameters passed: {}".format(str(datasets)))
    if not _function_input_validation(data, train_data, test_data):
        raise ValueError('Function input is incorrectly provided.')

    if data is not None:
        for col in list_of_cols:
            probabilities = data[col].value_counts(normalize=True)
            missing_data = data[col].isnull()
            data.loc[missing_data, col] = np.random.choice(probabilities.index, size=len(data[missing_data]), replace=True, p=probabilities.values)

        return data

    else:

        for col in list_of_cols:
            probabilities = train_data[col].value_counts(normalize=True)

            missing_data = train_data[col].isnull()
            train_data.loc[missing_data, col] = np.random.choice(probabilities.index, size=len(train_data[missing_data]), replace=True, p=probabilities.values)

            missing_data = test_data[col].isnull()
            test_data.loc[missing_data, col] = np.random.choice(probabilities.index, size=len(test_data[missing_data]), replace=True, p=probabilities.values)            

        return train_data, test_data
