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

    Either the full data or training data plus testing data MUST be provided, not both.

    Parameters
    ----------
    threshold : float
        Threshold value between 0 and 1 that if the column
        has more than the specified threshold of missing values, it is removed. 

    data: Dataframe or array like - 2d
        Full dataset, by default None.

    x_train: Dataframe or array like - 2d
        Training dataset, by default None.

    x_test: Dataframe or array like - 2d
        Testing dataset, by default None.
    
    Returns
    -------
    Dataframe, *Dataframe
        Transformed dataframe with rows with a missing values in a specific column are missing

    Returns 2 Dataframes if Train and Test data is provided.
    """

    data = datasets.pop('data', None)
    x_train = datasets.pop('x_train', None)
    x_test = datasets.pop('x_test', None)

    if threshold > 1 or threshold < 0:
        raise ValueError("Threshold cannot be greater than 1 or less than 0.")

    if datasets:
        raise TypeError("Invalid parameters passed: {}".format(str(datasets)))

    if not _function_input_validation(data, x_train, x_test):
        raise ValueError("Function input is incorrectly provided.")

    if data is not None:
        criteria_meeting_columns = data.columns[data.isnull().mean() < threshold]

        return data[criteria_meeting_columns]

    else:
        criteria_meeting_columns = x_train.columns(x_train.isnull().mean() < threshold)

        return x_train[criteria_meeting_columns], x_test[criteria_meeting_columns]

def remove_rows_threshold(threshold: float, **datasets):
    """
    Remove rows from the dataframe that have more than the threshold value of missing rows.
    Example: Remove rows where > 50% of the data is missing
    
    Either the full data or training data plus testing data MUST be provided, not both.

    Parameters
    ----------
    threshold : float
        Threshold value between 0 and 1 that if the row
        has more than the specified threshold of missing values, it is removed. 

    data: Dataframe or array like - 2d
        Full dataset, by default None.

    x_train: Dataframe or array like - 2d
        Training dataset, by default None.

    x_test: Dataframe or array like - 2d
        Testing dataset, by default None.
    
    Returns
    -------
    Dataframe, *Dataframe
        Transformed dataframe with rows with a missing values in a specific row are missing

    Returns 2 Dataframes if Train and Test data is provided.
    """

    if threshold > 1 or threshold < 0:
        raise ValueError("Threshold cannot be greater than 1 or less than 0.")

    data = datasets.pop('data', None)
    x_train = datasets.pop('x_train', None)
    x_test = datasets.pop('x_test', None)

    if datasets:
        raise TypeError("Invalid parameters passed: {}".format(str(datasets)))

    if not _function_input_validation(data, x_train, x_test):
        raise ValueError("Function input is incorrectly provided.")

    if data is not None:
        return data.dropna(thresh=round(data.shape[1] * threshold), axis=0)

    else:
        x_train = x_train.dropna(thresh=round(x_train.shape[1] * threshold), axis=0)
        x_test = x_test.dropna(thresh=round(x_test.shape[1] * threshold), axis=0)

        return x_train, x_test

def remove_duplicate_rows(list_of_cols=[], **datasets):
    """
    Removes rows that are exact duplicates of each other.
    
    Either the full data or training data plus testing data MUST be provided, not both.

    Parameters
    ----------
    list_of_cols: list, optional
        Columns to check if their values are duplicated, by default []

    data: Dataframe or array like - 2d
        Full dataset, by default None.

    x_train: Dataframe or array like - 2d
        Training dataset, by default None.

    x_test: Dataframe or array like - 2d
        Testing dataset, by default None.
    
    Returns
    -------
    Dataframe, *Dataframe
        Transformed dataframe with rows with a missing values in a specific row are missing

    Returns 2 Dataframes if Train and Test data is provided.
    """

    data = datasets.pop('data', None)
    x_train = datasets.pop('x_train', None)
    x_test = datasets.pop('x_test', None)

    if datasets:
        raise TypeError("Invalid parameters passed: {}".format(str(datasets)))
        
    if not _function_input_validation(data, x_train, x_test):
        raise ValueError('Function input is incorrectly provided.')

    if data is not None:        
        return data.drop_duplicates(list_of_cols)

    else:
        x_train = x_train.drop_duplicates(list_of_cols)
        x_test = x_test.drop_duplicates(list_of_cols)

        return x_train, x_test

def remove_duplicate_columns(**datasets):
    """
    Removes columns whose values are exact duplicates of each other.
    
    Either the full data or training data plus testing data MUST be provided, not both.

    Parameters
    ----------
    data: Dataframe or array like - 2d
        Full dataset, by default None.

    x_train: Dataframe or array like - 2d
        Training dataset, by default None.

    x_test: Dataframe or array like - 2d
        Testing dataset, by default None.
    
    Returns
    -------
    Dataframe, *Dataframe
        Transformed dataframe with rows with a missing values in a specific row are missing

    Returns 2 Dataframes if Train and Test data is provided.
    """

    data = datasets.pop('data', None)
    x_train = datasets.pop('x_train', None)
    x_test = datasets.pop('x_test', None)

    if datasets:
        raise TypeError("Invalid parameters passed: {}".format(str(datasets)))
    if not _function_input_validation(data, x_train, x_test):
        raise ValueError('Function input is incorrectly provided.')

    if data is not None:
        return data.T.drop_duplicates().T

    else:
        x_train = x_train.T.drop_duplicates().T
        x_test = x_test.T.drop_duplicates().T

        return x_train, x_test

def replace_missing_random_discrete(list_of_cols: list, **datasets):
    """
    Replace missing values in with a random number based off the distribution (number of occurences) 
    of the data.

    Either the full data or training data plus testing data MUST be provided, not both.

    Parameters
    ----------
    list_of_cols : list
        A list of specific columns to apply this technique to, by default []

    data: Dataframe or array like - 2d
        Full dataset, by default None.

    x_train: Dataframe or array like - 2d
        Training dataset, by default None.
        
    x_test: Dataframe or array like - 2d
        Testing dataset, by default None.
    
    Returns
    -------
    Dataframe, *Dataframe
        Transformed dataframe with rows with a missing values in a specific row are missing

    Returns 2 Dataframes if Train and Test data is provided.
    """

    data = datasets.pop('data', None)
    x_train = datasets.pop('x_train', None)
    x_test = datasets.pop('x_test', None)

    if datasets:
        raise TypeError("Invalid parameters passed: {}".format(str(datasets)))
    if not _function_input_validation(data, x_train, x_test):
        raise ValueError('Function input is incorrectly provided.')

    if data is not None:
        for col in list_of_cols:
            probabilities = data[col].value_counts(normalize=True)
            missing_data = data[col].isnull()
            data.loc[missing_data, col] = np.random.choice(probabilities.index, size=len(data[missing_data]), replace=True, p=probabilities.values)

        return data

    else:

        for col in list_of_cols:
            probabilities = x_train[col].value_counts(normalize=True)

            missing_data = x_train[col].isnull()
            x_train.loc[missing_data, col] = np.random.choice(probabilities.index, size=len(x_train[missing_data]), replace=True, p=probabilities.values)

            missing_data = x_test[col].isnull()
            x_test.loc[missing_data, col] = np.random.choice(probabilities.index, size=len(x_test[missing_data]), replace=True, p=probabilities.values)            

        return x_train, x_test
