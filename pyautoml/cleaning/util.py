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


def remove_columns_threshold(x_train, x_test=None, threshold=0.5):
    """
    Remove columns from the dataframe that have more than the threshold value of missing rows.
    Example: Remove columns where > 50% of the data is missing

    Parameters
    ----------
    x_train: Dataframe or array like - 2d
        Dataset

    x_test: Dataframe or array like - 2d
        Testing dataset, by default None.

    threshold : float
        Threshold value between 0 and 1 that if the column
        has more than the specified threshold of missing values, it is removed. 
        by default, 0.5
    
    Returns
    -------
    Dataframe, *Dataframe
        Transformed dataframe with rows with a missing values in a specific column are missing

    Returns 2 Dataframes if x_test is provided.
    """

    if threshold > 1 or threshold < 0:
        raise ValueError("Threshold cannot be greater than 1 or less than 0.")

    criteria_meeting_columns = x_train.columns[x_train.isnull().mean() < threshold]

    if x_test is not None:
        return x_train[criteria_meeting_columns], x_test[criteria_meeting_columns]
    else:
        return x_train[criteria_meeting_columns], None

def remove_rows_threshold(x_train, x_test=None, threshold=0.5):
    """
    Remove rows from the dataframe that have more than the threshold value of missing rows.
    Example: Remove rows where > 50% of the data is missing
    
    Parameters
    ----------
    x_train: Dataframe or array like - 2d
        Training dataset, by default None.

    x_test: Dataframe or array like - 2d
        Testing dataset, by default None.

    threshold : float
        Threshold value between 0 and 1 that if the row
        has more than the specified threshold of missing values, it is removed.
        by default, 0.5
    
    Returns
    -------
    Dataframe, *Dataframe
        Transformed dataframe with rows with a missing values in a specific row are missing

    Returns 2 Dataframes if x_test is provided.
    """

    if threshold > 1 or threshold < 0:
        raise ValueError("Threshold cannot be greater than 1 or less than 0.")

    x_train = x_train.dropna(thresh=round(x_train.shape[1] * threshold), axis=0)

    if x_test is not None:
        x_test = x_test.dropna(thresh=round(x_test.shape[1] * threshold), axis=0)

    return x_train, x_test

def remove_duplicate_rows(x_train, x_test=None, list_of_cols=[]):
    """
    Removes rows that are exact duplicates of each other.
    
    Parameters
    ----------
    x_train: Dataframe or array like - 2d
        Dataset

    x_test: Dataframe or array like - 2d
        Testing dataset, by default None.
    
    list_of_cols: list, optional
        Columns to check if their values are duplicated, by default []

    Returns
    -------
    Dataframe, *Dataframe
        Transformed dataframe with rows with a missing values in a specific row are missing

    Returns 2 Dataframes if x_test is provided.
    """

    x_train = x_train.drop_duplicates(list_of_cols)

    if x_test is not None:
        x_test = x_test.drop_duplicates(list_of_cols)

    return x_train, x_test

def remove_duplicate_columns(x_train, x_test=None):
    """
    Removes columns whose values are exact duplicates of each other.
    
    Parameters
    ----------
    x_train: Dataframe or array like - 2d
        Dataset.

    x_test: Dataframe or array like - 2d
        Testing dataset, by default None.
    
    Returns
    -------
    Dataframe, *Dataframe
        Transformed dataframe with rows with a missing values in a specific row are missing

    Returns 2 Dataframes if x_test is provided.
    """

    x_train = x_train.T.drop_duplicates().T
    
    if x_test is not None:
        x_test = x_test.T.drop_duplicates().T

    return x_train, x_test

def replace_missing_random_discrete(x_train, x_test=None, list_of_cols=[]):
    """
    Replace missing values in with a random number based off the distribution (number of occurences) 
    of the data.

    Parameters
    ----------
    x_train: Dataframe or array like - 2d
        Dataset
        
    x_test: Dataframe or array like - 2d
        Testing dataset, by default None.
        
    list_of_cols : list
        A list of specific columns to apply this technique to, by default []   
    
    Returns
    -------
    Dataframe, *Dataframe
        Transformed dataframe with rows with a missing values in a specific row are missing

    Returns 2 Dataframes if x_test is provided.
    """


    for col in list_of_cols:
        probabilities = x_train[col].value_counts(normalize=True)

        missing_data = x_train[col].isnull()
        x_train.loc[missing_data, col] = np.random.choice(probabilities.index, size=len(x_train[missing_data]), replace=True, p=probabilities.values)

        if x_test is not None:
            missing_data = x_test[col].isnull()
            x_test.loc[missing_data, col] = np.random.choice(probabilities.index, size=len(x_test[missing_data]), replace=True, p=probabilities.values)            

    return x_train, x_test
